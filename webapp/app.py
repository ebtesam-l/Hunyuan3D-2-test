# Single-page web app for Hunyuan3D-2: image -> textured 3D model (GLB)
# Backend: FastAPI + background worker thread (GPU jobs run serially).
# Run from repo root:  uvicorn webapp.app:app --host 0.0.0.0 --port 8080

import io
import logging
import os
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path
from queue import Queue

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("hy3d-webapp")

ROOT = Path(__file__).resolve().parent.parent          # repo root
STATIC_DIR = Path(__file__).resolve().parent / "static"
EXAMPLES_DIR = ROOT / "assets" / "example_images"
JOBS_DIR = Path(os.environ.get("HY3D_JOBS_DIR", "/tmp/hy3d_jobs"))
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Config (env-overridable) ----
SHAPE_MODEL = os.environ.get("HY3D_SHAPE_MODEL", "tencent/Hunyuan3D-2mini")
SHAPE_SUBFOLDER = os.environ.get("HY3D_SHAPE_SUBFOLDER", "hunyuan3d-dit-v2-mini-turbo")
TEX_MODEL = os.environ.get("HY3D_TEX_MODEL", "tencent/Hunyuan3D-2")
STEPS = int(os.environ.get("HY3D_STEPS", "5"))          # 5 for turbo models
GUIDANCE = float(os.environ.get("HY3D_GUIDANCE", "5.0"))
OCTREE_RES = int(os.environ.get("HY3D_OCTREE_RES", "256"))
NUM_CHUNKS = int(os.environ.get("HY3D_NUM_CHUNKS", "8000"))
MAX_FACES = int(os.environ.get("HY3D_MAX_FACES", "40000"))
LOW_VRAM = os.environ.get("HY3D_LOW_VRAM", "1") == "1"  # keep on for T4 16GB

app = FastAPI(title="Hunyuan3D-2 Web App")

# ---- State ----
MODELS = {}
MODELS_READY = threading.Event()
MODELS_ERROR = [None]
JOBS = {}           # job_id -> dict(status, stage, error, created)
JOB_QUEUE = Queue()

STAGES = [
    "queued",
    "removing background",
    "generating shape",
    "cleaning mesh",
    "painting texture",
    "exporting",
    "done",
]


def load_models():
    try:
        from hy3dgen.rembg import BackgroundRemover
        from hy3dgen.shapegen import (DegenerateFaceRemover, FaceReducer,
                                      FloaterRemover,
                                      Hunyuan3DDiTFlowMatchingPipeline)
        from hy3dgen.texgen import Hunyuan3DPaintPipeline

        logger.info("Loading background remover...")
        MODELS["rembg"] = BackgroundRemover()

        logger.info("Loading shape pipeline %s/%s ...", SHAPE_MODEL, SHAPE_SUBFOLDER)
        MODELS["shapegen"] = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            SHAPE_MODEL, subfolder=SHAPE_SUBFOLDER
        )

        logger.info("Loading texture pipeline %s ...", TEX_MODEL)
        MODELS["texgen"] = Hunyuan3DPaintPipeline.from_pretrained(TEX_MODEL)
        if LOW_VRAM:
            try:
                MODELS["texgen"].enable_model_cpu_offload()
                logger.info("Texture pipeline: CPU offload enabled (low VRAM mode).")
            except Exception as e:
                logger.warning("CPU offload not available: %s", e)

        MODELS["floater"] = FloaterRemover()
        MODELS["degenerate"] = DegenerateFaceRemover()
        MODELS["facereduce"] = FaceReducer()
        MODELS_READY.set()
        logger.info("All models loaded.")
    except Exception as e:
        MODELS_ERROR[0] = f"{e}"
        logger.error("Model loading failed:\n%s", traceback.format_exc())


def run_job(job_id: str):
    job = JOBS[job_id]
    job_dir = JOBS_DIR / job_id
    try:
        # 1. Background removal
        job["stage"] = "removing background"
        image = Image.open(job_dir / "original.png")
        has_alpha = image.mode == "RGBA" and image.getextrema()[3][0] < 255
        if not has_alpha:
            image = MODELS["rembg"](image.convert("RGB"))
        else:
            image = image.convert("RGBA")
        image.save(job_dir / "input.png")

        seed = job["seed"]
        generator = torch.Generator().manual_seed(seed)

        # 2. Shape generation
        job["stage"] = "generating shape"
        t0 = time.time()
        mesh = MODELS["shapegen"](
            image=image,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            octree_resolution=OCTREE_RES,
            num_chunks=NUM_CHUNKS,
            generator=generator,
            output_type="trimesh",
        )[0]
        logger.info("[%s] shape: %.1fs, %d faces", job_id, time.time() - t0, len(mesh.faces))

        # 3. Mesh cleanup
        job["stage"] = "cleaning mesh"
        mesh = MODELS["floater"](mesh)
        mesh = MODELS["degenerate"](mesh)
        mesh = MODELS["facereduce"](mesh, max_facenum=MAX_FACES)

        # 4. Texture
        if job["texture"]:
            job["stage"] = "painting texture"
            t0 = time.time()
            mesh = MODELS["texgen"](mesh, image)
            logger.info("[%s] texture: %.1fs", job_id, time.time() - t0)

        # 5. Export
        job["stage"] = "exporting"
        mesh.export(str(job_dir / "model.glb"), include_normals=job["texture"])

        job["stage"] = "done"
        job["status"] = "done"
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        logger.error("[%s] failed:\n%s", job_id, traceback.format_exc())
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def worker():
    load_models()
    while True:
        job_id = JOB_QUEUE.get()
        if not MODELS_READY.is_set():
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = MODELS_ERROR[0] or "models not loaded"
            continue
        JOBS[job_id]["status"] = "running"
        run_job(job_id)


threading.Thread(target=worker, daemon=True).start()


# ---- API ----
@app.get("/api/health")
def health():
    return {
        "ready": MODELS_READY.is_set(),
        "error": MODELS_ERROR[0],
        "queue": JOB_QUEUE.qsize(),
    }


@app.get("/api/examples")
def examples():
    if not EXAMPLES_DIR.exists():
        return {"images": []}
    names = sorted(p.name for p in EXAMPLES_DIR.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))[:24]
    return {"images": [f"/examples/{n}" for n in names]}


@app.post("/api/generate")
async def generate(
    image: UploadFile = File(...),
    seed: int = Form(1234),
    texture: bool = Form(True),
):
    if MODELS_ERROR[0]:
        raise HTTPException(500, f"Model loading failed: {MODELS_ERROR[0]}")
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)
    data = await image.read()
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(400, "Image too large (max 20 MB)")
    try:
        img = Image.open(io.BytesIO(data))
        img.save(job_dir / "original.png")
    except Exception:
        raise HTTPException(400, "Invalid image file")

    JOBS[job_id] = {
        "status": "queued", "stage": "queued", "error": None,
        "seed": seed, "texture": texture, "created": time.time(),
    }
    JOB_QUEUE.put(job_id)
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")
    return {
        "status": job["status"],
        "stage": job["stage"],
        "stage_index": STAGES.index(job["stage"]) if job["stage"] in STAGES else 0,
        "num_stages": len(STAGES),
        "error": job["error"],
        "models_ready": MODELS_READY.is_set(),
    }


@app.get("/api/result/{job_id}/{filename}")
def result(job_id: str, filename: str):
    if filename not in ("model.glb", "input.png", "original.png"):
        raise HTTPException(404, "Unknown file")
    path = JOBS_DIR / job_id / filename
    if not path.exists():
        raise HTTPException(404, "Not found")
    media = "model/gltf-binary" if filename.endswith(".glb") else "image/png"
    return FileResponse(path, media_type=media, filename=f"hunyuan3d_{job_id}_{filename}")


if EXAMPLES_DIR.exists():
    app.mount("/examples", StaticFiles(directory=str(EXAMPLES_DIR)), name="examples")
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
