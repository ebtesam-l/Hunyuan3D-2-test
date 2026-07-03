# Deploying the Hunyuan3D-2 web app on EC2 g4dn.2xlarge

## What's here

- `webapp/app.py` — FastAPI backend (upload image → rembg → shape gen → cleanup → texture → GLB)
- `webapp/static/index.html` — single-page UI (three.js viewer, render snapshot, downloads)
- `Dockerfile` — at repo root, CUDA 12.1, T4-ready

Models: `Hunyuan3D-2mini-turbo` (shape, 5 steps) + `Hunyuan3D-2 paint-turbo` (texture), with CPU offload for the 16 GB T4.

## EC2 setup (once)

Launch **g4dn.2xlarge** with the **AWS Deep Learning Base GPU AMI (Ubuntu 22.04)** — it ships with NVIDIA driver + Docker + nvidia-container-toolkit. Use **≥ 100 GB** gp3 root volume (image ~20 GB + ~15 GB weights + jobs).

If using a plain Ubuntu AMI instead:

```bash
sudo apt update && sudo apt install -y docker.io
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-driver-535 nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

Security group: allow inbound TCP 8080 (or 80 if you map it).

## Build & run

```bash
git clone <your repo> Hunyuan3D-2 && cd Hunyuan3D-2   # or scp this folder up
sudo docker build -t hunyuan3d-webapp .

sudo docker run -d --gpus all --restart unless-stopped \
  -p 8080:8080 \
  -v hy3d-cache:/root/.cache \
  --name hunyuan3d hunyuan3d-webapp
```

Open `http://<ec2-public-ip>:8080`. First start downloads ~15 GB of weights (watch with `sudo docker logs -f hunyuan3d`); the page header shows "loading models…" until ready. The cache volume makes restarts fast.

## Expected performance (T4)

- Shape (mini-turbo, 5 steps): ~15–30 s
- Texture: ~2–5 min (the slow part; CPU offload trades speed for fitting in 16 GB)
- Jobs are processed one at a time (single GPU queue).

## Tuning (env vars on `docker run -e`)

| Var | Default | Notes |
|---|---|---|
| `HY3D_SHAPE_MODEL` / `HY3D_SHAPE_SUBFOLDER` | `tencent/Hunyuan3D-2mini` / `hunyuan3d-dit-v2-mini-turbo` | Set to `tencent/Hunyuan3D-2` / `hunyuan3d-dit-v2-0-turbo` for higher quality |
| `HY3D_STEPS` | 5 | Use 30 for non-turbo models |
| `HY3D_OCTREE_RES` | 256 | 384 = finer geometry, slower |
| `HY3D_MAX_FACES` | 40000 | Face count after reduction |
| `HY3D_LOW_VRAM` | 1 | Keep 1 on T4 |

Note: Hunyuan3D-2 weights are under the Tencent Hunyuan non-commercial license — check it before commercial deployment.
