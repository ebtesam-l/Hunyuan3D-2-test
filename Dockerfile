# Hunyuan3D-2 web app — built for EC2 g4dn.2xlarge (NVIDIA T4, 16 GB VRAM)
# Build:  docker build -t hunyuan3d-webapp .
# Run:    docker run --gpus all -p 8080:8080 -v hy3d-cache:/root/.cache hunyuan3d-webapp

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    # T4 = compute capability 7.5 (needed to compile the CUDA rasterizer without a GPU present at build time)
    TORCH_CUDA_ARCH_LIST="7.5"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip git \
    libgl1 libglu1-mesa libglib2.0-0 libxrender1 libxext6 libsm6 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch with CUDA 12.1
RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu121

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Compile texture-generation extensions (CUDA rasterizer + differentiable renderer)
RUN cd hy3dgen/texgen/custom_rasterizer && python3 setup.py install && \
    cd ../differentiable_renderer && python3 setup.py install

# Model weights (~15 GB: Hunyuan3D-2mini shape + Hunyuan3D-2 paint) are downloaded
# from Hugging Face on first start and cached; mount /root/.cache as a volume to persist.

ENV HY3D_SHAPE_MODEL=tencent/Hunyuan3D-2mini \
    HY3D_SHAPE_SUBFOLDER=hunyuan3d-dit-v2-mini-turbo \
    HY3D_TEX_MODEL=tencent/Hunyuan3D-2 \
    HY3D_STEPS=5 \
    HY3D_LOW_VRAM=1

EXPOSE 8080
CMD ["python3", "-m", "uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "8080"]
