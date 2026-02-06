# Docker setup (AirLLM + architecture-aware extension)

This guide installs **AirLLM** and this repository's `moe_support` helpers inside a Docker container.

## 1) Build the image

```bash
docker build -t airllm-moe-support:latest .
```

## 2) Run an interactive container

```bash
docker run --rm -it \
  -e HF_TOKEN=${HF_TOKEN:-} \
  airllm-moe-support:latest
```

> If you need GPU access, use NVIDIA runtime flags on a host with the NVIDIA Container Toolkit:

```bash
docker run --rm -it --gpus all \
  -e HF_TOKEN=${HF_TOKEN:-} \
  airllm-moe-support:latest
```

## 3) Verify install inside container

```bash
python -c "import airllm, transformers, huggingface_hub; print('ok')"
```

## 4) Verify this extension tests

```bash
PYTHONPATH=/app pytest -q
```

## 5) Use the patch with AirLLM

```python
from moe_support import patch_airllm_automodel_from_pretrained
from airllm import AutoModel

patch_airllm_automodel_from_pretrained()

model = AutoModel.from_pretrained("zai-org/GLM-4.7", hf_token=None)
```

## Notes

- Some gated/private repos require `HF_TOKEN`.
- Runtime behavior still depends on AirLLM backend compatibility for the selected architecture.
- For production GPU workloads, prefer a CUDA base image with matching PyTorch/CUDA stack.
