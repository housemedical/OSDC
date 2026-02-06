# Architecture-Aware Support Extensions for AirLLM

This extension avoids per-model hardcoding by inferring compatibility from Hugging Face metadata.

## What it does

- Detects model family from `architecture` + `model_type` + repo id.
- Maps detected family to an existing AirLLM backend class.
- Infers layer naming schema (`embed`, `layer_prefix`, `norm`, `lm_head`) from shard index keys.
- Optionally patches `airllm.AutoModel.from_pretrained` so detection/schema inference happen automatically.

## Why this helps for diverse author architectures

Different model authors often change class names and internal roots (e.g. `language_model.model.layers` vs `model.layers` vs `transformer.h`).
This module uses generic rules and weight-map analysis so AirLLM can adapt to those variants without one-off model IDs.

## Usage

```python
from moe_support import patch_airllm_automodel_from_pretrained
from airllm import AutoModel

# Enable architecture-aware loading.
patch_airllm_automodel_from_pretrained()

# Then load any model repo as usual.
model = AutoModel.from_pretrained("moonshotai/Kimi-K2.5")
```

## Public API

- `detect_airllm_class_name(...)`
- `infer_layer_names_dict(...)`
- `inspect_hf_architecture(...)`
- `resolve_airllm_target(...)`
- `patch_airllm_automodel_from_pretrained()`

## Notes

- This improves automatic routing and layer-schema compatibility significantly.
- Some model families may still need architecture-specific forward/cache behavior inside AirLLM core if their runtime semantics diverge from the chosen backend class.


## Docker

A ready-to-use container setup is included at repository root:
- `Dockerfile`
- `DOCKER_SETUP.md`

Quick start:

```bash
docker build -t airllm-moe-support:latest .
docker run --rm -it -e HF_TOKEN=${HF_TOKEN:-} airllm-moe-support:latest
```
