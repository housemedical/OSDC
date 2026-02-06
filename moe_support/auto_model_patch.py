"""Architecture-aware AirLLM routing utilities.

This module extends AirLLM loading in a model-family-agnostic way:
- Detects likely AirLLM backend class from Hugging Face architecture/model_type.
- Infers layer naming schema from HF shard index keys (no model weights load needed).
- Optionally monkey-patches `airllm.AutoModel.from_pretrained` to use both.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_LAYER_NAMES = {
    "embed": "model.embed_tokens",
    "layer_prefix": "model.layers",
    "norm": "model.norm",
    "lm_head": "lm_head",
}


@dataclass(frozen=True)
class ArchitectureInfo:
    architecture: str
    model_type: str
    repo_id: str


def _normalize_tokens(*parts: str) -> str:
    return " ".join((p or "").lower() for p in parts)


def detect_airllm_class_name(
    architecture: str,
    model_type: str,
    repo_id: str,
    default: str = "AirLLMLlama2",
) -> str:
    """Map HF architecture/model_type/repo to best AirLLM class using generic rules."""
    normalized = _normalize_tokens(architecture, model_type, repo_id)

    rules: Sequence[Tuple[Tuple[str, ...], str]] = (
        (("qwen2", "qwen3", "qwen2moe", "qwenmoe"), "AirLLMQWen2"),
        (("qwen",), "AirLLMQWen"),
        (("baichuan",), "AirLLMBaichuan"),
        (("chatglm", "glm4", "glm-", "glm_"), "AirLLMChatGLM"),
        (("internlm", "interns1", "intern-s1"), "AirLLMInternLM"),
        (("mixtral",), "AirLLMMixtral"),
        (("mistral",), "AirLLMMistral"),
        (("llama",), "AirLLMLlama2"),
    )

    for keywords, class_name in rules:
        if any(k in normalized for k in keywords):
            return class_name
    return default


def _first_match(keys: Iterable[str], suffixes: Sequence[str]) -> Optional[str]:
    for key in keys:
        for suffix in suffixes:
            if key.endswith(suffix):
                return key[: -len(".weight")] if key.endswith(".weight") else key
    return None


def infer_layer_names_dict(weight_map_keys: Sequence[str]) -> Dict[str, str]:
    """Infer AirLLM layer name schema from shard index weight-map keys."""
    if not weight_map_keys:
        return dict(DEFAULT_LAYER_NAMES)

    # 1) infer repeating transformer block prefix (e.g. model.layers / transformer.h)
    prefix_counts: Dict[str, int] = {}
    for key in weight_map_keys:
        m = re.match(r"^(.*)\.(\d+)\.[^.]+", key)
        if not m:
            continue
        prefix = m.group(1)
        idx = int(m.group(2))
        if idx < 0:
            continue
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1

    layer_prefix = max(prefix_counts, key=prefix_counts.get) if prefix_counts else DEFAULT_LAYER_NAMES["layer_prefix"]

    embed = _first_match(
        weight_map_keys,
        (
            "embed_tokens.weight",
            "tok_embeddings.weight",
            "word_embeddings.weight",
            "wte.weight",
        ),
    ) or DEFAULT_LAYER_NAMES["embed"]

    norm = _first_match(
        weight_map_keys,
        (
            "model.norm.weight",
            "norm.weight",
            "final_layernorm.weight",
            "ln_f.weight",
        ),
    )
    if norm is None:
        # fallback: try any non-layer norm under same root
        root = layer_prefix.rsplit(".", 1)[0] if "." in layer_prefix else ""
        for key in weight_map_keys:
            if key.endswith("norm.weight") and ".layers." not in key and ".h." not in key:
                norm = key[: -len(".weight")]
                break
        if norm is None:
            norm = f"{root}.norm" if root else DEFAULT_LAYER_NAMES["norm"]

    lm_head = _first_match(weight_map_keys, ("lm_head.weight", "output.weight", "embed_out.weight")) or DEFAULT_LAYER_NAMES[
        "lm_head"
    ]

    return {
        "embed": embed,
        "layer_prefix": layer_prefix,
        "norm": norm,
        "lm_head": lm_head,
    }


def fetch_weight_map_keys(repo_id: str, hf_token: Optional[str] = None) -> List[str]:
    """Fetch shard index keys from HF Hub without downloading full checkpoints."""
    from huggingface_hub import hf_hub_download

    last_error: Optional[Exception] = None
    for filename in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        try:
            path = hf_hub_download(repo_id=repo_id, filename=filename, token=hf_token)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return list(data.get("weight_map", {}).keys())
        except Exception as exc:  # pragma: no cover - network/auth dependent
            last_error = exc

    if last_error is not None:
        raise last_error
    return []


def inspect_hf_architecture(repo_id: str, hf_token: Optional[str] = None) -> ArchitectureInfo:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(repo_id, trust_remote_code=True, token=hf_token)
    architecture = cfg.architectures[0] if getattr(cfg, "architectures", None) else ""
    model_type = getattr(cfg, "model_type", "") or ""
    return ArchitectureInfo(architecture=architecture, model_type=model_type, repo_id=repo_id)


def resolve_airllm_target(repo_id: str, hf_token: Optional[str] = None) -> Tuple[str, str, Dict[str, str]]:
    """Return `(module_name, class_name, inferred_layer_names_dict)` for a HF repo."""
    info = inspect_hf_architecture(repo_id=repo_id, hf_token=hf_token)
    class_name = detect_airllm_class_name(info.architecture, info.model_type, info.repo_id)

    try:
        keys = fetch_weight_map_keys(repo_id=repo_id, hf_token=hf_token)
        layer_names = infer_layer_names_dict(keys)
    except Exception:  # pragma: no cover - network/auth dependent
        layer_names = dict(DEFAULT_LAYER_NAMES)

    return "airllm", class_name, layer_names


def _make_layer_schema_subclass(base_cls, layer_names: Dict[str, str]):
    class AirLLMDynamicSchema(base_cls):
        def set_layer_names_dict(self):
            self.layer_names_dict = dict(layer_names)

    AirLLMDynamicSchema.__name__ = f"{base_cls.__name__}DynamicSchema"
    return AirLLMDynamicSchema


def patch_airllm_automodel_from_pretrained() -> None:
    """Patch `airllm.AutoModel.from_pretrained` with architecture-aware routing + schema inference."""
    try:
        import airllm
        from sys import platform
    except Exception as exc:  # pragma: no cover - import environment dependent
        raise RuntimeError("airllm must be installed to patch AutoModel") from exc

    auto_model_cls = airllm.AutoModel
    original_from_pretrained = auto_model_cls.from_pretrained

    @classmethod
    def _patched_from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        if platform == "darwin":
            # Keep upstream macOS behavior.
            return original_from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

        module_name, class_name, layer_names = resolve_airllm_target(
            repo_id=str(pretrained_model_name_or_path),
            hf_token=kwargs.get("hf_token"),
        )

        import importlib
        module = importlib.import_module(module_name)
        base_class = getattr(module, class_name)
        dynamic_class = _make_layer_schema_subclass(base_class, layer_names)
        return dynamic_class(pretrained_model_name_or_path, *inputs, **kwargs)

    auto_model_cls.from_pretrained = _patched_from_pretrained


__all__ = [
    "ArchitectureInfo",
    "DEFAULT_LAYER_NAMES",
    "detect_airllm_class_name",
    "infer_layer_names_dict",
    "fetch_weight_map_keys",
    "inspect_hf_architecture",
    "resolve_airllm_target",
    "patch_airllm_automodel_from_pretrained",
]
