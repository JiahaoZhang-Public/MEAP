from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Dict, Optional

import torch

from .backend import HFLLMBackend, inspect_model_architecture


@dataclass
class HFLoadedArtifacts:
    model: torch.nn.Module
    processor: Any
    backend: HFLLMBackend
    model_id_or_path: str
    device: torch.device
    dtype: torch.dtype
    from_cache: bool


_HF_CACHE: Dict[str, HFLoadedArtifacts] = {}


def _normalize_model_ref(model_id_or_path: str) -> str:
    if os.path.isdir(model_id_or_path):
        return os.path.abspath(model_id_or_path)
    return model_id_or_path


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    raise TypeError(f"device must be str|torch.device, got {type(device)}")


def _resolve_dtype(dtype: str | torch.dtype, device: torch.device) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if not isinstance(dtype, str):
        raise TypeError(f"dtype must be str|torch.dtype, got {type(dtype)}")

    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype}'. Expected one of {sorted(mapping)} or 'auto'.")
    return mapping[dtype]


def _cache_key(
    *,
    model_id_or_path: str,
    device: torch.device,
    dtype: torch.dtype,
    adapter_name: Optional[str],
    language_trunk_path: Optional[str],
    strict_arch: bool,
    model_kwargs: Optional[Dict[str, Any]],
) -> str:
    kwargs_key = json.dumps(model_kwargs or {}, sort_keys=True, default=str)
    return "|".join(
        [
            _normalize_model_ref(model_id_or_path),
            str(device),
            str(dtype),
            str(adapter_name),
            str(language_trunk_path),
            str(strict_arch),
            kwargs_key,
        ]
    )


def clear_hf_loader_cache() -> None:
    _HF_CACHE.clear()


def _fallback_processor(model_id_or_path: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_processor(model_id_or_path: str):
    from transformers import AutoProcessor

    try:
        return AutoProcessor.from_pretrained(model_id_or_path)
    except Exception:
        return _fallback_processor(model_id_or_path)


def _load_model(model_id_or_path: str, *, model_kwargs: Optional[Dict[str, Any]]):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(model_id_or_path, **(model_kwargs or {}))


def load_hf_backend_and_processor(
    *,
    model_id_or_path: str,
    device: str | torch.device = "auto",
    dtype: str | torch.dtype = "auto",
    model_kwargs: Optional[Dict[str, Any]] = None,
    adapter_name: str | None = None,
    language_trunk_path: str | None = None,
    strict_arch: bool = True,
    cache: bool = True,
) -> HFLoadedArtifacts:
    resolved_device = _resolve_device(device)
    resolved_dtype = _resolve_dtype(dtype, resolved_device)
    normalized_ref = _normalize_model_ref(model_id_or_path)

    key = _cache_key(
        model_id_or_path=normalized_ref,
        device=resolved_device,
        dtype=resolved_dtype,
        adapter_name=adapter_name,
        language_trunk_path=language_trunk_path,
        strict_arch=strict_arch,
        model_kwargs=model_kwargs,
    )
    if cache and key in _HF_CACHE:
        cached = _HF_CACHE[key]
        return HFLoadedArtifacts(
            model=cached.model,
            processor=cached.processor,
            backend=cached.backend,
            model_id_or_path=cached.model_id_or_path,
            device=cached.device,
            dtype=cached.dtype,
            from_cache=True,
        )

    try:
        model = _load_model(normalized_ref, model_kwargs=model_kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model '{normalized_ref}' via AutoModelForCausalLM: {exc}"
        ) from exc

    try:
        processor = _load_processor(normalized_ref)
    except Exception as exc:
        raise RuntimeError(f"Failed to load processor/tokenizer for '{normalized_ref}': {exc}") from exc

    model.eval()

    tokenizer = getattr(processor, "tokenizer", processor)
    try:
        backend = HFLLMBackend(
            model,
            tokenizer=tokenizer,
            device=resolved_device,
            dtype=resolved_dtype,
            adapter_name=adapter_name,
            language_trunk_path=language_trunk_path,
            strict_arch=strict_arch,
        )
    except Exception as exc:
        diagnostics = inspect_model_architecture(
            model,
            adapter_name=adapter_name,
            language_trunk_path=language_trunk_path,
        )
        candidates = diagnostics.get("candidate_backbones", [])
        selection_error = diagnostics.get("selection_error", "unknown")
        raise ValueError(
            "Unsupported model architecture for HFLLMBackend. "
            f"model='{normalized_ref}', adapter_name='{adapter_name}', "
            f"language_trunk_path='{language_trunk_path}', candidates={candidates}, "
            f"selection_error={selection_error}. "
            f"Original error: {exc}"
        ) from exc

    artifacts = HFLoadedArtifacts(
        model=model,
        processor=processor,
        backend=backend,
        model_id_or_path=normalized_ref,
        device=resolved_device,
        dtype=resolved_dtype,
        from_cache=False,
    )
    if cache:
        _HF_CACHE[key] = artifacts
    return artifacts


__all__ = [
    "HFLoadedArtifacts",
    "clear_hf_loader_cache",
    "load_hf_backend_and_processor",
]
