from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch

from .base import AdapterResolution, ArchitectureAdapter

AdapterType = Union[ArchitectureAdapter, Type[ArchitectureAdapter]]


def iter_decoder_backbone_candidates(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
    candidates: List[Tuple[str, torch.nn.Module]] = []

    def _add(path: str, module: Any) -> None:
        if isinstance(module, torch.nn.Module):
            candidates.append((path, module))

    _add("model", model)

    if hasattr(model, "language_model"):
        lm = model.language_model
        _add("model.language_model", lm)
        _add("model.language_model.model", getattr(lm, "model", None))
        _add("model.language_model.model.decoder", getattr(getattr(lm, "model", None), "decoder", None))
        _add("model.language_model.transformer", getattr(lm, "transformer", None))
        _add("model.language_model.decoder", getattr(lm, "decoder", None))

    _add("model.model", getattr(model, "model", None))
    _add("model.model.language_model", getattr(getattr(model, "model", None), "language_model", None))
    _add("model.model.text_model", getattr(getattr(model, "model", None), "text_model", None))
    _add("model.model.decoder", getattr(getattr(model, "model", None), "decoder", None))
    _add("model.text_model", getattr(model, "text_model", None))
    _add("model.model.model", getattr(getattr(model, "model", None), "model", None))
    _add("model.transformer", getattr(model, "transformer", None))
    _add("model.decoder", getattr(model, "decoder", None))

    deduped: List[Tuple[str, torch.nn.Module]] = []
    seen = set()
    for path, module in candidates:
        marker = id(module)
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append((path, module))
    return deduped


def _default_adapter_classes() -> List[Type[ArchitectureAdapter]]:
    from .adapters import (
        FalconLikeAdapter,
        GPT2LikeAdapter,
        LlamaLikeAdapter,
        MPTLikeAdapter,
        OPTLikeAdapter,
    )

    return [
        LlamaLikeAdapter,
        GPT2LikeAdapter,
        OPTLikeAdapter,
        FalconLikeAdapter,
        MPTLikeAdapter,
    ]


_REGISTERED_ADAPTER_CLASSES: List[Type[ArchitectureAdapter]] = _default_adapter_classes()


def get_registered_architecture_adapters() -> List[Type[ArchitectureAdapter]]:
    return list(_REGISTERED_ADAPTER_CLASSES)


def register_architecture_adapter(
    adapter_cls: Type[ArchitectureAdapter],
    *,
    prepend: bool = False,
) -> None:
    global _REGISTERED_ADAPTER_CLASSES
    existing = [cls for cls in _REGISTERED_ADAPTER_CLASSES if getattr(cls, "name", None) != getattr(adapter_cls, "name", None)]
    if prepend:
        _REGISTERED_ADAPTER_CLASSES = [adapter_cls, *existing]
    else:
        existing.append(adapter_cls)
        _REGISTERED_ADAPTER_CLASSES = existing


def reset_architecture_adapter_registry() -> None:
    global _REGISTERED_ADAPTER_CLASSES
    _REGISTERED_ADAPTER_CLASSES = _default_adapter_classes()


def instantiate_adapters(
    adapter_registry: Optional[Sequence[AdapterType]] = None,
) -> List[ArchitectureAdapter]:
    source = list(adapter_registry) if adapter_registry is not None else list(_REGISTERED_ADAPTER_CLASSES)
    out: List[ArchitectureAdapter] = []
    for item in source:
        if isinstance(item, type):
            out.append(item())
        else:
            out.append(item)
    return out


def resolve_adapter_resolution(
    model: torch.nn.Module,
    adapters: Sequence[ArchitectureAdapter],
    *,
    adapter_name: Optional[str] = None,
) -> Tuple[ArchitectureAdapter, AdapterResolution, List[str], List[Dict[str, Any]]]:
    candidates = iter_decoder_backbone_candidates(model)
    errors: List[str] = []
    diagnostics: List[Dict[str, Any]] = []

    for path, backbone in candidates:
        for adapter in adapters:
            if adapter_name is not None and adapter.name != adapter_name:
                continue
            try:
                resolved = adapter.resolve(path, backbone)
            except ValueError as exc:
                message = str(exc)
                errors.append(message)
                diagnostics.append(
                    {
                        "path": path,
                        "adapter": adapter.name,
                        "status": "error",
                        "detail": message,
                    }
                )
                continue
            if resolved is not None:
                diagnostics.append(
                    {
                        "path": path,
                        "adapter": adapter.name,
                        "status": "match",
                        "detail": "resolved",
                    }
                )
                return adapter, resolved, errors, diagnostics
            diagnostics.append(
                {
                    "path": path,
                    "adapter": adapter.name,
                    "status": "skip",
                    "detail": "match() returned False",
                }
            )

    raise ValueError(
        "Unsupported HF architecture for HFLLMBackend. "
        f"Tried backbones: {[path for path, _ in candidates]}. "
        f"Expected decoder-like structure with attention, MLP, and norm modules. "
        f"Details: {('; '.join(errors[:6]) if errors else 'no adapter produced a compatible backbone')}"
    )


def inspect_model_architecture(
    model: torch.nn.Module,
    adapter_registry: Optional[Sequence[AdapterType]] = None,
) -> Dict[str, Any]:
    adapters = instantiate_adapters(adapter_registry)
    candidates = iter_decoder_backbone_candidates(model)
    report: Dict[str, Any] = {
        "candidate_backbones": [path for path, _ in candidates],
        "adapters": [adapter.name for adapter in adapters],
        "matches": [],
    }

    for path, backbone in candidates:
        first_layer = None
        for layer_attr in ("layers", "h", "blocks"):
            maybe_layers = getattr(backbone, layer_attr, None)
            if maybe_layers is not None and len(maybe_layers) > 0:
                first_layer = maybe_layers[0]
                break

        def _path_exists(root: Any, dotted: str) -> bool:
            node = root
            for part in dotted.split("."):
                if node is None or not hasattr(node, part):
                    return False
                node = getattr(node, part)
            return True

        for adapter in adapters:
            required_modules = list(adapter.required_modules())
            missing_modules = [
                req
                for req in required_modules
                if not _path_exists(backbone, req)
                and not (first_layer is not None and _path_exists(first_layer, req))
            ]
            entry: Dict[str, Any] = {
                "path": path,
                "adapter": adapter.name,
                "matched": False,
                "arch_kind": getattr(adapter, "arch_kind", adapter.name),
                "required_modules": required_modules,
                "missing_modules": missing_modules,
                "error": "",
            }
            try:
                resolved = adapter.resolve(path, backbone)
                if resolved is not None:
                    entry["matched"] = True
                    entry["layer_count"] = len(resolved.layers)
                else:
                    entry["error"] = "match() returned False"
            except Exception as exc:  # pragma: no cover - purely diagnostic path
                entry["error"] = str(exc)
            report["matches"].append(entry)

    try:
        adapter, resolved, _, _ = resolve_adapter_resolution(model, adapters)
        report["selected"] = {
            "adapter": adapter.name,
            "arch_kind": resolved.arch_kind,
            "path": resolved.path,
            "layer_count": len(resolved.layers),
            "layer_accessors": resolved.layer_accessors,
        }
    except Exception as exc:
        report["selected"] = None
        report["selection_error"] = str(exc)

    return report
