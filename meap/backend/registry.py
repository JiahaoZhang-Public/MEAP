from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch

from .base import AdapterResolution, ArchitectureAdapter

AdapterType = Union[ArchitectureAdapter, Type[ArchitectureAdapter]]


@dataclass(frozen=True)
class ResolutionAttempt:
    path: str
    adapter: str
    arch_kind: str
    status: str  # skip | error | match
    detail: str
    required_modules: List[str]
    missing_modules: List[str]


class ResolutionError(ValueError):
    def __init__(self, message: str, diagnostics: Dict[str, Any]):
        super().__init__(message)
        self.diagnostics = diagnostics


# Candidate order is intentionally stable for deterministic adapter resolution.
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
    existing = [
        cls for cls in _REGISTERED_ADAPTER_CLASSES if getattr(cls, "name", None) != getattr(adapter_cls, "name", None)
    ]
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


def _find_first_layer(backbone: torch.nn.Module) -> Optional[torch.nn.Module]:
    for layer_attr in ("layers", "h", "blocks"):
        maybe_layers = getattr(backbone, layer_attr, None)
        if maybe_layers is not None and len(maybe_layers) > 0:
            return maybe_layers[0]
    return None


def _path_exists(root: Any, dotted: str) -> bool:
    node = root
    for part in dotted.split("."):
        if node is None or not hasattr(node, part):
            return False
        node = getattr(node, part)
    return True


def _resolve_explicit_candidate(
    model: torch.nn.Module,
    language_trunk_path: str,
) -> Tuple[str, torch.nn.Module]:
    def _resolve(path: str) -> Optional[torch.nn.Module]:
        parts = [part for part in path.split(".") if part]
        if len(parts) == 0:
            return None
        if parts[0] == "model":
            parts = parts[1:]
        node: Any = model
        for part in parts:
            if not hasattr(node, part):
                return None
            node = getattr(node, part)
        if isinstance(node, torch.nn.Module):
            return node
        return None

    candidate = _resolve(language_trunk_path)
    normalized_path = language_trunk_path
    if candidate is None and not language_trunk_path.startswith("model"):
        normalized_path = f"model.{language_trunk_path}"
        candidate = _resolve(normalized_path)
    if candidate is not None:
        return normalized_path, candidate

    available = [path for path, _ in iter_decoder_backbone_candidates(model)]
    raise ValueError(
        f"language_trunk_path='{language_trunk_path}' not found or not a torch.nn.Module. "
        f"Available candidates: {available}"
    )


def _adapter_module_requirements(
    backbone: torch.nn.Module,
    adapter: ArchitectureAdapter,
) -> Tuple[List[str], List[str]]:
    required_modules = list(adapter.required_modules())
    first_layer = _find_first_layer(backbone)
    missing_modules = [
        req
        for req in required_modules
        if not _path_exists(backbone, req)
        and not (first_layer is not None and _path_exists(first_layer, req))
    ]
    return required_modules, missing_modules


def _new_resolution_diagnostics(
    candidates: Sequence[Tuple[str, torch.nn.Module]],
    adapters: Sequence[ArchitectureAdapter],
) -> Dict[str, Any]:
    return {
        "candidate_backbones": [path for path, _ in candidates],
        "adapters": [adapter.name for adapter in adapters],
        "adapter_attempts": [],
        "errors": [],
        "selected": None,
    }


def resolve_adapter_resolution(
    model: torch.nn.Module,
    adapters: Sequence[ArchitectureAdapter],
    *,
    adapter_name: Optional[str] = None,
    language_trunk_path: Optional[str] = None,
) -> Tuple[ArchitectureAdapter, AdapterResolution, Dict[str, Any]]:
    if language_trunk_path is not None:
        try:
            candidates = [_resolve_explicit_candidate(model, language_trunk_path)]
        except ValueError as exc:
            diagnostics = _new_resolution_diagnostics([], adapters)
            diagnostics["requested_adapter_name"] = adapter_name
            diagnostics["requested_language_trunk_path"] = language_trunk_path
            diagnostics["selection_error"] = str(exc)
            raise ResolutionError(str(exc), diagnostics) from exc
    else:
        candidates = iter_decoder_backbone_candidates(model)

    adapters_to_try = [adapter for adapter in adapters if adapter_name is None or adapter.name == adapter_name]
    diagnostics = _new_resolution_diagnostics(candidates, adapters_to_try)
    diagnostics["requested_adapter_name"] = adapter_name
    diagnostics["requested_language_trunk_path"] = language_trunk_path
    if len(adapters_to_try) == 0 and adapter_name is not None:
        available = [adapter.name for adapter in adapters]
        diagnostics["selection_error"] = (
            f"Unknown adapter_name='{adapter_name}'. Available adapters: {available}"
        )
        raise ResolutionError(diagnostics["selection_error"], diagnostics)

    for path, backbone in candidates:
        for adapter in adapters_to_try:
            required_modules, missing_modules = _adapter_module_requirements(backbone, adapter)

            try:
                resolved = adapter.resolve(path, backbone)
            except ValueError as exc:
                detail = str(exc)
                attempt = ResolutionAttempt(
                    path=path,
                    adapter=adapter.name,
                    arch_kind=getattr(adapter, "arch_kind", adapter.name),
                    status="error",
                    detail=detail,
                    required_modules=required_modules,
                    missing_modules=missing_modules,
                )
                diagnostics["adapter_attempts"].append(attempt.__dict__)
                diagnostics["errors"].append(
                    {
                        "path": path,
                        "adapter": adapter.name,
                        "arch_kind": getattr(adapter, "arch_kind", adapter.name),
                        "detail": detail,
                        "missing_modules": missing_modules,
                    }
                )
                continue

            if resolved is not None:
                attempt = ResolutionAttempt(
                    path=path,
                    adapter=adapter.name,
                    arch_kind=getattr(adapter, "arch_kind", adapter.name),
                    status="match",
                    detail="resolved",
                    required_modules=required_modules,
                    missing_modules=missing_modules,
                )
                diagnostics["adapter_attempts"].append(attempt.__dict__)
                diagnostics["selected"] = {
                    "adapter": adapter.name,
                    "arch_kind": resolved.arch_kind,
                    "path": resolved.path,
                    "layer_count": len(resolved.layers),
                    "layer_accessors": resolved.layer_accessors,
                }
                return adapter, resolved, diagnostics

            attempt = ResolutionAttempt(
                path=path,
                adapter=adapter.name,
                arch_kind=getattr(adapter, "arch_kind", adapter.name),
                status="skip",
                detail="match() returned False",
                required_modules=required_modules,
                missing_modules=missing_modules,
            )
            diagnostics["adapter_attempts"].append(attempt.__dict__)

    details = diagnostics["errors"][:6]
    if details:
        details_text = "; ".join(
            f"{entry['adapter']}@{entry['path']}: {entry['detail']}"
            for entry in details
        )
    else:
        details_text = "no adapter produced a compatible backbone"
    diagnostics["selection_error"] = details_text

    raise ResolutionError(
        (
            "Unsupported HF architecture for HFLLMBackend. "
            f"Tried backbones: {diagnostics['candidate_backbones']}. "
            f"Expected decoder-like structure with attention, MLP, and norm modules. "
            f"Details: {details_text}"
        ),
        diagnostics,
    )


def inspect_model_architecture(
    model: torch.nn.Module,
    adapter_registry: Optional[Sequence[AdapterType]] = None,
    *,
    adapter_name: Optional[str] = None,
    language_trunk_path: Optional[str] = None,
) -> Dict[str, Any]:
    adapters_all = instantiate_adapters(adapter_registry)
    if adapter_name is not None:
        adapters = [adapter for adapter in adapters_all if adapter.name == adapter_name]
    else:
        adapters = adapters_all

    if language_trunk_path is not None:
        try:
            candidates = [_resolve_explicit_candidate(model, language_trunk_path)]
        except ValueError as exc:
            candidates = []
            selection_error = str(exc)
    else:
        candidates = iter_decoder_backbone_candidates(model)
        selection_error = ""

    report: Dict[str, Any] = {
        "candidate_backbones": [path for path, _ in candidates],
        "adapters": [adapter.name for adapter in adapters],
        "requested_adapter_name": adapter_name,
        "requested_language_trunk_path": language_trunk_path,
        "matches": [],
        "adapter_attempts": [],
        "errors": [],
        "selected": None,
    }
    if adapter_name is not None and len(adapters) == 0:
        report["selection_error"] = (
            f"Unknown adapter_name='{adapter_name}'. "
            f"Available adapters: {[adapter.name for adapter in adapters_all]}"
        )
    elif language_trunk_path is not None and len(candidates) == 0:
        report["selection_error"] = selection_error

    for path, backbone in candidates:
        for adapter in adapters:
            required_modules, missing_modules = _adapter_module_requirements(backbone, adapter)
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
        _, _, resolution_diag = resolve_adapter_resolution(
            model,
            adapters_all,
            adapter_name=adapter_name,
            language_trunk_path=language_trunk_path,
        )
        report["adapter_attempts"] = resolution_diag["adapter_attempts"]
        report["errors"] = resolution_diag["errors"]
        report["selected"] = resolution_diag["selected"]
    except ResolutionError as exc:
        report["adapter_attempts"] = exc.diagnostics["adapter_attempts"]
        report["errors"] = exc.diagnostics["errors"]
        report["selected"] = None
        report["selection_error"] = str(exc)

    return report
