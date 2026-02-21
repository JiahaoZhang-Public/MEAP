from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List


@dataclass(frozen=True)
class SupportedArchitecture:
    adapter_name: str
    arch_kind: str
    modalities: str
    tier: str


@dataclass(frozen=True)
class OfficialModel:
    model_id: str
    adapter_name: str
    modality: str
    tier: str


_SUPPORTED_ARCHITECTURES: tuple[SupportedArchitecture, ...] = (
    SupportedArchitecture(
        adapter_name="gpt2_like",
        arch_kind="gpt2_like",
        modalities="text",
        tier="core",
    ),
    SupportedArchitecture(
        adapter_name="opt_like",
        arch_kind="opt_like",
        modalities="text",
        tier="core",
    ),
    SupportedArchitecture(
        adapter_name="llama_like",
        arch_kind="llama_like",
        modalities="text,multimodal",
        tier="core",
    ),
)


_OFFICIAL_MODELS: tuple[OfficialModel, ...] = (
    OfficialModel(
        model_id="gpt2",
        adapter_name="gpt2_like",
        modality="text",
        tier="core",
    ),
    OfficialModel(
        model_id="distilgpt2",
        adapter_name="gpt2_like",
        modality="text",
        tier="core",
    ),
    OfficialModel(
        model_id="facebook/opt-125m",
        adapter_name="opt_like",
        modality="text",
        tier="core",
    ),
    OfficialModel(
        model_id="Qwen/Qwen2-0.5B",
        adapter_name="llama_like",
        modality="text",
        tier="core",
    ),
    OfficialModel(
        model_id="Qwen/Qwen2-VL-2B",
        adapter_name="llama_like",
        modality="multimodal",
        tier="core",
    ),
    OfficialModel(
        model_id="llava-hf/llava-1.5-7b-hf",
        adapter_name="llama_like",
        modality="multimodal",
        tier="core",
    ),
)


def list_supported_architectures() -> List[dict[str, str]]:
    return [asdict(item) for item in _SUPPORTED_ARCHITECTURES]


def list_official_models() -> List[dict[str, str]]:
    return [asdict(item) for item in _OFFICIAL_MODELS]


__all__ = [
    "OfficialModel",
    "SupportedArchitecture",
    "list_official_models",
    "list_supported_architectures",
]
