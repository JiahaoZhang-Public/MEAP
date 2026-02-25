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
    SupportedArchitecture(
        adapter_name="falcon_like",
        arch_kind="falcon_like",
        modalities="text",
        tier="core",
    ),
    SupportedArchitecture(
        adapter_name="gemma_like",
        arch_kind="gemma_like",
        modalities="text",
        tier="core",
    ),
    SupportedArchitecture(
        adapter_name="phi_like",
        arch_kind="phi_like",
        modalities="text",
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
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        adapter_name="llama_like",
        modality="text",
        tier="core",
    ),
    OfficialModel(
        model_id="tiiuae/falcon-rw-1b",
        adapter_name="falcon_like",
        modality="text",
        tier="core",
    ),
    OfficialModel(
        model_id="google/gemma-2-2b",
        adapter_name="gemma_like",
        modality="text",
        tier="nightly",
    ),
    OfficialModel(
        model_id="microsoft/phi-2",
        adapter_name="phi_like",
        modality="text",
        tier="nightly",
    ),
    OfficialModel(
        model_id="microsoft/Phi-3-mini-4k-instruct",
        adapter_name="phi_like",
        modality="text",
        tier="nightly",
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
        tier="nightly",
    ),
    OfficialModel(
        model_id="HuggingFaceTB/SmolVLM-Instruct",
        adapter_name="llama_like",
        modality="multimodal",
        tier="core",
    ),
    OfficialModel(
        model_id="fixie-ai/ultravox-v0_5-llama-3_2-1b",
        adapter_name="llama_like",
        modality="multimodal",
        tier="core",
    ),
    OfficialModel(
        model_id="Qwen/Qwen2-Audio-7B",
        adapter_name="llama_like",
        modality="multimodal",
        tier="nightly",
    ),
    OfficialModel(
        model_id="HuggingFaceM4/idefics2-8b",
        adapter_name="llama_like",
        modality="multimodal",
        tier="nightly",
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
