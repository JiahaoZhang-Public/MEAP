import importlib

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from meap.api import discover_circuit
from meap.backend import HFLLMBackend
from meap.hf_loader import HFLoadedArtifacts, clear_hf_loader_cache, load_hf_backend_and_processor

api_module = importlib.import_module("meap.api")
hf_loader_module = importlib.import_module("meap.hf_loader")


class DummyProcessor:
    def __call__(self, text=None, return_tensors="pt", padding=True, **kwargs):
        del kwargs
        assert return_tensors == "pt"
        assert padding is True

        seqs = []
        for item in text:
            token_ids = [1] + [2] * len(item.split())
            seqs.append(token_ids)

        max_len = max(len(s) for s in seqs)
        padded = [s + [0] * (max_len - len(s)) for s in seqs]
        masks = [[1] * len(s) + [0] * (max_len - len(s)) for s in seqs]
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


class RequiredImageSizeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        cfg = LlamaConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=64,
            max_position_embeddings=32,
        )
        self.language_model = LlamaForCausalLM(cfg)
        self.model = self.language_model.model
        self.lm_head = self.language_model.lm_head
        self.config = self.language_model.config
        self.last_seen_image_sizes = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        image_sizes=None,
        use_cache=False,
        return_dict=True,
        **kwargs,
    ):
        del kwargs
        if image_sizes is None:
            raise RuntimeError("image_sizes missing")
        self.last_seen_image_sizes = image_sizes
        return self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=return_dict,
        )


class ProcessorWithImageSizes:
    def __call__(self, text=None, return_tensors="pt", padding=True, **kwargs):
        del kwargs
        assert return_tensors == "pt"
        assert padding is True
        batch = len(text)
        input_ids = torch.tensor([[1, 2, 3] for _ in range(batch)], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_sizes": [(224, 224) for _ in range(batch)],
        }


def _tiny_artifacts(*, model=None, processor=None):
    if model is None:
        model = LlamaForCausalLM(
            LlamaConfig(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                vocab_size=64,
                max_position_embeddings=32,
            )
        )
    backend = HFLLMBackend(model)
    return HFLoadedArtifacts(
        model=model,
        processor=(processor or DummyProcessor()),
        backend=backend,
        model_id_or_path="dummy",
        device=backend.config.device,
        dtype=backend.config.dtype,
        from_cache=False,
    )


def test_discover_circuit_runs_with_tensor_pair_and_topk_sorted(monkeypatch):
    artifacts = _tiny_artifacts()
    monkeypatch.setattr(api_module, "load_hf_backend_and_processor", lambda **kwargs: artifacts)

    clean_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    corrupt_ids = torch.tensor([[1, 2, 9, 4]], dtype=torch.long)
    mask = torch.ones_like(clean_ids)

    result = discover_circuit(
        model_id_or_path="dummy",
        clean_samples={"input_ids": clean_ids, "attention_mask": mask},
        corrupt_samples={"input_ids": corrupt_ids, "attention_mask": mask.clone()},
        task="next_token",
        labels=torch.tensor([4]),
        method="EAP",
        top_k=5,
        cache=False,
    )

    assert result.scores.shape == (result.graph.n_forward, result.graph.n_backward)
    assert len(result.top_edges) == 5
    assert all(
        result.top_edges[i].abs_score >= result.top_edges[i + 1].abs_score
        for i in range(len(result.top_edges) - 1)
    )
    assert result.backend_info["adapter_name"] is not None


def test_discover_circuit_forwards_adapter_and_trunk_selectors(monkeypatch):
    artifacts = _tiny_artifacts()
    captured = {}

    def _fake_loader(**kwargs):
        captured.update(kwargs)
        return artifacts

    monkeypatch.setattr(api_module, "load_hf_backend_and_processor", _fake_loader)

    clean_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    corrupt_ids = torch.tensor([[1, 2, 9, 4]], dtype=torch.long)
    mask = torch.ones_like(clean_ids)

    _ = discover_circuit(
        model_id_or_path="dummy",
        clean_samples={"input_ids": clean_ids, "attention_mask": mask},
        corrupt_samples={"input_ids": corrupt_ids, "attention_mask": mask.clone()},
        task="next_token",
        labels=torch.tensor([4]),
        adapter_name="llama_like",
        language_trunk_path="model.model",
        cache=False,
    )

    assert captured["adapter_name"] == "llama_like"
    assert captured["language_trunk_path"] == "model.model"


def test_discover_circuit_choice_classification_metric(monkeypatch):
    artifacts = _tiny_artifacts()
    monkeypatch.setattr(api_module, "load_hf_backend_and_processor", lambda **kwargs: artifacts)

    clean_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    corrupt_ids = torch.tensor([[1, 2, 9, 4]], dtype=torch.long)
    mask = torch.ones_like(clean_ids)

    result = discover_circuit(
        model_id_or_path="dummy",
        clean_samples={"input_ids": clean_ids, "attention_mask": mask},
        corrupt_samples={"input_ids": corrupt_ids, "attention_mask": mask.clone()},
        task="choice_classification",
        labels={"targets": [0], "choice_token_ids": [3, 4, 5]},
        method="EAP",
        top_k=3,
        cache=False,
    )

    assert result.run_info["task"] == "choice_classification"
    assert len(result.top_edges) == 3


def test_discover_circuit_rejects_missing_labels(monkeypatch):
    artifacts = _tiny_artifacts()
    monkeypatch.setattr(api_module, "load_hf_backend_and_processor", lambda **kwargs: artifacts)

    clean_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    corrupt_ids = torch.tensor([[1, 2, 9, 4]], dtype=torch.long)
    mask = torch.ones_like(clean_ids)

    with pytest.raises(ValueError, match="task='next_token' requires labels"):
        discover_circuit(
            model_id_or_path="dummy",
            clean_samples={"input_ids": clean_ids, "attention_mask": mask},
            corrupt_samples={"input_ids": corrupt_ids, "attention_mask": mask.clone()},
            task="next_token",
            labels=None,
            cache=False,
        )


def test_discover_circuit_preserves_non_tensor_processor_outputs(monkeypatch):
    model = RequiredImageSizeModel()
    artifacts = _tiny_artifacts(model=model, processor=ProcessorWithImageSizes())
    monkeypatch.setattr(api_module, "load_hf_backend_and_processor", lambda **kwargs: artifacts)

    _ = discover_circuit(
        model_id_or_path="dummy",
        clean_samples={"text": ["hello world"]},
        corrupt_samples={"text": ["hello moon"]},
        task="next_token",
        labels=[3],
        method="EAP",
        cache=False,
    )

    assert model.last_seen_image_sizes is not None


def test_hf_loader_cache_reuses_model_backend(monkeypatch):
    clear_hf_loader_cache()

    calls = {"model": 0, "processor": 0, "backend": 0}

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Cfg", (), {})()

        def eval(self):
            return self

    class DummyBackend:
        def __init__(self, model, **kwargs):
            del kwargs
            self.model = model
            self.adapter_name = "dummy"
            self.language_trunk_path = "model"
            self.arch_kind = "dummy"

    def fake_load_model(model_id_or_path, *, model_kwargs):
        del model_id_or_path, model_kwargs
        calls["model"] += 1
        return DummyModel()

    def fake_load_processor(model_id_or_path):
        del model_id_or_path
        calls["processor"] += 1
        return object()

    def fake_backend(model, **kwargs):
        del kwargs
        calls["backend"] += 1
        return DummyBackend(model)

    monkeypatch.setattr(hf_loader_module, "_load_model", fake_load_model)
    monkeypatch.setattr(hf_loader_module, "_load_processor", fake_load_processor)
    monkeypatch.setattr(hf_loader_module, "HFLLMBackend", fake_backend)

    first = load_hf_backend_and_processor(
        model_id_or_path="dummy-model",
        device="cpu",
        dtype="float32",
        cache=True,
    )
    second = load_hf_backend_and_processor(
        model_id_or_path="dummy-model",
        device="cpu",
        dtype="float32",
        cache=True,
    )

    assert first.from_cache is False
    assert second.from_cache is True
    assert calls["model"] == 1
    assert calls["processor"] == 1
    assert calls["backend"] == 1


def test_hf_loader_cache_key_includes_adapter_and_trunk(monkeypatch):
    clear_hf_loader_cache()

    calls = {"backend": 0}

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Cfg", (), {})()

        def eval(self):
            return self

    class DummyBackend:
        def __init__(self, model, **kwargs):
            del kwargs
            self.model = model
            self.adapter_name = "dummy"
            self.language_trunk_path = "model"
            self.arch_kind = "dummy"

    monkeypatch.setattr(hf_loader_module, "_load_model", lambda *args, **kwargs: DummyModel())
    monkeypatch.setattr(hf_loader_module, "_load_processor", lambda *args, **kwargs: object())

    def _fake_backend(model, **kwargs):
        del model, kwargs
        calls["backend"] += 1
        return DummyBackend(model=None)

    monkeypatch.setattr(hf_loader_module, "HFLLMBackend", _fake_backend)

    _ = load_hf_backend_and_processor(
        model_id_or_path="dummy-model",
        device="cpu",
        dtype="float32",
        adapter_name="llama_like",
        language_trunk_path="model.model",
        cache=True,
    )
    _ = load_hf_backend_and_processor(
        model_id_or_path="dummy-model",
        device="cpu",
        dtype="float32",
        adapter_name="llama_like",
        language_trunk_path="model.model",
        cache=True,
    )
    _ = load_hf_backend_and_processor(
        model_id_or_path="dummy-model",
        device="cpu",
        dtype="float32",
        adapter_name="gpt2_like",
        language_trunk_path="model.transformer",
        cache=True,
    )

    assert calls["backend"] == 2
