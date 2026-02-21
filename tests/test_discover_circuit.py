import importlib

import torch

from meap.hf_loader import clear_hf_loader_cache, load_hf_backend_and_processor

hf_loader_module = importlib.import_module("meap.hf_loader")


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
