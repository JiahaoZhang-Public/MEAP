from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

import torch
from torch import Tensor

BackendHookSpec = Tuple[str, Callable]


@dataclass
class BackendRunInputs:
    input_ids: Optional[Tensor]
    inputs_embeds: Optional[Tensor]
    attention_mask: Optional[Tensor]
    extra_fwd_hooks: List[BackendHookSpec] = field(default_factory=list)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendConfig:
    device: torch.device
    dtype: torch.dtype
    d_model: int
    n_layers: int
    n_heads: int
    parallel_attn_mlp: bool
    n_key_value_heads: Optional[int]
    use_attn_result: bool
    use_split_qkv_input: bool
    use_hook_mlp_in: bool
    use_normalization_before_and_after: bool = False

    def to_graph_dict(self) -> Dict[str, Any]:
        return {
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "parallel_attn_mlp": self.parallel_attn_mlp,
            "d_model": self.d_model,
        }


@runtime_checkable
class ModelBackend(Protocol):
    kind: str

    @property
    def config(self) -> BackendConfig:
        ...

    @property
    def tokenizer(self):
        ...

    @property
    def tokenization_model(self):
        ...

    def prepare_inputs(self, inputs: Dict[str, Tensor]) -> BackendRunInputs:
        ...

    def forward(
        self,
        run_inputs: BackendRunInputs,
        *,
        fwd_hooks: Optional[List[BackendHookSpec]] = None,
        bwd_hooks: Optional[List[BackendHookSpec]] = None,
    ) -> Tensor:
        ...

    def zero_grad(self) -> None:
        ...

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        ...


@dataclass
class _HookTarget:
    module: torch.nn.Module
    mode: str  # 'forward' or 'pre'


def _parse_torch_dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str) and hasattr(torch, dtype):
        parsed = getattr(torch, dtype)
        if isinstance(parsed, torch.dtype):
            return parsed
    return torch.float32


def _first_parameter_device_dtype(model: torch.nn.Module) -> Tuple[torch.device, torch.dtype]:
    first_param = next(model.parameters(), None)
    if first_param is None:
        return torch.device("cpu"), torch.float32
    return first_param.device, first_param.dtype


def _is_hooked_transformer_model(model: Any) -> bool:
    try:
        from transformer_lens import HookedTransformer
    except Exception:
        return False
    return isinstance(model, HookedTransformer)


class TLensBackend:
    kind = "tlens"

    def __init__(self, model: Any):
        if not _is_hooked_transformer_model(model):
            raise TypeError("TLensBackend requires a transformer_lens.HookedTransformer instance")
        self.model = model
        cfg = model.cfg
        device = torch.device(getattr(cfg, "device", "cpu"))
        dtype = _parse_torch_dtype(getattr(cfg, "dtype", torch.float32))
        self._config = BackendConfig(
            device=device,
            dtype=dtype,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            parallel_attn_mlp=cfg.parallel_attn_mlp,
            n_key_value_heads=cfg.n_key_value_heads,
            use_attn_result=bool(getattr(cfg, "use_attn_result", False)),
            use_split_qkv_input=bool(getattr(cfg, "use_split_qkv_input", False)),
            use_hook_mlp_in=bool(getattr(cfg, "use_hook_mlp_in", False)),
            use_normalization_before_and_after=bool(
                getattr(cfg, "use_normalization_before_and_after", False)
            ),
        )

    @property
    def config(self) -> BackendConfig:
        return self._config

    @property
    def tokenizer(self):
        return self.model.tokenizer

    @property
    def tokenization_model(self):
        return self.model

    def prepare_inputs(self, inputs: Dict[str, Tensor]) -> BackendRunInputs:
        device = self._config.device
        dtype = self._config.dtype

        tokens = inputs.get("input_ids")
        if tokens is None:
            tokens = inputs.get("tokens")

        input_embeds = inputs.get("inputs_embeds")

        if tokens is None and input_embeds is None:
            raise ValueError("Model inputs must include input_ids/tokens or inputs_embeds")

        if input_embeds is not None:
            input_embeds = input_embeds.to(device=device, dtype=dtype)

        if tokens is None:
            if input_embeds is None or input_embeds.ndim != 3:
                raise ValueError("inputs_embeds must be rank-3 [batch, seq, d_model]")
            pad_token_id = getattr(self.model.tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(self.model.tokenizer, "eos_token_id", 0)
            tokens = torch.full(
                input_embeds.shape[:2],
                int(pad_token_id),
                device=device,
                dtype=torch.long,
            )
        else:
            tokens = tokens.to(device=device, dtype=torch.long)

        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=device)
        else:
            attention_mask = torch.ones(tokens.shape, device=device, dtype=torch.long)

        if attention_mask.shape != tokens.shape:
            raise ValueError(
                "attention_mask shape must match tokens shape. "
                f"Got attention_mask={tuple(attention_mask.shape)} tokens={tuple(tokens.shape)}"
            )

        extra_fwd_hooks: List[BackendHookSpec] = []
        if input_embeds is not None:

            def _embed_override_hook(
                activations: Tensor,
                hook,
                prepared_embeds: Tensor = input_embeds,
            ) -> Tensor:
                return prepared_embeds

            extra_fwd_hooks.append(("hook_embed", _embed_override_hook))

        return BackendRunInputs(
            input_ids=tokens,
            inputs_embeds=None,
            attention_mask=attention_mask,
            extra_fwd_hooks=extra_fwd_hooks,
            model_kwargs={},
        )

    def forward(
        self,
        run_inputs: BackendRunInputs,
        *,
        fwd_hooks: Optional[List[BackendHookSpec]] = None,
        bwd_hooks: Optional[List[BackendHookSpec]] = None,
    ) -> Tensor:
        merged_fwd_hooks = list(run_inputs.extra_fwd_hooks)
        if fwd_hooks:
            merged_fwd_hooks.extend(fwd_hooks)

        kwargs = {"attention_mask": run_inputs.attention_mask}

        if not merged_fwd_hooks and not bwd_hooks:
            return self.model(run_inputs.input_ids, **kwargs)

        hook_kwargs = {"fwd_hooks": merged_fwd_hooks}
        if bwd_hooks is not None:
            hook_kwargs["bwd_hooks"] = bwd_hooks

        with self.model.hooks(**hook_kwargs):
            return self.model(run_inputs.input_ids, **kwargs)

    def zero_grad(self) -> None:
        self.model.zero_grad(set_to_none=True)

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        return self.model.parameters()


class HFLLMBackend:
    kind = "hf"

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        tokenizer=None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if not hasattr(model, "config"):
            raise TypeError("HFLLMBackend requires a model with a HuggingFace-like .config")

        self.model = model
        if device is not None or dtype is not None:
            cast_kwargs: Dict[str, Any] = {}
            if device is not None:
                cast_kwargs["device"] = device
            if dtype is not None:
                cast_kwargs["dtype"] = dtype
            self.model.to(**cast_kwargs)

        self._tokenizer = tokenizer
        self._base_model = self._resolve_decoder_backbone(self.model)
        self._layers = self._resolve_decoder_layers(self._base_model)
        self._hook_targets = self._build_hook_targets()
        self._config = self._normalize_config(self.model)

    @property
    def config(self) -> BackendConfig:
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def tokenization_model(self):
        return None

    @property
    def supported_hook_names(self) -> List[str]:
        return sorted(self._hook_targets.keys())

    def _resolve_decoder_backbone(self, model: torch.nn.Module) -> torch.nn.Module:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model

        raise ValueError(
            "Unsupported HF architecture for HFLLMBackend. Expected decoder backbone with "
            "attributes: model.layers, model.embed_tokens, per-layer self_attn/mlp."
        )

    def _resolve_decoder_layers(self, base_model: torch.nn.Module) -> List[torch.nn.Module]:
        layers = getattr(base_model, "layers", None)
        if layers is None:
            raise ValueError("Unsupported HF architecture: missing model.layers")
        if len(layers) == 0:
            raise ValueError("Unsupported HF architecture: model.layers is empty")

        required_layer_attrs = ("self_attn", "mlp", "input_layernorm", "post_attention_layernorm")
        required_attn_attrs = ("q_proj", "k_proj", "v_proj")

        for idx, layer in enumerate(layers):
            missing = [attr for attr in required_layer_attrs if not hasattr(layer, attr)]
            if missing:
                raise ValueError(
                    "Unsupported HF architecture: each decoder layer must expose self_attn/mlp and "
                    f"norms. Layer {idx} missing: {missing}"
                )
            attn = layer.self_attn
            missing_attn = [attr for attr in required_attn_attrs if not hasattr(attn, attr)]
            if missing_attn:
                raise ValueError(
                    "Unsupported HF architecture: self_attn must expose q_proj/k_proj/v_proj. "
                    f"Layer {idx} missing: {missing_attn}"
                )
        return list(layers)

    def _normalize_config(self, model: torch.nn.Module) -> BackendConfig:
        cfg = model.config
        missing_cfg = [
            key
            for key in ("hidden_size", "num_hidden_layers", "num_attention_heads")
            if not hasattr(cfg, key)
        ]
        if missing_cfg:
            raise ValueError(
                "Unsupported HF config for HFLLMBackend. Missing fields: "
                f"{missing_cfg}."
            )

        device, model_dtype = _first_parameter_device_dtype(self.model)
        n_key_value_heads = getattr(cfg, "num_key_value_heads", None)

        return BackendConfig(
            device=device,
            dtype=model_dtype,
            d_model=int(cfg.hidden_size),
            n_layers=int(cfg.num_hidden_layers),
            n_heads=int(cfg.num_attention_heads),
            parallel_attn_mlp=False,
            n_key_value_heads=(None if n_key_value_heads is None else int(n_key_value_heads)),
            use_attn_result=True,
            use_split_qkv_input=True,
            use_hook_mlp_in=True,
            use_normalization_before_and_after=False,
        )

    def _build_hook_targets(self) -> Dict[str, _HookTarget]:
        if not hasattr(self._base_model, "embed_tokens"):
            raise ValueError("Unsupported HF architecture: missing model.embed_tokens")

        targets: Dict[str, _HookTarget] = {
            "hook_embed": _HookTarget(self._base_model.embed_tokens, "forward"),
        }

        for layer_idx, layer in enumerate(self._layers):
            targets[f"blocks.{layer_idx}.attn.hook_result"] = _HookTarget(layer.self_attn, "forward")
            targets[f"blocks.{layer_idx}.hook_q_input"] = _HookTarget(layer.self_attn.q_proj, "pre")
            targets[f"blocks.{layer_idx}.hook_k_input"] = _HookTarget(layer.self_attn.k_proj, "pre")
            targets[f"blocks.{layer_idx}.hook_v_input"] = _HookTarget(layer.self_attn.v_proj, "pre")
            targets[f"blocks.{layer_idx}.hook_mlp_out"] = _HookTarget(layer.mlp, "forward")
            targets[f"blocks.{layer_idx}.hook_mlp_in"] = _HookTarget(layer.mlp, "pre")

        resid_module = getattr(self._base_model, "norm", self._layers[-1])
        targets[f"blocks.{len(self._layers) - 1}.hook_resid_post"] = _HookTarget(resid_module, "pre")

        return targets

    def prepare_inputs(self, inputs: Dict[str, Tensor]) -> BackendRunInputs:
        device = self._config.device
        dtype = self._config.dtype

        input_ids = inputs.get("input_ids")
        if input_ids is None:
            input_ids = inputs.get("tokens")
        inputs_embeds = inputs.get("inputs_embeds")

        if input_ids is None and inputs_embeds is None:
            raise ValueError("Model inputs must include input_ids/tokens or inputs_embeds")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("HFLLMBackend expects exactly one of input_ids/tokens or inputs_embeds")

        if input_ids is not None:
            input_ids = input_ids.to(device=device, dtype=torch.long)
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(device=device, dtype=dtype)

        if input_ids is not None:
            batch_shape = input_ids.shape
        else:
            if inputs_embeds is None or inputs_embeds.ndim != 3:
                raise ValueError("inputs_embeds must be rank-3 [batch, seq, d_model]")
            batch_shape = inputs_embeds.shape[:2]

        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=device)
        else:
            attention_mask = torch.ones(batch_shape, device=device, dtype=torch.long)

        if attention_mask.shape != batch_shape:
            raise ValueError(
                "attention_mask shape must match token/embed sequence shape. "
                f"Got attention_mask={tuple(attention_mask.shape)} expected={tuple(batch_shape)}"
            )

        model_kwargs: Dict[str, Any] = {}
        for optional_kwarg in ("position_ids", "cache_position"):
            if optional_kwarg in inputs:
                model_kwargs[optional_kwarg] = inputs[optional_kwarg].to(device=device)

        return BackendRunInputs(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            extra_fwd_hooks=[],
            model_kwargs=model_kwargs,
        )

    def _make_forward_hook(self, name: str, hook_fn: Callable):
        hook_ctx = SimpleNamespace(name=name)

        def _wrapped(module, inputs, output):
            maybe_output = hook_fn(output, hook_ctx)
            if maybe_output is not None:
                return maybe_output
            return output

        return _wrapped

    def _make_pre_hook(self, name: str, hook_fn: Callable):
        hook_ctx = SimpleNamespace(name=name)

        def _wrapped(module, inputs):
            if len(inputs) == 0:
                return None
            maybe_input = hook_fn(inputs[0], hook_ctx)
            if maybe_input is None:
                return None
            return (maybe_input, *inputs[1:])

        return _wrapped

    def forward(
        self,
        run_inputs: BackendRunInputs,
        *,
        fwd_hooks: Optional[List[BackendHookSpec]] = None,
        bwd_hooks: Optional[List[BackendHookSpec]] = None,
    ) -> Tensor:
        if bwd_hooks:
            raise RuntimeError(
                "HFLLMBackend currently does not support backward hook interception. "
                "Use TLens backend for gradient-based attribution methods."
            )

        merged_fwd_hooks = list(run_inputs.extra_fwd_hooks)
        if fwd_hooks:
            merged_fwd_hooks.extend(fwd_hooks)

        handles = []
        try:
            for name, hook_fn in merged_fwd_hooks:
                target = self._hook_targets.get(name)
                if target is None:
                    raise ValueError(
                        f"Unsupported HF hook point: {name}. "
                        f"Supported hook names include: {self.supported_hook_names[:12]} ..."
                    )

                if target.mode == "forward":
                    handle = target.module.register_forward_hook(
                        self._make_forward_hook(name, hook_fn)
                    )
                elif target.mode == "pre":
                    handle = target.module.register_forward_pre_hook(
                        self._make_pre_hook(name, hook_fn)
                    )
                else:
                    raise RuntimeError(f"Unknown hook mode: {target.mode}")

                handles.append(handle)

            outputs = self.model(
                input_ids=run_inputs.input_ids,
                inputs_embeds=run_inputs.inputs_embeds,
                attention_mask=run_inputs.attention_mask,
                use_cache=False,
                return_dict=True,
                **run_inputs.model_kwargs,
            )

            if hasattr(outputs, "logits") and outputs.logits is not None:
                return outputs.logits
            if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                return outputs.last_hidden_state

            raise RuntimeError("HF model forward output does not expose logits or last_hidden_state")
        finally:
            for handle in reversed(handles):
                handle.remove()

    def zero_grad(self) -> None:
        self.model.zero_grad(set_to_none=True)

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        return self.model.parameters()


def resolve_backend(model: Any, backend: Optional[ModelBackend]) -> ModelBackend:
    if backend is not None:
        return backend

    if _is_hooked_transformer_model(model):
        return TLensBackend(model)

    raise ValueError(
        "A backend must be provided when model is not a transformer_lens.HookedTransformer. "
        "Instantiate HFLLMBackend(model=...) or TLensBackend(model=...) and pass backend=..."
    )


def is_hf_backend(backend: ModelBackend) -> bool:
    return getattr(backend, "kind", "") == "hf"
