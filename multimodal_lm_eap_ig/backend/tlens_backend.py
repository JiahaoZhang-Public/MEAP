from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import torch
from torch import Tensor

from .base import (
    BackendConfig,
    BackendHookSpec,
    BackendRunInputs,
    first_parameter_device_dtype,
    is_hooked_transformer_model,
    parse_torch_dtype,
)


def _first_device_dtype(model: Any):
    if hasattr(model, "parameters"):
        return first_parameter_device_dtype(model)
    return torch.device("cpu"), torch.float32


class TLensBackend:
    kind = "tlens"

    def __init__(self, model: Any):
        if not is_hooked_transformer_model(model):
            raise TypeError("TLensBackend requires a transformer_lens.HookedTransformer instance")
        self.model = model
        cfg = model.cfg
        device = torch.device(getattr(cfg, "device", _first_device_dtype(model)[0]))
        dtype = parse_torch_dtype(getattr(cfg, "dtype", _first_device_dtype(model)[1]))
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
                del activations, hook
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
