from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from .base import (
    AdapterResolution,
    ArchitectureAdapter,
    BackendConfig,
    BackendHookSpec,
    BackendRunInputs,
    HookTarget,
    first_parameter_device_dtype,
)
from .operators import ProjectionAttentionOperator
from .registry import (
    AdapterType,
    ResolutionError,
    inspect_model_architecture,
    instantiate_adapters,
    resolve_adapter_resolution,
)


class HFLLMBackend:
    kind = "hf"

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        tokenizer=None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        ungroup_gqa: bool = True,
        adapter_registry: Optional[Sequence[AdapterType]] = None,
        adapter_name: Optional[str] = None,
        language_trunk_path: Optional[str] = None,
        strict_arch: bool = True,
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
        self._strict_arch = strict_arch
        self._adapter_registry = instantiate_adapters(adapter_registry)
        self._adapter, resolution, diagnostics = self._resolve_architecture(
            self.model,
            adapter_name=adapter_name,
            language_trunk_path=language_trunk_path,
        )
        self._base_model = resolution.base_model
        self._language_trunk_path = resolution.path
        self._arch_kind = resolution.arch_kind
        self._resolution_diagnostics = diagnostics
        self._layers = resolution.layers
        self._layer_accessors = resolution.layer_accessors
        self._embed_module = resolution.embed_module
        self._resid_module = resolution.resid_module
        self._qkv_operators: Dict[str, ProjectionAttentionOperator] = {}
        self._attn_result_operators: Dict[str, ProjectionAttentionOperator] = {}
        self._config = self._normalize_config(self.model)
        if ungroup_gqa:
            self._maybe_ungroup_gqa_inplace()
        self._hook_targets = self._build_hook_targets()
        self._ln1_residual_cache: Dict[int, Tensor] = {}

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

    @property
    def adapter_name(self) -> str:
        return self._adapter.name

    @property
    def language_trunk_path(self) -> str:
        return self._language_trunk_path

    @property
    def arch_kind(self) -> str:
        return self._arch_kind

    @property
    def resolution_diagnostics(self) -> Dict[str, Any]:
        return dict(self._resolution_diagnostics)

    def _diagnostics_summary(self, diagnostics: Dict[str, Any], *, max_items: int = 5) -> str:
        selected = diagnostics.get("selected")
        attempts = diagnostics.get("adapter_attempts", [])
        parts: List[str] = []
        if isinstance(selected, dict):
            parts.append(
                "selected="
                f"{selected.get('adapter', '?')}@{selected.get('path', '?')}"
            )
        if attempts:
            failed = [
                f"{item.get('adapter', '?')}@{item.get('path', '?')}:{item.get('detail', '')}"
                for item in attempts
                if item.get("status") != "match"
            ]
            if failed:
                parts.append("attempts=" + " | ".join(failed[:max_items]))
        if not parts:
            return "no adapter diagnostics available"
        return "; ".join(parts)

    def _expand_projection_to_full_heads(
        self,
        proj: torch.nn.Module,
        *,
        repeats: int,
        new_n_heads: int,
        old_n_heads: int,
        head_dim: int,
    ) -> torch.nn.Module:
        if not isinstance(proj, torch.nn.Linear):
            raise RuntimeError(
                "GQA ungroup currently supports nn.Linear k/v projections only; "
                f"got {proj.__class__.__name__}"
            )

        with torch.no_grad():
            weight = proj.weight.detach().clone()
            if weight.shape[0] != old_n_heads * head_dim:
                raise RuntimeError(
                    "Projection output shape is incompatible with head_dim: "
                    f"out={weight.shape[0]} old_n_heads={old_n_heads} head_dim={head_dim}"
                )

            expanded_weight = (
                weight.view(old_n_heads, head_dim, weight.shape[1])
                .repeat_interleave(repeats, dim=0)
                .reshape(new_n_heads * head_dim, weight.shape[1])
            )
            if expanded_weight.shape[0] != new_n_heads * head_dim:
                raise RuntimeError(
                    "Expanded projection weight has unexpected shape: "
                    f"{tuple(expanded_weight.shape)} expected out={new_n_heads * head_dim}"
                )
            proj.weight = torch.nn.Parameter(expanded_weight.to(weight.device, weight.dtype))

            if proj.bias is not None:
                bias = proj.bias.detach().clone()
                if bias.shape[0] != old_n_heads * head_dim:
                    raise RuntimeError(
                        "Projection bias shape is incompatible with head_dim: "
                        f"out={bias.shape[0]} old_n_heads={old_n_heads} head_dim={head_dim}"
                    )
                expanded_bias = (
                    bias.view(old_n_heads, head_dim)
                    .repeat_interleave(repeats, dim=0)
                    .reshape(new_n_heads * head_dim)
                )
                proj.bias = torch.nn.Parameter(expanded_bias.to(bias.device, bias.dtype))

            proj.out_features = new_n_heads * head_dim
        return proj

    def _maybe_ungroup_gqa_inplace(self) -> None:
        if not self._adapter.supports_gqa_ungroup:
            return
        # Mirror TransformerLens ungroup_grouped_query_attention semantics by expanding
        # k/v projections to n_heads and disabling repeat_kv groups in attention modules.
        attn_attr = self._layer_accessors["attn"]
        changed = False
        for layer in self._layers:
            attn = getattr(layer, attn_attr)
            head_dim = getattr(attn, "head_dim", None)
            if head_dim is None and hasattr(attn, "config"):
                cfg_head_dim = getattr(attn.config, "head_dim", None)
                if cfg_head_dim is not None:
                    head_dim = cfg_head_dim
            if head_dim is None and hasattr(attn, "q_proj"):
                hidden = int(getattr(attn.q_proj, "in_features", 0))
                n_attn_cfg = getattr(getattr(attn, "config", None), "num_attention_heads", None)
                if n_attn_cfg:
                    head_dim = hidden // int(n_attn_cfg)

            n_heads = None
            if head_dim is not None and hasattr(attn, "q_proj"):
                n_heads = int(attn.q_proj.out_features) // int(head_dim)
            if n_heads is None:
                n_heads = getattr(attn, "num_heads", None)
            if n_heads is None and hasattr(attn, "config"):
                n_heads = getattr(attn.config, "num_attention_heads", None)

            n_kv_heads = None
            if head_dim is not None and hasattr(attn, "k_proj"):
                n_kv_heads = int(attn.k_proj.out_features) // int(head_dim)
            if n_kv_heads is None:
                n_kv_heads = getattr(attn, "num_key_value_heads", None)
            if n_kv_heads is None and hasattr(attn, "config"):
                n_kv_heads = getattr(attn.config, "num_key_value_heads", None)

            if n_heads is None or n_kv_heads is None or head_dim is None:
                continue
            if not isinstance(n_heads, int) or not isinstance(n_kv_heads, int):
                continue
            if n_kv_heads <= 0 or n_kv_heads == n_heads:
                continue
            if n_heads % n_kv_heads != 0:
                raise RuntimeError(
                    f"Cannot ungroup GQA when n_heads={n_heads} is not divisible by n_kv_heads={n_kv_heads}"
                )
            if not hasattr(attn, "k_proj") or not hasattr(attn, "v_proj"):
                continue

            repeats = n_heads // n_kv_heads
            self._expand_projection_to_full_heads(
                attn.k_proj,
                repeats=repeats,
                new_n_heads=n_heads,
                old_n_heads=n_kv_heads,
                head_dim=int(head_dim),
            )
            self._expand_projection_to_full_heads(
                attn.v_proj,
                repeats=repeats,
                new_n_heads=n_heads,
                old_n_heads=n_kv_heads,
                head_dim=int(head_dim),
            )

            attn.num_key_value_heads = n_heads
            if hasattr(attn, "num_key_value_groups"):
                attn.num_key_value_groups = 1
            changed = True

        if changed:
            # Keep top-level config metadata unchanged; only attention module internals are ungrouped.
            return

    def _resolve_architecture(
        self,
        model: torch.nn.Module,
        *,
        adapter_name: Optional[str] = None,
        language_trunk_path: Optional[str] = None,
    ) -> Tuple[ArchitectureAdapter, AdapterResolution, Dict[str, Any]]:
        try:
            adapter, resolved, diagnostics = resolve_adapter_resolution(
                model,
                self._adapter_registry,
                adapter_name=adapter_name,
                language_trunk_path=language_trunk_path,
            )
            return adapter, resolved, diagnostics
        except ResolutionError as exc:
            if self._strict_arch:
                summary = self._diagnostics_summary(exc.diagnostics)
                raise ValueError(f"{exc} Resolution summary: {summary}") from exc
            diag = inspect_model_architecture(
                model,
                self._adapter_registry,
                adapter_name=adapter_name,
                language_trunk_path=language_trunk_path,
            )
            selected = diag.get("selected")
            if selected is not None:
                adapter, resolved, diagnostics = resolve_adapter_resolution(
                    model,
                    self._adapter_registry,
                    adapter_name=selected["adapter"],
                    language_trunk_path=selected.get("path"),
                )
                return adapter, resolved, diagnostics
            raise ValueError(f"{exc} Diagnostics: {diag.get('selection_error', 'no match')}") from exc

    def _normalize_config(self, model: torch.nn.Module) -> BackendConfig:
        cfg = model.config
        text_cfg = getattr(cfg, "text_config", None)
        source_cfg = text_cfg if text_cfg is not None else cfg
        d_model = (
            getattr(source_cfg, "hidden_size", None)
            if getattr(source_cfg, "hidden_size", None) is not None
            else (
                getattr(source_cfg, "n_embd", None)
                if getattr(source_cfg, "n_embd", None) is not None
                else getattr(source_cfg, "d_model", None)
            )
        )
        n_layers = (
            getattr(source_cfg, "num_hidden_layers", None)
            if getattr(source_cfg, "num_hidden_layers", None) is not None
            else (
                getattr(source_cfg, "n_layer", None)
                if getattr(source_cfg, "n_layer", None) is not None
                else getattr(source_cfg, "n_layers", None)
            )
        )
        n_heads = (
            getattr(source_cfg, "num_attention_heads", None)
            if getattr(source_cfg, "num_attention_heads", None) is not None
            else (
                getattr(source_cfg, "n_head", None)
                if getattr(source_cfg, "n_head", None) is not None
                else getattr(source_cfg, "n_heads", None)
            )
        )
        missing_cfg = []
        if d_model is None:
            missing_cfg.append("hidden_size/n_embd/d_model")
        if n_layers is None:
            missing_cfg.append("num_hidden_layers/n_layer/n_layers")
        if n_heads is None:
            missing_cfg.append("num_attention_heads/n_head/n_heads")
        if missing_cfg:
            raise ValueError(
                "Unsupported HF config for HFLLMBackend. Missing fields: "
                f"{missing_cfg}."
            )

        device, model_dtype = first_parameter_device_dtype(self.model)
        n_key_value_heads = getattr(source_cfg, "num_key_value_heads", None)

        return BackendConfig(
            device=device,
            dtype=model_dtype,
            d_model=int(d_model),
            n_layers=int(n_layers),
            n_heads=int(n_heads),
            parallel_attn_mlp=False,
            n_key_value_heads=(None if n_key_value_heads is None else int(n_key_value_heads)),
            use_attn_result=True,
            use_split_qkv_input=True,
            use_hook_mlp_in=True,
            use_normalization_before_and_after=False,
        )

    def _build_hook_targets(self) -> Dict[str, HookTarget]:
        targets: Dict[str, HookTarget] = {"hook_embed": HookTarget(self._embed_module, "forward")}

        attn_attr = self._layer_accessors["attn"]
        mlp_attr = self._layer_accessors["mlp"]
        for layer_idx, layer in enumerate(self._layers):
            attn_module = getattr(layer, attn_attr)
            mlp_module = getattr(layer, mlp_attr)
            ln2_module = getattr(layer, self._layer_accessors["ln2"])

            attn_result_module = self._adapter.attn_result_module(attn_module)
            targets[f"blocks.{layer_idx}.attn.hook_result"] = HookTarget(
                attn_result_module,
                "pre",
            )
            attn_hook_name = f"blocks.{layer_idx}.attn.hook_result"
            self._attn_result_operators[attn_hook_name] = self._adapter.attention_operator(
                attn_module,
                qkv="q",
                backend_config=self._config,
            )
            qkv_modules = self._adapter.qkv_hook_modules(attn_module)
            for qkv in ("q", "k", "v"):
                hook_name = f"blocks.{layer_idx}.hook_{qkv}_input"
                targets[hook_name] = HookTarget(qkv_modules[qkv], "forward")
                self._qkv_operators[hook_name] = self._adapter.attention_operator(
                    attn_module,
                    qkv=qkv,
                    backend_config=self._config,
                )

            targets[f"blocks.{layer_idx}.hook_mlp_out"] = HookTarget(mlp_module, "forward")
            targets[f"blocks.{layer_idx}.hook_mlp_in"] = HookTarget(ln2_module, "forward")

        targets[f"blocks.{len(self._layers) - 1}.hook_resid_post"] = HookTarget(
            self._resid_module,
            "pre",
        )

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
        core_keys = {"input_ids", "tokens", "inputs_embeds", "attention_mask"}
        for key, value in inputs.items():
            if key in core_keys:
                continue
            if torch.is_tensor(value):
                if torch.is_floating_point(value):
                    model_kwargs[key] = value.to(device=device, dtype=dtype)
                else:
                    model_kwargs[key] = value.to(device=device)
            else:
                model_kwargs[key] = value

        return BackendRunInputs(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            extra_fwd_hooks=[],
            model_kwargs=model_kwargs,
        )

    def _make_forward_hook(self, name: str, hook_fn: Callable, target: HookTarget):
        return self._make_composed_forward_hook(name, target, [hook_fn], [])

    def _apply_forward_hook_chain(
        self,
        activation: Tensor,
        hook_ctx: SimpleNamespace,
        fwd_hook_fns: List[Callable],
    ) -> Tensor:
        updated = activation
        for hook_fn in fwd_hook_fns:
            maybe_output = hook_fn(updated, hook_ctx)
            if maybe_output is not None:
                updated = maybe_output
        return updated

    def _attach_backward_hook_chain(
        self,
        activation: Tensor,
        hook_ctx: SimpleNamespace,
        bwd_hook_fns: List[Callable],
    ) -> None:
        if not bwd_hook_fns:
            return
        if not torch.is_tensor(activation):
            raise RuntimeError(
                f"Backward hooks for {hook_ctx.name} require tensor activation, got {type(activation)}"
            )
        if not activation.requires_grad:
            return

        def _bwd_chain(grad: Tensor) -> Tensor:
            updated = grad
            for hook_fn in bwd_hook_fns:
                maybe_grad = hook_fn(updated, hook_ctx)
                if maybe_grad is not None:
                    updated = maybe_grad
            return updated

        activation.register_hook(_bwd_chain)

    def _make_composed_forward_hook(
        self,
        name: str,
        target: HookTarget,
        fwd_hook_fns: List[Callable],
        bwd_hook_fns: List[Callable],
    ):
        hook_ctx = SimpleNamespace(name=name)

        def _wrapped(module, inputs, output):
            if name.endswith(".hook_mlp_in"):
                if len(inputs) == 0:
                    raise RuntimeError(f"{name} expects at least one module input tensor")
                ln_input = inputs[0]
                original_ln_input = ln_input.clone()
                updated = self._apply_forward_hook_chain(ln_input, hook_ctx, fwd_hook_fns)
                ln_output = output
                input_changed = (updated is not ln_input) or (not torch.equal(updated, original_ln_input))
                if input_changed:
                    ln_output = self._apply_norm_module(module, updated)
                self._attach_mlp_in_backward_hook_from_ln_output(
                    ln_output=ln_output,
                    ln_input=updated,
                    ln_module=module,
                    hook_name=name,
                    hook_ctx=hook_ctx,
                    bwd_hook_fns=bwd_hook_fns,
                )
                return ln_output

            if (
                name.endswith(".hook_q_input")
                or name.endswith(".hook_k_input")
                or name.endswith(".hook_v_input")
            ):
                if len(inputs) == 0:
                    raise RuntimeError(f"{name} expects at least one module input tensor")
                projected = self._project_qkv_input_for_hook(inputs[0], name)
                original_projected = projected.clone()
                updated = self._apply_forward_hook_chain(projected, hook_ctx, fwd_hook_fns)
                projection_output = output
                input_changed = (updated is not projected) or (not torch.equal(updated, original_projected))
                if input_changed:
                    projection_output = self._replace_projection_output_from_qkv_input(
                        original_projected_input=projected,
                        updated_projected_input=updated,
                        projection_output=output,
                        projection_module=module,
                        hook_name=name,
                    )
                self._attach_qkv_backward_hook_from_projection_output(
                    projection_output=projection_output,
                    projection_module=module,
                    hook_name=name,
                    hook_ctx=hook_ctx,
                    bwd_hook_fns=bwd_hook_fns,
                )
                return projection_output

            if target.output_index is None:
                updated = self._apply_forward_hook_chain(output, hook_ctx, fwd_hook_fns)
                self._attach_backward_hook_chain(updated, hook_ctx, bwd_hook_fns)
                return updated

            if not isinstance(output, (tuple, list)):
                raise RuntimeError(
                    f"Hook point {name} expected tuple/list output for index "
                    f"{target.output_index}, got {type(output)}"
                )
            if not (0 <= target.output_index < len(output)):
                raise RuntimeError(
                    f"Hook point {name} output_index={target.output_index} out of range "
                    f"for output length {len(output)}"
                )

            selected = output[target.output_index]
            updated = self._apply_forward_hook_chain(selected, hook_ctx, fwd_hook_fns)
            self._attach_backward_hook_chain(updated, hook_ctx, bwd_hook_fns)
            if updated is selected:
                return output

            output_items = list(output)
            output_items[target.output_index] = updated
            if isinstance(output, tuple):
                return tuple(output_items)
            return output_items

        return _wrapped

    def _make_composed_pre_hook(
        self,
        name: str,
        fwd_hook_fns: List[Callable],
        bwd_hook_fns: List[Callable],
    ):
        hook_ctx = SimpleNamespace(name=name)

        def _wrapped(module, inputs):
            if len(inputs) == 0:
                return None

            if name.endswith(".attn.hook_result"):
                current = inputs[0]
                projected = self._project_attention_result_from_proj_input(current, module, name)
                original_projected = projected.clone()
                updated = self._apply_forward_hook_chain(projected, hook_ctx, fwd_hook_fns)
                self._attach_backward_hook_chain(updated, hook_ctx, bwd_hook_fns)
                output_changed = (updated is not projected) or (not torch.equal(updated, original_projected))
                if not output_changed:
                    return None
                replaced_input = self._replace_projection_input_from_attn_result(
                    original_projection_input=current,
                    updated_projected_output=updated,
                    projection_module=module,
                    hook_name=name,
                )
                return (replaced_input, *inputs[1:])

            if (
                name.endswith(".hook_q_input")
                or name.endswith(".hook_k_input")
                or name.endswith(".hook_v_input")
            ):
                raise RuntimeError(f"{name} expected forward hook target, got pre-hook target")

            current = inputs[0]
            updated = self._apply_forward_hook_chain(current, hook_ctx, fwd_hook_fns)
            self._attach_backward_hook_chain(updated, hook_ctx, bwd_hook_fns)
            if updated is current:
                return None
            return (updated, *inputs[1:])

        return _wrapped

    def _module_pinv(self, module: torch.nn.Module) -> Tensor:
        if not hasattr(self, "_module_pinv_cache"):
            self._module_pinv_cache = {}
        cache = self._module_pinv_cache
        key = id(module)
        if key not in cache:
            if not hasattr(module, "weight"):
                raise RuntimeError(
                    f"Cannot compute pseudo-inverse for module without weight: {module.__class__.__name__}"
                )
            weight = module.weight.detach().float()
            if module.__class__.__name__ == "Conv1D":
                cache[key] = torch.linalg.pinv(weight).to(module.weight.device, module.weight.dtype)
            else:
                cache[key] = torch.linalg.pinv(weight.transpose(0, 1)).to(
                    module.weight.device,
                    module.weight.dtype,
                )
        return cache[key]

    def _replace_projection_input_from_attn_result(
        self,
        *,
        original_projection_input: Tensor,
        updated_projected_output: Tensor,
        projection_module: torch.nn.Module,
        hook_name: str,
    ) -> Tensor:
        operator = self._attn_result_operators.get(hook_name)
        if operator is None:
            raise RuntimeError(f"Missing attention-result operator for hook {hook_name}")
        return operator.replace_projection_input_from_attn_result(
            original_projection_input=original_projection_input,
            updated_projected_output=updated_projected_output,
            projection_module=projection_module,
            hook_name=hook_name,
        )

    def _project_attention_result_from_proj_input(
        self,
        projection_input: Tensor,
        projection_module: torch.nn.Module,
        hook_name: str,
    ) -> Tensor:
        operator = self._attn_result_operators.get(hook_name)
        if operator is None:
            raise RuntimeError(f"Missing attention-result operator for hook {hook_name}")
        return operator.project_attention_result_from_proj_input(
            projection_input=projection_input,
            projection_module=projection_module,
            hook_name=hook_name,
        )

    def _project_qkv_input_for_hook(
        self,
        qkv_input: Tensor,
        hook_name: str,
    ) -> Tensor:
        operator = self._qkv_operators.get(hook_name)
        if operator is None:
            raise RuntimeError(f"Missing qkv operator for hook {hook_name}")
        return operator.project_qkv_input_for_hook(qkv_input=qkv_input, hook_name=hook_name)

    def _apply_norm_module(self, module: torch.nn.Module, inputs: Tensor) -> Tensor:
        if isinstance(module, torch.nn.LayerNorm):
            return F.layer_norm(
                inputs,
                module.normalized_shape,
                module.weight,
                module.bias,
                module.eps,
            )
        if hasattr(module, "weight"):
            eps = float(getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6)))
            variance = inputs.float().pow(2).mean(-1, keepdim=True)
            hidden_states = inputs.float() * torch.rsqrt(variance + eps)
            output = hidden_states.to(inputs.dtype) * module.weight
            if getattr(module, "bias", None) is not None:
                output = output + module.bias
            return output
        raise RuntimeError(
            f"Unsupported normalization module for hook_mlp_in replacement: {module.__class__.__name__}"
        )

    def _replace_projection_output_from_qkv_input(
        self,
        *,
        original_projected_input: Tensor,
        updated_projected_input: Tensor,
        projection_output: Tensor,
        projection_module: torch.nn.Module,
        hook_name: str,
    ) -> Tensor:
        operator = self._qkv_operators.get(hook_name)
        if operator is None:
            raise RuntimeError(f"Missing qkv operator for hook {hook_name}")
        return operator.replace_projection_output_from_qkv_input(
            original_projected_input=original_projected_input,
            updated_projected_input=updated_projected_input,
            projection_output=projection_output,
            projection_module=projection_module,
            hook_name=hook_name,
        )

    def _layer_index_from_hook_name(self, hook_name: str) -> int:
        parts = hook_name.split(".")
        if len(parts) < 2 or parts[0] != "blocks":
            raise ValueError(f"Cannot infer layer index from hook name: {hook_name}")
        return int(parts[1])

    def _layernorm_input_grads(
        self,
        grads_wrt_ln_out: Tensor,
        ln_input: Tensor,
        layernorm_module: torch.nn.Module,
    ) -> Tensor:
        if not isinstance(layernorm_module, torch.nn.LayerNorm):
            # RMSNorm-style modules (e.g. Llama/Qwen) typically expose only a weight parameter.
            if hasattr(layernorm_module, "weight"):
                ln_input_fp = ln_input.float().unsqueeze(2)
                grads_fp = grads_wrt_ln_out.float()
                weight = layernorm_module.weight.float().view(1, 1, 1, -1)
                eps = float(getattr(layernorm_module, "variance_epsilon", getattr(layernorm_module, "eps", 1e-6)))

                gw = grads_fp * weight
                variance = ln_input_fp.pow(2).mean(dim=-1, keepdim=True)
                inv_rms = torch.rsqrt(variance + eps)
                mean_gw_x = (gw * ln_input_fp).mean(dim=-1, keepdim=True)
                grads_ln_in = gw * inv_rms - ln_input_fp * inv_rms.pow(3) * mean_gw_x
                return grads_ln_in.to(dtype=grads_wrt_ln_out.dtype)
            return grads_wrt_ln_out

        ln_input_fp = ln_input.float()
        grads_fp = grads_wrt_ln_out.float()
        weight = layernorm_module.weight.float()
        eps = float(layernorm_module.eps)
        n_dim = ln_input_fp.shape[-1]

        mean = ln_input_fp.mean(dim=-1, keepdim=True)
        var = ((ln_input_fp - mean) ** 2).mean(dim=-1, keepdim=True)
        std = torch.sqrt(var + eps)
        xhat = (ln_input_fp - mean) / std

        gw = grads_fp * weight.view(1, 1, 1, -1)
        sum_gw = gw.sum(dim=-1, keepdim=True)
        sum_gw_xhat = (gw * xhat.unsqueeze(2)).sum(dim=-1, keepdim=True)
        grads_ln_in = (1.0 / n_dim) / std.unsqueeze(2) * (
            n_dim * gw - sum_gw - xhat.unsqueeze(2) * sum_gw_xhat
        )
        return grads_ln_in.to(dtype=grads_wrt_ln_out.dtype)

    def _project_qkv_gradient_to_input_heads(
        self,
        grad_output: Tensor,
        projection_module: torch.nn.Module,
        hook_name: str,
    ) -> Tensor:
        operator = self._qkv_operators.get(hook_name)
        if operator is None:
            raise RuntimeError(f"Missing qkv operator for hook {hook_name}")
        projected = operator.project_qkv_gradient_to_input_heads(
            grad_output=grad_output,
            projection_module=projection_module,
            hook_name=hook_name,
        )
        layer_idx = self._layer_index_from_hook_name(hook_name)
        if layer_idx in self._ln1_residual_cache:
            ln_input = self._ln1_residual_cache[layer_idx]
            ln1_module = getattr(self._layers[layer_idx], self._layer_accessors["ln1"])
            return self._layernorm_input_grads(projected, ln_input, ln1_module)
        return projected

    def _attach_qkv_backward_hook_from_projection_output(
        self,
        projection_output: Tensor,
        projection_module: torch.nn.Module,
        hook_name: str,
        hook_ctx: SimpleNamespace,
        bwd_hook_fns: List[Callable],
    ) -> None:
        if not bwd_hook_fns:
            return
        if not torch.is_tensor(projection_output):
            raise RuntimeError(
                f"{hook_name} backward hooks require tensor projection output, got {type(projection_output)}"
            )
        if not projection_output.requires_grad:
            return

        def _bwd_chain(grad: Tensor) -> Tensor:
            projected_grad = self._project_qkv_gradient_to_input_heads(grad, projection_module, hook_name)
            updated = projected_grad
            for hook_fn in bwd_hook_fns:
                maybe_grad = hook_fn(updated, hook_ctx)
                if maybe_grad is not None:
                    updated = maybe_grad
            return grad

        projection_output.register_hook(_bwd_chain)

    def _attach_mlp_in_backward_hook_from_ln_output(
        self,
        ln_output: Tensor,
        ln_input: Tensor,
        ln_module: torch.nn.Module,
        hook_name: str,
        hook_ctx: SimpleNamespace,
        bwd_hook_fns: List[Callable],
    ) -> None:
        if not bwd_hook_fns:
            return
        if not torch.is_tensor(ln_output):
            raise RuntimeError(
                f"{hook_name} backward hooks require tensor layernorm output, got {type(ln_output)}"
            )
        if not ln_output.requires_grad:
            return

        cached_ln_input = ln_input.detach()

        def _bwd_chain(grad: Tensor) -> Tensor:
            projected_grad = self._layernorm_input_grads(
                grad.unsqueeze(2),
                cached_ln_input,
                ln_module,
            ).squeeze(2)
            updated = projected_grad
            for hook_fn in bwd_hook_fns:
                maybe_grad = hook_fn(updated, hook_ctx)
                if maybe_grad is not None:
                    updated = maybe_grad
            return grad

        ln_output.register_hook(_bwd_chain)

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

        merged_bwd_hooks = list(bwd_hooks) if bwd_hooks else []

        if not merged_fwd_hooks and not merged_bwd_hooks:
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

        fwd_by_name: Dict[str, List[Callable]] = {}
        bwd_by_name: Dict[str, List[Callable]] = {}
        hook_order: List[str] = []

        for name, hook_fn in merged_fwd_hooks:
            if name not in fwd_by_name:
                fwd_by_name[name] = []
            fwd_by_name[name].append(hook_fn)
            if name not in hook_order:
                hook_order.append(name)

        for name, hook_fn in merged_bwd_hooks:
            if name not in bwd_by_name:
                bwd_by_name[name] = []
            bwd_by_name[name].append(hook_fn)
            if name not in hook_order:
                hook_order.append(name)

        self._ln1_residual_cache = {}
        handles = []
        try:
            captured_ln1_layers = set()
            for name in hook_order:
                if not (
                    name.endswith(".hook_q_input")
                    or name.endswith(".hook_k_input")
                    or name.endswith(".hook_v_input")
                ):
                    continue
                layer_idx = self._layer_index_from_hook_name(name)
                if layer_idx in captured_ln1_layers:
                    continue
                ln1_module = getattr(self._layers[layer_idx], self._layer_accessors["ln1"])

                def _capture_ln1_input(module, inputs, idx: int = layer_idx):
                    if len(inputs) > 0:
                        self._ln1_residual_cache[idx] = inputs[0].detach()

                handles.append(ln1_module.register_forward_pre_hook(_capture_ln1_input))
                captured_ln1_layers.add(layer_idx)

            for name in hook_order:
                target = self._hook_targets.get(name)
                if target is None:
                    raise ValueError(
                        f"Unsupported HF hook point: {name}. "
                        f"Supported hook names include: {self.supported_hook_names[:12]} ..."
                    )

                fwd_fns = fwd_by_name.get(name, [])
                bwd_fns = bwd_by_name.get(name, [])

                if target.mode == "forward":
                    handle = target.module.register_forward_hook(
                        self._make_composed_forward_hook(name, target, fwd_fns, bwd_fns)
                    )
                elif target.mode == "pre":
                    handle = target.module.register_forward_pre_hook(
                        self._make_composed_pre_hook(name, fwd_fns, bwd_fns)
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
