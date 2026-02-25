from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ProjectionAttentionOperator:
    n_heads: int
    d_model: int
    arch_kind: str
    projection_kind: str

    def _qkv_letter_from_hook_name(self, hook_name: str) -> str:
        if hook_name.endswith(".hook_q_input"):
            return "q"
        if hook_name.endswith(".hook_k_input"):
            return "k"
        if hook_name.endswith(".hook_v_input"):
            return "v"
        raise ValueError(f"Cannot infer qkv letter from hook name: {hook_name}")

    def _fused_linear_qkv_slice(
        self,
        *,
        out_dim: int,
        hook_name: str,
    ) -> tuple[int, int, int]:
        d_head = self.d_model // self.n_heads
        q_dim = self.d_model
        if out_dim == 3 * self.d_model:
            kv_dim = self.d_model
        elif out_dim > self.d_model and (out_dim - self.d_model) % (2 * d_head) == 0:
            kv_heads = (out_dim - self.d_model) // (2 * d_head)
            kv_dim = kv_heads * d_head
        else:
            raise RuntimeError(
                f"{hook_name} unsupported fused-linear qkv output dim={out_dim} for d_model={self.d_model}"
            )

        if hook_name.endswith(".hook_q_input"):
            return 0, q_dim, self.n_heads
        if hook_name.endswith(".hook_k_input"):
            return q_dim, q_dim + kv_dim, kv_dim // d_head
        if hook_name.endswith(".hook_v_input"):
            return q_dim + kv_dim, q_dim + 2 * kv_dim, kv_dim // d_head
        raise ValueError(f"Cannot infer qkv letter from hook name: {hook_name}")

    def project_attention_result_from_proj_input(
        self,
        projection_input: Tensor,
        projection_module: torch.nn.Module,
        hook_name: str,
    ) -> Tensor:
        if projection_input.ndim != 3:
            raise RuntimeError(
                f"{hook_name} expected rank-3 projection input [batch, seq, d_model], "
                f"got {tuple(projection_input.shape)}"
            )
        if not hasattr(projection_module, "weight"):
            raise RuntimeError(f"{hook_name} projection module missing weight parameter")

        weight = projection_module.weight
        if weight.ndim != 2:
            raise RuntimeError(f"{hook_name} projection weight must be rank-2, got {weight.ndim}")

        d_model = int(projection_input.shape[-1])
        if d_model != self.d_model:
            raise RuntimeError(
                f"{hook_name} projection input dim mismatch: d_model={d_model}, expected={self.d_model}"
            )
        if d_model % self.n_heads != 0:
            raise RuntimeError(
                f"{hook_name} cannot split d_model={d_model} into n_heads={self.n_heads}"
            )
        d_head = d_model // self.n_heads
        if int(weight.shape[1]) != d_model:
            if not (
                self.arch_kind == "gpt2_like"
                and projection_module.__class__.__name__ == "Conv1D"
                and int(weight.shape[0]) == d_model
                and int(weight.shape[1]) == d_model
            ):
                raise RuntimeError(
                    f"{hook_name} projection weight input dim mismatch: "
                    f"weight_in={int(weight.shape[1])}, d_model={d_model}"
                )

        projection_heads = projection_input.view(
            projection_input.shape[0],
            projection_input.shape[1],
            self.n_heads,
            d_head,
        )
        if self.arch_kind == "gpt2_like" and projection_module.__class__.__name__ == "Conv1D":
            weight_heads = weight.view(self.n_heads, d_head, d_model)
            return torch.einsum("bphd,hdm->bphm", projection_heads, weight_heads)

        weight_heads = weight.view(int(weight.shape[0]), self.n_heads, d_head)
        return torch.einsum("bphd,ohd->bpho", projection_heads, weight_heads)

    def replace_projection_input_from_attn_result(
        self,
        *,
        original_projection_input: Tensor,
        updated_projected_output: Tensor,
        projection_module: torch.nn.Module,
        hook_name: str,
    ) -> Tensor:
        if updated_projected_output.ndim != 4:
            raise RuntimeError(
                f"{hook_name} expected rank-4 updated attn result [batch, pos, heads, d_model], "
                f"got {tuple(updated_projected_output.shape)}"
            )

        d_model = int(original_projection_input.shape[-1])
        if d_model != self.d_model:
            raise RuntimeError(
                f"{hook_name} projection input dim mismatch: d_model={d_model}, expected={self.d_model}"
            )
        if d_model % self.n_heads != 0:
            raise RuntimeError(
                f"{hook_name} cannot split d_model={d_model} into n_heads={self.n_heads}"
            )
        d_head = d_model // self.n_heads
        weight = projection_module.weight
        if weight.ndim != 2:
            raise RuntimeError(f"{hook_name} projection weight must be rank-2, got {weight.ndim}")

        if projection_module.__class__.__name__ == "Conv1D":
            head_projectors = weight.view(self.n_heads, d_head, d_model)
        else:
            head_projectors = (
                weight.view(d_model, self.n_heads, d_head)
                .permute(1, 2, 0)
                .contiguous()
            )

        pinv_heads = torch.linalg.pinv(head_projectors.to(dtype=torch.float64))
        replaced_heads = torch.einsum(
            "bpho,hod->bphd",
            updated_projected_output.to(dtype=torch.float64),
            pinv_heads,
        ).to(dtype=weight.dtype, device=weight.device)
        replaced = replaced_heads.reshape(
            updated_projected_output.shape[0],
            updated_projected_output.shape[1],
            d_model,
        )

        if replaced.shape != original_projection_input.shape:
            raise RuntimeError(
                f"{hook_name} replacement input shape mismatch: "
                f"expected {tuple(original_projection_input.shape)}, got {tuple(replaced.shape)}"
            )
        return replaced

    def project_qkv_input_for_hook(
        self,
        qkv_input: Tensor,
        hook_name: str,
    ) -> Tensor:
        if qkv_input.ndim != 3:
            raise RuntimeError(
                f"{hook_name} expected rank-3 projection input [batch, seq, d_model], "
                f"got {tuple(qkv_input.shape)}"
            )
        return qkv_input.unsqueeze(2).repeat(1, 1, self.n_heads, 1)

    def replace_projection_output_from_qkv_input(
        self,
        *,
        original_projected_input: Tensor,
        updated_projected_input: Tensor,
        projection_output: Tensor,
        projection_module: torch.nn.Module,
        hook_name: str,
    ) -> Tensor:
        if updated_projected_input.shape != original_projected_input.shape:
            raise RuntimeError(
                f"{hook_name} replacement shape mismatch: expected {tuple(original_projected_input.shape)}, "
                f"got {tuple(updated_projected_input.shape)}"
            )
        if projection_output.ndim != 3:
            raise RuntimeError(
                f"{hook_name} expected rank-3 projection output [batch, seq, dim], "
                f"got {tuple(projection_output.shape)}"
            )

        delta_heads = updated_projected_input - original_projected_input
        projection_spec = self.projection_kind

        if projection_spec == "fused_linear_interleaved":
            raise RuntimeError(
                f"{hook_name} uses interleaved fused-linear qkv layout not supported in this stage"
            )

        if projection_spec == "fused_conv1d" and hasattr(projection_module, "weight"):
            weight = projection_module.weight
            d_head = self.d_model // self.n_heads
            if int(weight.shape[0]) != self.d_model or int(weight.shape[1]) != 3 * self.d_model:
                raise RuntimeError(
                    f"{hook_name} unsupported Conv1D weight shape for replacement: {tuple(weight.shape)}"
                )

            qkv_letter = self._qkv_letter_from_hook_name(hook_name)
            qkv_index = "qkv".index(qkv_letter)
            start = qkv_index * self.d_model
            end = (qkv_index + 1) * self.d_model

            weight_chunk = weight[:, start:end].view(self.d_model, self.n_heads, d_head)
            delta_chunk = torch.einsum("bphm,mhd->bphd", delta_heads, weight_chunk).reshape(
                projection_output.shape[0],
                projection_output.shape[1],
                self.d_model,
            )
            replaced = projection_output.clone()
            replaced[:, :, start:end] = replaced[:, :, start:end] + delta_chunk
            return replaced

        if not hasattr(projection_module, "weight"):
            raise RuntimeError(f"{hook_name} projection module missing weight for replacement")

        weight = projection_module.weight
        if weight.ndim != 2:
            raise RuntimeError(
                f"{hook_name} projection weight must be rank-2 for replacement, got {weight.ndim}"
            )

        out_dim = int(weight.shape[0])
        in_dim = int(weight.shape[1])
        if in_dim != self.d_model:
            raise RuntimeError(
                f"{hook_name} projection input dim mismatch for replacement: "
                f"weight_in={in_dim}, expected={self.d_model}"
            )

        d_head = self.d_model // self.n_heads
        if projection_spec == "fused_linear":
            start, end, out_heads = self._fused_linear_qkv_slice(out_dim=out_dim, hook_name=hook_name)
            weight = weight[start:end, :]
            projection_output = projection_output.clone()
            chunk_output = projection_output[:, :, start:end]
        else:
            if out_dim % d_head != 0:
                raise RuntimeError(
                    f"{hook_name} projection output dim {out_dim} is not divisible by d_head={d_head}"
                )
            out_heads = out_dim // d_head
            chunk_output = projection_output
            start = 0
            end = out_dim

        if out_heads == self.n_heads:
            grouped_delta = delta_heads
        elif self.n_heads % out_heads == 0:
            group_size = self.n_heads // out_heads
            grouped_delta = delta_heads.reshape(
                delta_heads.shape[0],
                delta_heads.shape[1],
                out_heads,
                group_size,
                delta_heads.shape[-1],
            ).mean(dim=3)
        else:
            raise RuntimeError(
                f"{hook_name} cannot map n_heads={self.n_heads} to out_heads={out_heads}"
            )

        weight_heads = weight.view(out_heads, d_head, self.d_model)
        delta_out_heads = torch.einsum("bphm,hdm->bphd", grouped_delta, weight_heads)
        delta_out = delta_out_heads.reshape(
            projection_output.shape[0],
            projection_output.shape[1],
            end - start,
        )
        if projection_spec == "fused_linear":
            projection_output[:, :, start:end] = chunk_output + delta_out
            return projection_output
        return projection_output + delta_out

    def project_qkv_gradient_to_input_heads(
        self,
        grad_output: Tensor,
        projection_module: torch.nn.Module,
        hook_name: str,
    ) -> Tensor:
        if grad_output.ndim != 3:
            raise RuntimeError(
                f"{hook_name} expected rank-3 projection output gradients [batch, seq, dim], "
                f"got {tuple(grad_output.shape)}"
            )
        if not hasattr(projection_module, "weight"):
            raise RuntimeError(f"{hook_name} projection module missing weight parameter")

        weight = projection_module.weight
        if weight.ndim != 2:
            raise RuntimeError(f"{hook_name} projection weight must be rank-2, got {weight.ndim}")

        qkv_letter = self._qkv_letter_from_hook_name(hook_name)
        qkv_index = "qkv".index(qkv_letter)
        d_head = self.d_model // self.n_heads
        projection_spec = self.projection_kind

        if projection_spec == "fused_linear_interleaved":
            raise RuntimeError(
                f"{hook_name} uses interleaved fused-linear qkv layout not supported in this stage"
            )

        if projection_spec == "fused_conv1d" and int(weight.shape[0]) == self.d_model and int(weight.shape[1]) == 3 * self.d_model:
            start = qkv_index * self.d_model
            end = (qkv_index + 1) * self.d_model
            grad_chunk = grad_output[:, :, start:end]
            grad_heads = grad_chunk.view(
                grad_chunk.shape[0],
                grad_chunk.shape[1],
                self.n_heads,
                d_head,
            )
            weight_chunk = weight[:, start:end].view(self.d_model, self.n_heads, d_head)
            return torch.einsum("bphd,mhd->bphm", grad_heads, weight_chunk)

        out_dim = int(weight.shape[0])
        in_dim = int(weight.shape[1])
        if in_dim != self.d_model:
            raise RuntimeError(
                f"{hook_name} projection input dim mismatch: weight_in={in_dim}, expected={self.d_model}"
            )
        if projection_spec == "fused_linear":
            start, end, out_heads = self._fused_linear_qkv_slice(out_dim=out_dim, hook_name=hook_name)
            grad_output = grad_output[:, :, start:end]
            weight = weight[start:end, :]
        else:
            if out_dim % d_head != 0:
                raise RuntimeError(
                    f"{hook_name} projection output dim {out_dim} is not divisible by d_head={d_head}"
                )
            out_heads = out_dim // d_head
        grad_heads = grad_output.view(
            grad_output.shape[0],
            grad_output.shape[1],
            out_heads,
            d_head,
        )
        weight_heads = weight.view(out_heads, d_head, self.d_model)
        per_head_grads = torch.einsum("bphd,hdm->bphm", grad_heads, weight_heads)
        if out_heads == self.n_heads:
            return per_head_grads
        if self.n_heads % out_heads != 0:
            raise RuntimeError(
                f"{hook_name} cannot broadcast out_heads={out_heads} to n_heads={self.n_heads}"
            )
        repeat_factor = self.n_heads // out_heads
        return per_head_grads.repeat_interleave(repeat_factor, dim=2)
