from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional
from urllib.request import urlopen

import torch

DEFAULT_IMAGE_URL = (
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"
)
DEFAULT_AUDIO_URL = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"
)


class VisualizationDependencyError(RuntimeError):
    """Raised when graph rendering dependencies are missing."""


@dataclass
class GraphArtifacts:
    full_json: str
    topn_json: str
    topn_png: str


@dataclass
class GraphStats:
    n_forward: int
    n_backward: int
    n_edges_total: int
    n_edges_in_topn_graph: int
    score_shape_0: int
    score_shape_1: int


def to_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def classify_error(exc: Exception) -> str:
    text = str(exc).lower()
    if "pygraphviz" in text:
        return "dependency_missing"
    if isinstance(exc, VisualizationDependencyError):
        return "dependency_missing"
    if "urlopen" in text or "http error" in text or "timed out" in text:
        return "sample_fetch"
    if "from_pretrained" in text or "configuration class" in text:
        return "model_load"
    if "unsupported hf architecture" in text:
        return "unsupported_arch"
    if "out of memory" in text or "cuda oom" in text:
        return "oom"
    return "runtime"


def _single_token_id(tokenizer: Any, token: str) -> Optional[int]:
    try:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
    except Exception:
        return None
    if len(token_ids) == 1:
        return int(token_ids[0])
    return None


def resolve_target_pair(tokenizer: Optional[Any], model: Any) -> torch.Tensor:
    if tokenizer is not None:
        candidates = [(" Paris", " Berlin"), (" yes", " no"), (".", ",")]
        for pos, neg in candidates:
            pos_id = _single_token_id(tokenizer, pos)
            neg_id = _single_token_id(tokenizer, neg)
            if pos_id is not None and neg_id is not None:
                return torch.tensor([[pos_id, neg_id]], dtype=torch.long)

    eos_id = getattr(getattr(model, "config", None), "eos_token_id", None)
    vocab_size = int(getattr(getattr(model, "config", None), "vocab_size", 0) or 0)
    if eos_id is not None and vocab_size > 1:
        alt = int((int(eos_id) + 1) % vocab_size)
        return torch.tensor([[int(eos_id), alt]], dtype=torch.long)
    raise ValueError("Could not resolve a valid single-token target pair for metric computation.")


def metric_logit_diff(logits: torch.Tensor, clean_logits: Optional[torch.Tensor], batch) -> torch.Tensor:
    del clean_logits
    labels = batch.labels
    if labels is None:
        raise ValueError("labels are required for metric_logit_diff")

    if not torch.is_tensor(labels):
        labels = torch.tensor(labels, dtype=torch.long, device=logits.device)
    else:
        labels = labels.to(device=logits.device, dtype=torch.long)

    if labels.ndim == 1:
        labels = labels.unsqueeze(0)
    if labels.shape[-1] != 2:
        raise ValueError(f"labels must have shape [batch, 2], got {tuple(labels.shape)}")

    input_lengths = batch.input_lengths.to(device=logits.device, dtype=torch.long)
    batch_idx = torch.arange(logits.shape[0], device=logits.device)
    final_logits = logits[batch_idx, input_lengths - 1]
    selected = torch.gather(final_logits, dim=-1, index=labels)
    return -(selected[:, 0] - selected[:, 1]).mean()


def _tensor_summary(inputs: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for key, tensor in inputs.items():
        summary[key] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "device": str(tensor.device),
        }
    return summary


def write_model_input_summary(prepared_batch, path: Path, extra: Optional[Dict[str, Any]] = None) -> None:
    payload: Dict[str, Any] = {
        "clean_inputs": _tensor_summary(prepared_batch.clean_inputs),
        "corrupt_inputs": _tensor_summary(prepared_batch.corrupt_inputs),
        "input_lengths": prepared_batch.input_lengths.tolist(),
    }
    if torch.is_tensor(prepared_batch.labels):
        payload["labels"] = prepared_batch.labels.tolist()
    else:
        payload["labels"] = prepared_batch.labels
    if extra:
        payload.update(extra)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def export_graph_artifacts(graph, output_dir: Path, *, topn: int) -> tuple[GraphArtifacts, GraphStats]:
    full_json = output_dir / "graph_full.json"
    topn_json = output_dir / "graph_topn.json"
    topn_png = output_dir / "graph_topn.png"

    graph.reset(empty=False)
    graph.to_json(str(full_json))

    n_edges_total = int(graph.real_edge_mask.sum().item())
    n_keep = max(1, min(int(topn), n_edges_total))
    graph.apply_topn(n_keep, absolute=True, reset=True, prune=True)
    graph.to_json(str(topn_json))

    topn_png_path = _export_topn_png_best_effort(graph, topn_png)

    artifacts = GraphArtifacts(
        full_json=str(full_json),
        topn_json=str(topn_json),
        topn_png=topn_png_path,
    )
    stats = GraphStats(
        n_forward=int(graph.n_forward),
        n_backward=int(graph.n_backward),
        n_edges_total=n_edges_total,
        n_edges_in_topn_graph=int(graph.count_included_edges()),
        score_shape_0=int(graph.scores.shape[0]),
        score_shape_1=int(graph.scores.shape[1]),
    )
    return artifacts, stats


def _resolve_root_mask(graph, *, root_policy: Literal["input_layer0"]) -> torch.Tensor:
    if root_policy != "input_layer0":
        raise ValueError(f"Unsupported root_policy: {root_policy}")

    root_mask = torch.zeros(graph.n_forward, dtype=torch.bool, device=graph.in_graph.device)
    # Always keep input as a valid source root.
    root_mask[0] = True
    # For multimodal runs, layer-0 nodes are where modality information enters residual stream.
    for name, node in graph.nodes.items():
        if name == "m0" or name.startswith("a0."):
            idx = graph.forward_index(node, attn_slice=False)
            if isinstance(idx, int):
                root_mask[idx] = True
    return root_mask


def _prune_with_root_mask(graph, *, root_mask: torch.Tensor, max_iter: int = 64) -> None:
    if getattr(graph, "neurons_in_graph", None) is not None:
        graph.nodes_in_graph &= graph.neurons_in_graph.any(dim=1)

    for _ in range(max_iter):
        nodes_with_outgoing = graph.in_graph.any(dim=1)
        backward_active = graph.in_graph.any(dim=0).float()
        nodes_with_ingoing = torch.matmul(graph.forward_to_backward.float(), backward_active) > 0
        nodes_with_ingoing |= root_mask

        old_nodes_in_graph = graph.nodes_in_graph.clone()
        graph.nodes_in_graph[:] = nodes_with_outgoing & nodes_with_ingoing

        forward_in_graph = graph.nodes_in_graph.float()
        backward_in_graph = torch.matmul(forward_in_graph, graph.forward_to_backward.float())
        backward_in_graph[-1] = 1.0  # logits destination always valid
        edge_remask = (forward_in_graph.view(-1, 1) > 0) & (backward_in_graph.view(1, -1) > 0)

        old_edges_in_graph = graph.in_graph.clone()
        graph.in_graph &= edge_remask

        if torch.equal(old_nodes_in_graph, graph.nodes_in_graph) and torch.equal(old_edges_in_graph, graph.in_graph):
            break

    if getattr(graph, "neurons_in_graph", None) is not None:
        graph.neurons_in_graph &= graph.nodes_in_graph.view(-1, 1)


def export_graph_artifacts_nonempty_topn(
    graph,
    output_dir: Path,
    *,
    topn: int,
    root_policy: Literal["input_layer0"] = "input_layer0",
) -> tuple[GraphArtifacts, GraphStats, Dict[str, Any]]:
    full_json = output_dir / "graph_full.json"
    topn_json = output_dir / "graph_topn.json"
    topn_png = output_dir / "graph_topn.png"

    graph.reset(empty=False)
    graph.to_json(str(full_json))

    n_edges_total = int(graph.real_edge_mask.sum().item())
    n_keep = max(1, min(int(topn), n_edges_total))

    # Keep exact topn edges first, then prune connectivity from modality-aware roots.
    graph.apply_topn(n_keep, absolute=True, reset=True, prune=False)
    root_mask = _resolve_root_mask(graph, root_policy=root_policy)
    _prune_with_root_mask(graph, root_mask=root_mask)
    graph.to_json(str(topn_json))

    topn_png_path = _export_topn_png_best_effort(graph, topn_png)

    artifacts = GraphArtifacts(
        full_json=str(full_json),
        topn_json=str(topn_json),
        topn_png=topn_png_path,
    )
    stats = GraphStats(
        n_forward=int(graph.n_forward),
        n_backward=int(graph.n_backward),
        n_edges_total=n_edges_total,
        n_edges_in_topn_graph=int(graph.count_included_edges()),
        score_shape_0=int(graph.scores.shape[0]),
        score_shape_1=int(graph.scores.shape[1]),
    )
    selection = {
        "selection_mode": "topn_root_prune",
        "root_policy": root_policy,
        "topn_requested": int(topn),
        "topn_effective": int(n_keep),
    }
    return artifacts, stats, selection


def write_run_summary(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _export_topn_png_best_effort(graph, topn_png: Path) -> str:
    try:
        graph.to_image(str(topn_png))
        return str(topn_png)
    except Exception as exc:  # noqa: BLE001
        text = str(exc).lower()
        # Keep JSON/stat outputs usable even when local visualization deps are missing.
        if (
            "pygraphviz" in text
            or "program dot not found" in text
            or "graphviz" in text
            or isinstance(exc, ModuleNotFoundError)
        ):
            return ""
        raise


def load_image_from_path_or_url(*, image_path: str, image_url: str):
    from PIL import Image

    if image_path:
        return Image.open(image_path).convert("RGB")
    with urlopen(image_url, timeout=60) as response:
        data = response.read()
    return Image.open(BytesIO(data)).convert("RGB")


def load_audio_from_path_or_url(*, audio_path: str, audio_url: str, sampling_rate: int = 16000):
    import librosa

    if audio_path:
        audio, sr = librosa.load(audio_path, sr=sampling_rate)
        return audio, sr

    with urlopen(audio_url, timeout=60) as response:
        data = response.read()
    audio, sr = librosa.load(BytesIO(data), sr=sampling_rate)
    return audio, sr


def dataclass_dict(obj: Any) -> Dict[str, Any]:
    return asdict(obj)
