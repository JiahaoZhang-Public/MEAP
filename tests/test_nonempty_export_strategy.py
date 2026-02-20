import json

from examples.common import export_graph_artifacts, export_graph_artifacts_nonempty_topn
import torch


class _FakeNode:
    def __init__(self, name: str):
        self.name = name


class _FakeGraph:
    def __init__(self):
        self.n_forward = 4
        self.n_backward = 3  # last index is treated as logits destination
        self.nodes = {
            "input": _FakeNode("input"),
            "a0.h0": _FakeNode("a0.h0"),
            "m0": _FakeNode("m0"),
            "m1": _FakeNode("m1"),
        }
        self._forward_index = {"input": 0, "a0.h0": 1, "m0": 2, "m1": 3}
        self.forward_to_backward = torch.tensor(
            [
                [1, 0, 0],  # input
                [0, 1, 0],  # a0.h0
                [0, 1, 0],  # m0
                [0, 1, 0],  # m1
            ],
            dtype=torch.bool,
        )
        self.scores = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.2, 9.0],  # a0.h0 -> logits (high)
                [0.1, 0.2, 8.0],  # m0 -> logits (high)
                [0.1, 0.2, 1.0],
            ],
            dtype=torch.float32,
        )
        self.real_edge_mask = torch.ones((self.n_forward, self.n_backward), dtype=torch.bool)
        self.in_graph = torch.zeros((self.n_forward, self.n_backward), dtype=torch.bool)
        self.nodes_in_graph = torch.zeros(self.n_forward, dtype=torch.bool)
        self.neurons_in_graph = None

    def forward_index(self, node, attn_slice=False):
        del attn_slice
        return self._forward_index[node.name]

    def reset(self, empty=True):
        if empty:
            self.in_graph[:] = False
            self.nodes_in_graph[:] = False
        else:
            self.in_graph[:] = self.real_edge_mask
            self.nodes_in_graph[:] = True

    def _standard_prune(self):
        for _ in range(16):
            nodes_with_outgoing = self.in_graph.any(dim=1)
            nodes_with_ingoing = torch.matmul(
                self.forward_to_backward.float(),
                self.in_graph.any(dim=0).float(),
            ) > 0
            nodes_with_ingoing[0] = True

            old_nodes = self.nodes_in_graph.clone()
            self.nodes_in_graph[:] = nodes_with_outgoing & nodes_with_ingoing

            forward_in_graph = self.nodes_in_graph.float()
            backward_in_graph = torch.matmul(forward_in_graph, self.forward_to_backward.float())
            backward_in_graph[-1] = 1.0
            edge_remask = (forward_in_graph.view(-1, 1) > 0) & (backward_in_graph.view(1, -1) > 0)

            old_edges = self.in_graph.clone()
            self.in_graph &= edge_remask
            if torch.equal(old_nodes, self.nodes_in_graph) and torch.equal(old_edges, self.in_graph):
                break

    def apply_topn(self, n, absolute=True, level="edge", reset=True, prune=True):
        del level
        if reset:
            self.reset(empty=True)
        scores = self.scores.abs() if absolute else self.scores
        order = torch.argsort(scores.view(-1), descending=True)
        self.in_graph.view(-1)[order[:n]] = True
        self.in_graph.view(-1)[order[n:]] = False
        if prune:
            self._standard_prune()

    def count_included_edges(self):
        return int(self.in_graph.sum().item())

    def to_json(self, filename: str):
        payload = {
            "nodes": {name: {"in_graph": bool(self.nodes_in_graph[idx])} for name, idx in self._forward_index.items()},
            "edges_in_graph": int(self.count_included_edges()),
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def to_image(self, filename: str):
        with open(filename, "wb") as f:
            f.write(b"fake")


def test_nonempty_export_keeps_edges_with_root_policy(tmp_path):
    g = _FakeGraph()
    artifacts, stats, selection = export_graph_artifacts_nonempty_topn(g, tmp_path, topn=2)

    assert stats.n_edges_total == g.n_forward * g.n_backward
    assert stats.n_edges_in_topn_graph > 0
    assert selection == {
        "selection_mode": "topn_root_prune",
        "root_policy": "input_layer0",
        "topn_requested": 2,
        "topn_effective": 2,
    }
    assert (tmp_path / "graph_full.json").exists()
    assert (tmp_path / "graph_topn.json").exists()
    assert (tmp_path / "graph_topn.png").exists()
    assert artifacts.topn_png.endswith("graph_topn.png")


def test_standard_export_can_be_empty_with_same_graph(tmp_path):
    g = _FakeGraph()
    _, stats = export_graph_artifacts(g, tmp_path, topn=2)
    assert stats.n_edges_in_topn_graph == 0
