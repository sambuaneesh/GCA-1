# coding: utf-8
"""
Edge-conditioned graph encoder with attention pooling + classifier.

- Message depends on source node feature, edge feature (entity emb or zeros),
  and edge type (TEMP=0, ENTITY=1) via a tiny MLP.
- Undirected entity links are already created as two directed edges.
"""

from typing import Optional

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeCondConv(nn.Module):
    """One layer of edge-conditioned message passing."""

    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        # edge type is one-hot of size 2; we concat into edge feature inside forward
        msg_in = in_dim + edge_dim + 2
        self.msg_mlp = nn.Sequential(
            nn.Linear(msg_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.self_lin = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor) -> torch.Tensor:
        g = g.local_var()
        g.ndata["h"] = h

        # build one-hot edge type (2 types: 0 temporal, 1 entity)
        e_type = g.edata.get("e_type", None)
        if e_type is None or e_type.numel() == 0:
            # no edges
            return self.norm(self.self_lin(h))

        onehot = F.one_hot(e_type, num_classes=2).float()
        e_feat = g.edata.get("e_feat", None)
        if e_feat is None or e_feat.numel() == 0:
            e_feat = torch.zeros(e_type.shape[0], 0, device=h.device)
        e_concat = torch.cat([e_feat, onehot], dim=-1)
        g.edata["e_concat"] = e_concat

        def _msg(edges):
            # [src.h | e_concat] -> message
            z = torch.cat([edges.src["h"], edges.data["e_concat"]], dim=-1)
            m = self.msg_mlp(z)
            return {"m": m}

        g.update_all(_msg, dgl.function.sum("m", "m_sum"))
        m_sum = g.ndata.get("m_sum", torch.zeros_like(self.self_lin(h)))
        out = self.self_lin(h) + m_sum
        out = self.dropout(F.relu(out))
        return self.norm(out)


class TemporalEntityEncoder(nn.Module):
    """Stack of EdgeCondConv + attention pooling + classifier."""

    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int = 256, num_layers: int = 2, num_classes: int = 6):
        super().__init__()
        assert num_layers >= 1
        self.layers = nn.ModuleList(
            [EdgeCondConv(in_dim if i == 0 else hidden_dim, edge_dim, hidden_dim) for i in range(num_layers)]
        )

        # attention pooling
        self.att = nn.Linear(hidden_dim, 1)

        # small classifier
        self.cls = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        h = g.ndata["feat"]
        edge_dim = g.edata["e_feat"].shape[1] if g.num_edges() > 0 and "e_feat" in g.edata else 0

        for layer in self.layers:
            h = layer(g, h)

        # attention pooling over nodes (works for single or batched graphs)
        with g.local_scope():
            g.ndata["h"] = h
            g.ndata["score"] = self.att(h)  # [N,1]
            a = dgl.softmax_nodes(g, "score")  # segment-wise softmax per graph
            g.ndata["h_weighted"] = h * a
            readout = dgl.sum_nodes(g, "h_weighted")  # [B, H] or [H]
            if readout.dim() == 1:  # single-graph case -> [1,H]
                readout = readout.unsqueeze(0)

        logits = self.cls(readout)  # [B, C]
        return logits
