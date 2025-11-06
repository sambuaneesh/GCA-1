# coding: utf-8
"""
Edge-conditioned graph encoder with attention pooling + classifier.

- The message depends on the source node feature, edge feature (entity embeddings or zeros),
  and edge type (TEMP=0, ENTITY=1) via a small MLP.
- Undirected entity links are already created as two directed edges.
"""

from typing import Optional

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeCondConv(nn.Module):
    """Single layer of edge-conditioned message passing."""

    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()

        # Define the message input dimensions (node features + edge features + edge type)
        msg_in_dim = in_dim + edge_dim + 2

        # MLP for message passing
        self.msg_mlp = nn.Sequential(
            nn.Linear(msg_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Linear transformation for the node's own feature
        self.self_lin = nn.Linear(in_dim, hidden_dim)

        # Normalization and dropout layers
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for edge-conditioned message passing.

        Args:
            g: DGL graph object.
            h: Node feature tensor of shape [N, in_dim].

        Returns:
            Tensor of shape [N, hidden_dim] after message passing and transformations.
        """
        # Make a local copy of the graph to avoid modifying the original
        g = g.local_var()
        g.ndata["h"] = h  # Assign node features to graph

        # Get edge type and features (if available)
        e_type = g.edata.get("e_type", None)
        if e_type is None or e_type.numel() == 0:
            # No edges, return the node features transformed
            return self.norm(self.self_lin(h))

        # One-hot encode the edge types (0 = TEMP, 1 = ENTITY)
        onehot = F.one_hot(e_type, num_classes=2).float()

        # Get edge features or use zeros if not available
        e_feat = g.edata.get("e_feat", None)
        if e_feat is None or e_feat.numel() == 0:
            e_feat = torch.zeros(e_type.shape[0], 0, device=h.device)

        # Concatenate edge features with one-hot encoded edge type
        e_concat = torch.cat([e_feat, onehot], dim=-1)
        g.edata["e_concat"] = e_concat

        # Message function for edge-conditioned passing
        def message_fn(edges):
            z = torch.cat(
                [edges.src["h"], edges.data["e_concat"]], dim=-1
            )  # Concatenate source node features and edge features
            m = self.msg_mlp(z)  # Apply MLP to the concatenated features
            return {"m": m}

        # Update the graph by applying the message function
        g.update_all(message_fn, dgl.function.sum("m", "m_sum"))
        m_sum = g.ndata.get("m_sum", torch.zeros_like(self.self_lin(h)))

        # Final node feature after message passing, normalization, and dropout
        out = self.self_lin(h) + m_sum
        out = self.dropout(F.relu(out))
        return self.norm(out)


class TemporalEntityEncoder(nn.Module):
    """Stack of EdgeCondConv layers with attention pooling and classifier."""

    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int = 256, num_layers: int = 2, num_classes: int = 6):
        super().__init__()

        assert num_layers >= 1, "Number of layers must be at least 1"

        # Stack of EdgeCondConv layers
        self.layers = nn.ModuleList(
            [EdgeCondConv(in_dim if i == 0 else hidden_dim, edge_dim, hidden_dim) for i in range(num_layers)]
        )

        # Attention pooling layer
        self.att = nn.Linear(hidden_dim, 1)

        # Simple classifier (MLP) for classification
        self.cls = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        """
        Forward pass for the TemporalEntityEncoder model.

        Args:
            g: DGL graph object with node and edge features.

        Returns:
            Logits tensor of shape [B, num_classes] for classification.
        """
        h = g.ndata["feat"]  # Get initial node features
        edge_dim = g.edata["e_feat"].shape[1] if g.num_edges() > 0 and "e_feat" in g.edata else 0

        # Apply edge-conditioned convolution layers
        for layer in self.layers:
            h = layer(g, h)

        # Attention pooling over node features
        with g.local_scope():
            g.ndata["h"] = h
            g.ndata["score"] = self.att(h)  # Calculate attention scores for nodes
            attention_weights = dgl.softmax_nodes(g, "score")  # Softmax over nodes

            # Apply attention weights to node features
            g.ndata["h_weighted"] = h * attention_weights
            readout = dgl.sum_nodes(g, "h_weighted")  # Aggregate weighted node features

            if readout.dim() == 1:  # Single-graph case, add batch dimension
                readout = readout.unsqueeze(0)

        # Apply the classifier to the pooled features
        logits = self.cls(readout)  # Class logits for each graph in the batch
        return logits
