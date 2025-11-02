import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv


class GCARGCN(nn.Module):
    """
    Relation-aware GCN that consumes external node features (e.g., SBERT)
    and performs message passing over typed edges.

    Args:
        in_dim (int): input feature size
        h_dim (int): hidden dimension
        out_dim (int): output dimension (set = h_dim if you just need embeddings)
        num_rels (int): total number of relation types (global mapping)
        num_layers (int): total RGCN layers (>= 2)
        num_bases (int): number of basis matrices for regularization; -1 => use all
        dropout (float): dropout after each layer
        self_loop (bool): include self-loop transform
    """

    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        num_rels: int,
        num_layers: int = 2,
        num_bases: int = -1,
        dropout: float = 0.0,
        self_loop: bool = True,
    ):
        super().__init__()
        assert num_layers >= 2, "num_layers must be >= 2"
        if num_bases == -1:
            num_bases = num_rels

        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_layers = num_layers
        self.num_bases = num_bases
        self.dropout = nn.Dropout(dropout)

        # Project arbitrary input features to hidden size if needed
        self.in_proj = nn.Linear(in_dim, h_dim) if in_dim != h_dim else None

        layers = []

        # All RGCN layers operate on hidden size except the last which goes to out_dim
        # First + hidden layers
        for _ in range(num_layers - 1):
            layers.append(
                RelGraphConv(
                    h_dim,
                    h_dim,
                    num_rels,
                    regularizer="basis",
                    num_bases=num_bases,
                    self_loop=self_loop,
                )
            )
        # Output layer
        layers.append(
            RelGraphConv(
                h_dim,
                out_dim,
                num_rels,
                regularizer="basis",
                num_bases=num_bases,
                self_loop=self_loop,
            )
        )
        self.layers = nn.ModuleList(layers)

    @staticmethod
    def ensure_norm(g: dgl.DGLGraph):
        if "norm" not in g.edata:
            g.edata["norm"] = dgl.norm_by_dst(g).unsqueeze(1)

    def forward(self, g: dgl.DGLGraph, node_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: *homogeneous* DGLGraph with typed edges (g.edata[dgl.ETYPE] present)
            node_feats: tensor of shape [N, in_dim]
        Returns:
            Tensor of shape [N, out_dim]
        """
        self.ensure_norm(g)
        h = node_feats if self.in_proj is None else self.in_proj(node_feats)

        # First .. penultimate layer: ReLU + dropout
        for layer in self.layers[:-1]:
            h = layer(g, h, g.edata[dgl.ETYPE], g.edata["norm"])
            h = self.dropout(F.relu(h))

        # Final layer: no activation by default
        h = self.layers[-1](g, h, g.edata[dgl.ETYPE], g.edata["norm"])
        return h


class GraphLinkPredict(nn.Module):
    """
    Optional link-prediction head (DistMult-style) on top of GCARGCN encoder.
    Useful for pretraining the encoder when task labels are unavailable.
    """

    def __init__(self, encoder: GCARGCN, reg_param: float = 0.01):
        super().__init__()
        self.encoder = encoder
        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(encoder.num_rels, encoder.out_dim))
        nn.init.xavier_uniform_(self.w_relation, gain=nn.init.calculate_gain("relu"))

    def forward(self, g, node_feats):
        return self.encoder(g, node_feats)

    def calc_score(self, embedding: torch.Tensor, triplets: torch.Tensor) -> torch.Tensor:
        # triplets: [*, 3] (s, r, o)
        s = embedding[triplets[:, 0]]
        r = self.w_relation[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        return torch.sum(s * r * o, dim=1)

    def regularization_loss(self, embedding: torch.Tensor) -> torch.Tensor:
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, embedding: torch.Tensor, triplets: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        score = self.calc_score(embedding, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embedding)
        return predict_loss + self.reg_param * reg_loss
