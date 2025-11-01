# coding: utf-8
"""
Thin integration layer to use a pretrained DGL-based GCA RGCN checkpoint
for scoring triples inside this repository.

- Loads the encoder+link-predict head saved by train_gca_rgcn.py
- Builds a DGLGraph from (nodes, triples)
- Produces DistMult-style scores for provided triples

Requirements:
    pip install dgl torch sentence-transformers

Author: integration shim for GCA RGCN
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import dgl
import torch
import torch.nn as nn
from dgl.nn import RelGraphConv

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


# ----------------------------
# Model (matches the trainer)
# ----------------------------
class GCARGCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        num_rels: int,
        num_layers: int = 2,
        num_bases: int = -1,
        dropout: float = 0.1,
        self_loop: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_layers = num_layers
        self.num_bases = num_bases
        self.self_loop = self_loop

        h_dims = [in_dim] + [h_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for i in range(num_layers):
            in_c = h_dims[i]
            out_c = h_dims[i + 1]
            act = nn.ReLU() if i < num_layers - 1 else None
            layer = RelGraphConv(
                in_feat=in_c,
                out_feat=out_c,
                num_rels=num_rels,
                regularizer="basis" if num_bases and num_bases > 0 else "none",
                num_bases=num_bases if num_bases and num_bases > 0 else None,
                self_loop=self_loop,
                activation=act,
                dropout=dropout,
            )
            self.layers.append(layer)

    def forward(self, g: dgl.DGLGraph, feats: torch.Tensor) -> torch.Tensor:
        h = feats
        etype = g.edata[dgl.ETYPE]
        for i, layer in enumerate(self.layers):
            h = layer(g, h, etype)
            if i < len(self.layers) - 1:
                h = self.dropout(h)
        return h


class GraphLinkPredict(nn.Module):
    """
    DistMult-style link predictor over the encoder's output.
    """

    def __init__(self, encoder: GCARGCN):
        super().__init__()
        self.encoder = encoder
        self.rel = nn.Parameter(torch.empty(encoder.num_rels, encoder.out_dim))
        nn.init.xavier_uniform_(self.rel)

    def forward(self, g: dgl.DGLGraph, feats: torch.Tensor) -> torch.Tensor:
        return self.encoder(g, feats)

    def calc_score(
        self, node_emb: torch.Tensor, triplets: torch.Tensor
    ) -> torch.Tensor:
        """
        triplets: LongTensor [B, 3] with (head_idx, rel_id, tail_idx)
        returns logits (pre-sigmoid) [B]
        """
        s = node_emb[triplets[:, 0]]
        r = self.rel[triplets[:, 1]]
        o = node_emb[triplets[:, 2]]
        return torch.sum(s * r * o, dim=1)

    def get_loss(
        self, node_emb: torch.Tensor, samples: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        logits = self.calc_score(node_emb, samples)
        return nn.functional.binary_cross_entropy_with_logits(logits, labels.float())


# --------------------------------------
# Utilities: relation map + text featurer
# --------------------------------------
def load_relation_map(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    if "__UNK__" not in m:
        # reserve 0 for UNK if absent; shift others up by 1 for safety
        if 0 in set(m.values()):
            # leave as-is; just add UNK to a free id
            unk_id = max(m.values()) + 1
        else:
            # shift and put UNK at 0
            m = {k: v + 1 for k, v in m.items()}
            unk_id = 0
        m["__UNK__"] = unk_id
    return m


class TextFeaturer:
    """
    SBERT featurer. If sentence-transformers is unavailable or a model name is None,
    fall back to random features (deterministic if seed set by caller).
    """

    def __init__(self, model_name: Optional[str], feat_dim: int):
        self.model_name = model_name if (model_name and SentenceTransformer) else None
        self.feat_dim = feat_dim
        self.embedder = SentenceTransformer(model_name) if self.model_name else None

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        if self.embedder is None:
            # random features (caller may set torch.manual_seed for reproducibility)
            return torch.randn(len(texts), self.feat_dim)
        vec = self.embedder.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return torch.tensor(vec, dtype=torch.float32)


# --------------------------------------
# Graph building + scoring API
# --------------------------------------
def _index_nodes(nodes: List[str]) -> Dict[str, int]:
    return {name: i for i, name in enumerate(nodes)}


def build_dgl_from_triples(
    nodes: List[str],
    triples: List[Tuple[str, str, str]],
    relation2id: Dict[str, int],
    node_feats: torch.Tensor,
) -> Tuple[dgl.DGLGraph, Dict[str, int], torch.Tensor, torch.Tensor]:
    """
    Build a DGLGraph and the (head,rel,tail) LongTensor index triplets from strings.

    Returns:
        g, node2idx, feat (N, D), triplet_idx (E, 3)
    """
    node2idx = _index_nodes(nodes)
    src_idx, dst_idx, rel_id = [], [], []
    tri_idx = []

    unk = relation2id.get("__UNK__", 0)
    for h, r, t in triples:
        hi = node2idx[h]
        ti = node2idx[t]
        ri = relation2id.get(r, unk)
        src_idx.append(hi)
        dst_idx.append(ti)
        rel_id.append(ri)
        tri_idx.append((hi, ri, ti))

    g = dgl.graph((torch.tensor(src_idx), torch.tensor(dst_idx)), num_nodes=len(nodes))
    g.ndata["feat"] = node_feats
    g.edata[dgl.ETYPE] = torch.tensor(rel_id, dtype=torch.long)

    triplet_idx = torch.tensor(tri_idx, dtype=torch.long)
    return g, node2idx, node_feats, triplet_idx


def load_pretrained_predictor(
    ckpt_path: str,
    device: torch.device,
) -> GraphLinkPredict:
    """
    Load GraphLinkPredict + GCARGCN from a checkpoint produced by train_gca_rgcn.py
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    if "state_dict" not in ckpt:
        raise ValueError(f"Checkpoint at {ckpt_path} missing 'state_dict' key")

    enc = GCARGCN(
        in_dim=int(ckpt["in_dim"]),
        h_dim=int(ckpt["h_dim"]),
        out_dim=int(ckpt["out_dim"]),
        num_rels=int(ckpt["num_rels"]),
        num_layers=int(ckpt["num_layers"]),
        num_bases=int(ckpt["num_bases"]),
        dropout=float(ckpt["dropout"]),
        self_loop=bool(ckpt.get("self_loop", True)),
    ).to(device)

    model = GraphLinkPredict(enc).to(device)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state_dict: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model


@torch.no_grad()
def score_triples_with_checkpoint(
    *,
    ckpt_path: str,
    relation_map_path: str,
    nodes: List[str],
    triples: List[Tuple[str, str, str]],
    sbert_model: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2",
    force_feat_dim: Optional[int] = None,
    device_str: str = "cuda",
) -> List[float]:
    """
    Convenience API: score a list of (h, r, t) triples given node strings.

    Returns a list of raw logits (higher => more plausible).
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    predictor = load_pretrained_predictor(ckpt_path, device)
    rel2id = load_relation_map(relation_map_path)

    # Prepare node features (must match checkpoint in_dim)
    in_dim = int(force_feat_dim or predictor.encoder.in_dim)
    featurer = TextFeaturer(sbert_model, in_dim)
    feats = featurer.encode(nodes)

    # If SBERT dim doesn't match in_dim, fix by linear proj
    if feats.shape[1] != in_dim:
        print(
            f"[WARN] SBERT feats {feats.shape[1]} != required in_dim {in_dim}. "
            f"Projecting to in_dim."
        )
        proj = nn.Linear(feats.shape[1], in_dim, bias=False)
        with torch.no_grad():
            feats = proj(feats)

    # Build graph + score
    g, _, feat_t, tri_idx = build_dgl_from_triples(nodes, triples, rel2id, feats)
    g = g.to(device)
    feat_t = feat_t.to(device)
    tri_idx = tri_idx.to(device)

    node_emb = predictor(g, feat_t)
    logits = predictor.calc_score(node_emb, tri_idx)
    return logits.detach().cpu().tolist()
