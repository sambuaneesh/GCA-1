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
import re
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
    """
    Relation-aware GCN from the training script.
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
        if num_bases == -1 or num_bases > num_rels:
            num_bases = num_rels

        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_layers = num_layers
        self.num_bases = num_bases
        self.dropout = nn.Dropout(dropout)

        # Project input features if in_dim doesn't match h_dim
        self.in_proj = nn.Linear(in_dim, h_dim) if in_dim != h_dim else None

        layers = []
        for i in range(num_layers):
            in_c = h_dim
            out_c = out_dim if i == num_layers - 1 else h_dim
            act = nn.ReLU() if i < num_layers - 1 else None

            # In the training script, dropout is applied outside the layer.
            # The original scoring script had it inside. We match the training script.
            layer = RelGraphConv(
                in_feat=in_c,
                out_feat=out_c,
                num_rels=num_rels,
                regularizer="basis",
                num_bases=num_bases,
                self_loop=self_loop,
                activation=act,
                dropout=0.0,  # Dropout is applied externally in the forward pass
            )
            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, g: dgl.DGLGraph, feats: torch.Tensor) -> torch.Tensor:
        # Ensure norm exists. The scoring function now does this, but it's safe to have here too.
        if "norm" not in g.edata:
            g.edata["norm"] = dgl.norm_by_dst(g).unsqueeze(1)

        h = self.in_proj(feats) if self.in_proj is not None else feats

        for i, layer in enumerate(self.layers):
            # Pass the edge type and the norm tensor
            h = layer(g, h, g.edata[dgl.ETYPE], g.edata["norm"])
            if i < len(self.layers) - 1:
                h = self.dropout(h)
        return h


class GraphLinkPredict(nn.Module):
    """
    Link-prediction head from the training script.
    Uses `w_relation` parameter name.
    """

    def __init__(self, encoder: GCARGCN, reg_param: float = 0.01):
        super().__init__()
        self.encoder = encoder
        self.reg_param = reg_param
        # This parameter name MUST match the checkpoint
        self.w_relation = nn.Parameter(torch.Tensor(encoder.num_rels, encoder.out_dim))
        nn.init.xavier_uniform_(self.w_relation, gain=nn.init.calculate_gain("relu"))

    def forward(self, g, node_feats):
        return self.encoder(g, node_feats)

    def calc_score(self, embedding: torch.Tensor, triplets: torch.Tensor) -> torch.Tensor:
        """
        triplets: LongTensor [B, 3] with (head_idx, rel_id, tail_idx)
        returns logits (pre-sigmoid) [B]
        """
        s = embedding[triplets[:, 0]]
        r = self.w_relation[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        return torch.sum(s * r * o, dim=1)

    def get_loss(self, embedding: torch.Tensor, triplets: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        score = self.calc_score(embedding, triplets)
        predict_loss = nn.functional.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embedding)
        return predict_loss + self.reg_param * reg_loss

    def regularization_loss(self, embedding: torch.Tensor) -> torch.Tensor:
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))


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
        vec = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
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


def _infer_dims_from_state(state_dict):
    # Find layer weight keys like: encoder.layers.{i}.linear_r.W
    layer_keys = [k for k in state_dict.keys() if k.startswith("encoder.layers.") and k.endswith("linear_r.W")]
    if not layer_keys:
        raise ValueError("Could not find any 'encoder.layers.*.linear_r.W' in checkpoint state_dict.")

    # Extract layer indices and sort
    layer_ids = sorted(int(re.findall(r"encoder\.layers\.(\d+)\.", k)[0]) for k in layer_keys)
    first_key = f"encoder.layers.{layer_ids[0]}.linear_r.W"
    last_key = f"encoder.layers.{layer_ids[-1]}.linear_r.W"

    # Shapes are [num_rels, in_feat, out_feat]
    nr_first, in_first, out_first = state_dict[first_key].shape
    nr_last, in_last, out_last = state_dict[last_key].shape

    num_layers = layer_ids[-1] - layer_ids[0] + 1
    num_rels = nr_first  # should match across layers

    # Heuristic: for 2+ layers, h_dim is the out_feat of layer 0, out_dim is out_feat of last layer
    in_dim = in_first
    h_dim = out_first
    out_dim = out_last

    return in_dim, h_dim, out_dim, num_rels, num_layers


def load_pretrained_predictor(
    ckpt_path: str,
    device: torch.device,
) -> GraphLinkPredict:
    """
    Load GraphLinkPredict + GCARGCN from a checkpoint produced by train_gca_rgcn.py
    """
    # This warning is fine since we created the file
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "state_dict" not in ckpt:
        raise ValueError(f"Checkpoint at {ckpt_path} missing 'state_dict' key")

    # Reconstruct encoder using metadata from checkpoint
    in_dim = int(ckpt.get("in_dim", 384))  # Default to SBERT dim if not saved
    h_dim = int(ckpt["h_dim"])
    out_dim = int(ckpt["out_dim"])
    num_rels = int(ckpt["num_rels"])
    num_layers = int(ckpt["num_layers"])
    num_bases = int(ckpt["num_bases"])
    dropout = float(ckpt["dropout"])
    self_loop = bool(ckpt.get("self_loop", True))

    print(
        f"[RGCN] Using dims from checkpoint: in={in_dim}, h={h_dim}, out={out_dim}, "
        f"num_rels={num_rels}, num_layers={num_layers}, num_bases={num_bases}"
    )

    encoder = GCARGCN(
        in_dim=in_dim,
        h_dim=h_dim,
        out_dim=out_dim,
        num_rels=num_rels,
        num_layers=num_layers,
        num_bases=num_bases,
        dropout=dropout,
        self_loop=self_loop,
    ).to(device)

    model = GraphLinkPredict(encoder).to(device)

    # Load the state dict. It should now match perfectly.
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=True)
    if missing or unexpected:
        # This should not happen now, but it's good practice to keep the check
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
    force_feat_dim: Optional[int] = None,  # This argument is now less important
    device_str: str = "cuda",
) -> List[float]:
    """
    Convenience API: score a list of (h, r, t) triples given node strings.
    Returns a list of raw logits (higher => more plausible).
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # --- MODIFIED PART ---
    # First, peek into the checkpoint to get the correct in_dim
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    in_dim = int(ckpt.get("in_dim", 384))  # Get in_dim from checkpoint

    # Now load the full model onto the target device
    predictor = load_pretrained_predictor(ckpt_path, device)
    # --- END MODIFIED PART ---

    rel2id = load_relation_map(relation_map_path)

    # Prepare node features
    featurer = TextFeaturer(sbert_model, in_dim)  # Use the correct in_dim for the featurer
    feats = featurer.encode(nodes)

    # This check is now slightly redundant but harmless.
    # The projection layer inside GCARGCN will handle the mismatch.
    if feats.shape[1] != in_dim:
        print(
            f"[WARN] SBERT feats {feats.shape[1]} != required in_dim {in_dim}. "
            f"The model's projection layer will handle this."
        )

    # Build graph + score
    g, _, feat_t, tri_idx = build_dgl_from_triples(nodes, triples, rel2id, feats)
    g = g.to(device)
    feat_t = feat_t.to(device)
    tri_idx = tri_idx.to(device)

    # Ensure norm is present before passing to the model
    if "norm" not in g.edata:
        g.edata["norm"] = dgl.norm_by_dst(g).unsqueeze(1)

    node_emb = predictor(g, feat_t)
    logits = predictor.calc_score(node_emb, tri_idx)
    return logits.detach().cpu().tolist()
