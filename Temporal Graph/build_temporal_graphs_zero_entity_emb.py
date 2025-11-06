# coding: utf-8
"""
Build per-dialogue DGL graphs from diahalu_temporal.json.

- Node: each sentence in "dialogues" (encoded by a sentence encoder)
- Temporal edges: directed (i -> i+1) + back edge (i+1 -> i), edge_type = 0, edge embedding = zeros
- Entity edges: undirected (i <-> j) for every shared entity; edge_type = 1, edge embedding = zeros  <-- CHANGED
- Saves graphs to Temporal Graph/processed/dgl_temporal/*.dgl
- Also writes a label map for the 6-way classifier.

Usage:
  python build_temporal_graphs.py \
      --input "Temporal Graph/processed/diahalu_temporal.json" \
      --outdir "Temporal Graph/processed/dgl_temporal" \
      --encoder sentence-transformers/all-MiniLM-L6-v2
"""

import argparse
import json
import os
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import dgl
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def _encode_batch(embedder, texts: List[str], device: str = "cpu") -> torch.Tensor:
    with torch.no_grad():
        v = embedder.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=device if torch.cuda.is_available() else "cpu",
        )
    return v.cpu().float()


def _pairs(indices: List[int]) -> List[Tuple[int, int]]:
    # all unordered pairs (i, j), i < j
    out = []
    n = len(indices)
    for a in range(n):
        for b in range(a + 1, n):
            out.append((indices[a], indices[b]))
    return out


def _canon_entity(e: str) -> str:
    """Lowercase, trim punctuation/extra spaces; returns '' if unusable."""
    if not isinstance(e, str):
        return ""
    e = e.strip().strip(string.punctuation)
    e = re.sub(r"\s+", " ", e).lower()
    return e


def build_graph_for_item(
    dialogues: List[str],
    entities_per_turn: List[List[str]],
    sent_embedder,
    ent_embedder,  # kept for signature compatibility; unused now that entity edges are zeroed
    feat_dim: int,
) -> dgl.DGLGraph:
    N = len(dialogues)

    node_feats = _encode_batch(sent_embedder, dialogues)
    assert node_feats.shape[0] == N

    src, dst = [], []
    e_types = []  # 0 = TEMP, 1 = ENTITY
    e_feats = []

    zero_edge = torch.zeros(feat_dim)

    # TEMPORAL: undirected edges between i and i+1
    for i in range(N - 1):
        src.append(i)
        dst.append(i + 1)
        e_types.append(0)
        e_feats.append(zero_edge)
        src.append(i + 1)
        dst.append(i)
        e_types.append(0)
        e_feats.append(zero_edge)

    # ENTITY: undirected edges for every shared entity (now ZERO edge features)
    ent2turns: Dict[str, List[int]] = defaultdict(list)
    for i, ents in enumerate(entities_per_turn):
        turn_ents = set()
        for e in ents:
            ce = _canon_entity(e)
            if ce:
                turn_ents.add(ce)
        for ce in turn_ents:
            ent2turns[ce].append(i)

    # No entity encoding step; we force zero vectors for entity edges as well
    for _, turns in ent2turns.items():
        if len(turns) < 2:
            continue
        for i, j in _pairs(turns):
            v = zero_edge
            src.extend([i, j])
            dst.extend([j, i])
            e_types.extend([1, 1])
            e_feats.extend([v, v])

    # Build graph
    if not src:
        g = dgl.graph(([], []), num_nodes=N)
    else:
        g = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=N)

    g.ndata["feat"] = node_feats
    g.edata["e_type"] = torch.tensor(e_types, dtype=torch.long) if len(e_types) else torch.empty(0, dtype=torch.long)
    g.edata["e_feat"] = torch.stack(e_feats) if len(e_feats) else torch.empty(0, feat_dim)

    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="Temporal Graph/processed/diahalu_temporal.json")
    ap.add_argument("--outdir", default="Temporal Graph/processed/dgl_temporal")
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(args.encoder, device=device)
    feat_dim = embedder.get_sentence_embedding_dimension()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Fixed label mapping
    FIXED_LABEL_MAP = {
        "factual": 0,
        "Reasoning Error": 1,
        "Incoherence": 2,
        "Irrelevance": 3,
        "Overreliance": 4,
        "non-factual": 5,  # catch-all
    }
    with open(outdir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(FIXED_LABEL_MAP, f, indent=2, ensure_ascii=False)

    # Build & save one graph per item
    meta_records = []
    for idx, item in enumerate(tqdm(data, desc="Building graphs")):
        dialogues: List[str] = item["dialogues"]
        ents: List[List[str]] = item["entities"]
        assert len(dialogues) == len(ents), f"dialogues/entities length mismatch at idx {idx}"

        g = build_graph_for_item(dialogues, ents, embedder, embedder, feat_dim)

        if (item.get("label") or "").strip().lower() == "factual":
            y = FIXED_LABEL_MAP["factual"]
        else:
            raw_t = item.get("type")
            if isinstance(raw_t, list) or raw_t is None or str(raw_t).strip() == "":
                tnorm = "non-factual"
            else:
                tnorm = str(raw_t).strip()
            y = FIXED_LABEL_MAP.get(tnorm, FIXED_LABEL_MAP["non-factual"])

        g.ndata["y"] = torch.full((g.num_nodes(),), y, dtype=torch.long)

        save_path = outdir / f"graph_{idx:05d}.dgl"
        dgl.save_graphs(str(save_path), [g])

        meta_records.append(
            {
                "idx": idx,
                "y": y,
                "label": item["label"],
                "type": item.get("type"),
                "domain": item.get("domain"),
                "source": item.get("source"),
                "LLM": item.get("LLM"),
                "num_nodes": g.num_nodes(),
                "num_edges": g.num_edges(),
            }
        )

    with open(outdir / "index.json", "w", encoding="utf-8") as f:
        json.dump(meta_records, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(meta_records)} graphs to {outdir}")


if __name__ == "__main__":
    main()
