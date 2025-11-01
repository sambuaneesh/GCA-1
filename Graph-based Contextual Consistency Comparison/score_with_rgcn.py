# coding: utf-8
"""
CLI to score triples in your datasets using a pretrained GCA RGCN checkpoint.

Two input styles supported:

1) Simple JSON (list of records):
    [
      {
        "nodes": ["Albert Einstein","Ulm","Germany"],
        "triples": [["Albert Einstein","born_in","Ulm"],
                    ["Ulm","located_in","Germany"]]
      },
      ...
    ]

2) The repo's extracted format (list of entries) where each entry has:
      entry["sample0"]["nodes"]   -> list[str]
      entry["sample0"]["graph"]   -> list[[h, r, t] or tuples]
   We will append:
      entry["triples_score"]      -> list[float] aligned with sample0.graph

Usage:
    python score_with_rgcn.py \
        --ckpt checkpoints/gca_rgcn.ckpt \
        --relation-map data/gca_graphs/relation2id.json \
        --input path/to/input.json \
        --output path/to/output.json \
        --device cuda
"""

import argparse
import json
from typing import Any, Dict, List, Tuple

from rgcn_gca_encoder import score_triples_with_checkpoint


def _is_repo_extracted_format(obj: Any) -> bool:
    # heuristic: list of dicts, each has 'sample0' with nodes+graph
    if not isinstance(obj, list):
        return False
    if not obj:
        return False
    sample = obj[0]
    return (
        isinstance(sample, dict)
        and "sample0" in sample
        and isinstance(sample["sample0"], dict)
        and "nodes" in sample["sample0"]
        and "graph" in sample["sample0"]
    )


def _as_triples_list(graph_any) -> List[Tuple[str, str, str]]:
    triples = []
    for tri in graph_any:
        if isinstance(tri, (list, tuple)) and len(tri) == 3:
            h, r, t = tri
        else:
            # graph entries might be serialized as "h, r, t" strings
            s = str(tri)
            parts = [p.strip() for p in s.split(",")]
            h, r = parts[0], parts[1]
            t = ",".join(parts[2:]).strip()
        triples.append((h, r, t))
    return triples


def score_repo_extracted(
    data: List[Dict[str, Any]], ckpt: str, relmap: str, device: str, sbert_model: str
) -> List[Dict[str, Any]]:
    out = []
    for entry in data:
        nodes = entry["sample0"]["nodes"]
        graph = _as_triples_list(entry["sample0"]["graph"])
        scores = score_triples_with_checkpoint(
            ckpt_path=ckpt,
            relation_map_path=relmap,
            nodes=nodes,
            triples=graph,
            sbert_model=sbert_model,
            device_str=device,
        )
        entry["triples_score"] = scores  # used by downstream metrics scripts
        out.append(entry)
    return out


def score_simple(
    data: List[Dict[str, Any]], ckpt: str, relmap: str, device: str, sbert_model: str
) -> List[Dict[str, Any]]:
    out = []
    for record in data:
        nodes = record["nodes"]
        triples = [(h, r, t) for h, r, t in record["triples"]]
        scores = score_triples_with_checkpoint(
            ckpt_path=ckpt,
            relation_map_path=relmap,
            nodes=nodes,
            triples=triples,
            sbert_model=sbert_model,
            device_str=device,
        )
        out.append({**record, "triples_score": scores})
    return out


def main():
    ap = argparse.ArgumentParser("Score triples with pretrained GCA RGCN")
    ap.add_argument(
        "--ckpt", required=True, help="Path to gca_rgcn.ckpt produced by training"
    )
    ap.add_argument(
        "--relation-map",
        required=True,
        help="Path to relation2id.json used at train time",
    )
    ap.add_argument(
        "--input", required=True, help="Input JSON file (see header for formats)"
    )
    ap.add_argument("--output", required=True, help="Where to write augmented JSON")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--sbert-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SBERT model for node features; set to '' to use random features",
    )
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if _is_repo_extracted_format(data):
        scored = score_repo_extracted(
            data, args.ckpt, args.relation_map, args.device, args.sbert_model
        )
    else:
        scored = score_simple(
            data, args.ckpt, args.relation_map, args.device, args.sbert_model
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(scored, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
