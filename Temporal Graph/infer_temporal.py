# coding: utf-8
"""
Inference script for temporal-entity graph classifier.
Loads a trained checkpoint and evaluates on a selected validation sample.

Usage:
  python infer_temporal.py \
      --graphs processed/dgl_temporal \
      --checkpoint processed/temporal_entity_ckpt.pt \
      --seed 42 \
      --val-num 0
"""

import argparse
import json
from pathlib import Path
from typing import List

import dgl
import torch
from model_temporal_gnn import TemporalEntityEncoder


def _load_graphs(folder: str) -> List[dgl.DGLGraph]:
    folder = Path(folder)
    graphs = []
    for p in sorted(folder.glob("*.dgl")):
        gs, _ = dgl.load_graphs(str(p))
        graphs.extend(gs)
    return graphs


def main():
    ap = argparse.ArgumentParser(description="Inference for temporal-entity graph classifier")
    ap.add_argument(
        "--graphs", default="Temporal Graph/processed/dgl_temporal", help="Path to folder containing DGL graphs"
    )
    ap.add_argument(
        "--checkpoint", default="Temporal Graph/processed/temporal_entity_ckpt.pt", help="Path to trained checkpoint"
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed for validation split consistency")
    ap.add_argument(
        "--json-data",
        default="Temporal Graph/processed/diahalu_temporal.json",
        help="Path to the original JSON data file",
    )
    ap.add_argument("--val-num", type=int, default=0, help="Index of the validation sample to use")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading graphs from {args.graphs}...")
    graphs = _load_graphs(args.graphs)
    if not graphs:
        raise RuntimeError("No graphs found. Check --graphs path.")

    labels = [int(g.ndata["y"][0].item()) for g in graphs]

    # Filter out empty graphs (same as training)
    keep = [i for i, g in enumerate(graphs) if g.num_nodes() > 0]
    graphs = [graphs[i] for i in keep]
    labels = [labels[i] for i in keep]

    print(f"Loaded {len(graphs)} valid graphs")

    # Create validation split with same seed logic as training
    n = len(graphs)
    ggen = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(n, generator=ggen)
    n_train = int(n * 0.8)
    val_idx = idx[n_train:].tolist()

    print(f"Validation set size: {len(val_idx)} graphs")

    # Get selected validation sample
    selected_val_idx = val_idx[args.val_num]
    val_graph = graphs[selected_val_idx]
    val_label = labels[selected_val_idx]

    print(f"Using selected validation sample (index {selected_val_idx})")
    print(f"  Label: {val_label}")
    print(f"  Nodes: {val_graph.num_nodes()}")
    print(f"  Edges: {val_graph.num_edges()}")

    # Load the original JSON data to get the sample details
    print(f"\nLoading original data from {args.json_data}...")
    with open(args.json_data, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    # The graph index corresponds to the JSON data index (after filtering empty graphs)
    # Since we filtered graphs, we need to use the original index before filtering
    original_idx = keep[selected_val_idx]
    sample_data = json_data[original_idx]

    print("\n" + "=" * 80)
    print(f"ORIGINAL SAMPLE DATA (index {original_idx})")
    print("=" * 80)
    print(json.dumps(sample_data, indent=2, ensure_ascii=False))
    print("=" * 80 + "\n")

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    if not Path(args.checkpoint).exists():
        raise RuntimeError(f"Checkpoint {args.checkpoint} not found.")

    ckpt = torch.load(args.checkpoint, map_location=device)

    # Initialize model with saved configuration
    model = TemporalEntityEncoder(
        in_dim=ckpt["in_dim"],
        edge_dim=ckpt["edge_dim"],
        hidden_dim=ckpt["hidden"],
        num_layers=ckpt["layers"],
        num_classes=ckpt["num_classes"],
    ).to(device)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print(f"Model loaded successfully:")
    print(f"  in_dim={ckpt['in_dim']}, edge_dim={ckpt['edge_dim']}")
    print(f"  hidden_dim={ckpt['hidden']}, num_layers={ckpt['layers']}")
    print(f"  num_classes={ckpt['num_classes']}")

    # Run inference on selected validation sample
    print("\n" + "=" * 80)
    print("RUNNING INFERENCE ON SELECTED VALIDATION SAMPLE")
    print("=" * 80 + "\n")

    val_graph = val_graph.to(device)

    with torch.no_grad():
        logits, attention_weights = model(val_graph, output_attention=True)
        pred = logits.argmax(dim=-1).item()
        probs = torch.softmax(logits, dim=-1)[0]

    class_names = ["Factual", "Reasoning Error", "Incoherence", "Irrelevance", "Overreliance", "Non-Factual"]

    print(f"Ground Truth Label: {val_label} ({class_names[val_label]})")
    print(f"Predicted Label:    {pred} ({class_names[pred]})")
    print(f"Correct: {'YES' if pred == val_label else 'NO'}")
    print()

    print("Class Probabilities:")
    for i, prob in enumerate(probs):
        print(f"  {i} ({class_names[i]:<20}): {prob:.4f}")

    print("\nAttention Weights:")
    dialogues = sample_data.get("dialogues", [])
    for i, weight in enumerate(attention_weights):
        print(f"  {i}: {weight.item():.4f} | {dialogues[i]}")


if __name__ == "__main__":
    main()
