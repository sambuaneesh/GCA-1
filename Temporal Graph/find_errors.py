# coding: utf-8
"""
Find misclassified samples in temporal-entity graph classifier.
Loads a trained checkpoint and evaluates on the validation set, printing indices of errors.

Usage:
  python find_errors.py \
      --graphs processed/dgl_temporal \
      --checkpoint processed/temporal_entity_ckpt.pt \
      --seed 42
"""

import argparse
from pathlib import Path
from typing import List

import dgl
import torch
from dgl.dataloading import GraphDataLoader
from model_temporal_gnn import TemporalEntityEncoder


def _load_graphs(folder: str) -> List[dgl.DGLGraph]:
    folder = Path(folder)
    graphs = []
    for p in sorted(folder.glob("*.dgl")):
        gs, _ = dgl.load_graphs(str(p))
        graphs.extend(gs)
    return graphs


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, graphs: List[dgl.DGLGraph], labels: List[int]):
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        y = self.labels[idx]
        return g, torch.tensor(y, dtype=torch.long)


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    bg = dgl.batch(graphs)
    y = torch.stack(labels, dim=0)
    return bg, y


def main():
    ap = argparse.ArgumentParser(description="Find misclassified samples in temporal-entity graph classifier")
    ap.add_argument(
        "--graphs", default="Temporal Graph/processed/dgl_temporal", help="Path to folder containing DGL graphs"
    )
    ap.add_argument(
        "--checkpoint", default="Temporal Graph/processed/temporal_entity_ckpt.pt", help="Path to trained checkpoint"
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed for validation split consistency")
    ap.add_argument("--batch-size", type=int, default=16, help="Batch size for inference")
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

    print(f"Validation set size: {len(val_idx)} graphs\n")

    # Create validation dataset
    val_graphs = [graphs[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    val_ds = GraphDataset(val_graphs, val_labels)
    val_loader = GraphDataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
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

    print("Running inference on validation set...\n")

    # Run inference on validation set
    all_y, all_pred = [], []
    with torch.no_grad():
        for bg, y in val_loader:
            bg = bg.to(device)
            y = y.to(device)
            logits = model(bg)
            pred = logits.argmax(dim=-1).cpu()
            all_pred.extend(pred.tolist())
            all_y.extend(y.cpu().tolist())

    # Class names
    class_names = ["Factual", "Reasoning Error", "Incoherence", "Irrelevance", "Overreliance", "Non Factual"]

    # Find misclassified samples
    errors = []
    for i, (true_label, pred_label) in enumerate(zip(all_y, all_pred)):
        if true_label != pred_label:
            errors.append(
                {
                    "val_idx": i,  # Index in validation set (use this in infer_temporal.py)
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "true_name": class_names[true_label],
                    "pred_name": class_names[pred_label],
                }
            )

    # Print results
    print("=" * 100)
    print(f"MISCLASSIFIED SAMPLES: {len(errors)} / {len(all_y)} ({100 * len(errors) / len(all_y):.2f}%)")
    print("=" * 100)
    print()
    print(f"{'Index':<8} {'True Label':<30} {'Predicted Label':<30}")
    print("-" * 100)

    for err in errors:
        true_str = f"{err['true_label']} ({err['true_name']})"
        pred_str = f"{err['pred_label']} ({err['pred_name']})"
        print(f"{err['val_idx']:<8} {true_str:<30} {pred_str:<30}")

    print()
    print("=" * 100)
    print("To inspect a specific misclassified sample, use:")
    print("  Change 'selected_val_idx = val_idx[<INDEX>]' in infer_temporal.py")
    print(f"  where <INDEX> is one of the values from the 'Index' column above")
    print("=" * 100)


if __name__ == "__main__":
    main()
