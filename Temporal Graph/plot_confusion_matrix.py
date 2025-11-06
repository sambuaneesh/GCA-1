# coding: utf-8
"""
Plot confusion matrix for temporal-entity graph classifier.
Loads a trained checkpoint and evaluates on the validation set.

Usage:
  python plot_confusion_matrix.py \
      --graphs processed/dgl_temporal \
      --checkpoint processed/temporal_entity_ckpt.pt \
      --seed 42
"""

import argparse
from pathlib import Path
from typing import List

import dgl
import matplotlib.pyplot as plt
import torch
from dgl.dataloading import GraphDataLoader
from model_temporal_gnn import TemporalEntityEncoder
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


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
    ap = argparse.ArgumentParser(description="Plot confusion matrix for temporal-entity graph classifier")
    ap.add_argument(
        "--graphs", default="Temporal Graph/processed/dgl_temporal", help="Path to folder containing DGL graphs"
    )
    ap.add_argument(
        "--checkpoint", default="Temporal Graph/processed/temporal_entity_ckpt.pt", help="Path to trained checkpoint"
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed for validation split consistency")
    ap.add_argument("--batch-size", type=int, default=16, help="Batch size for inference")
    ap.add_argument("--output", default="Temporal Graph/processed/confusion_matrix.png", help="Output plot path")
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

    print("Running inference on validation set...")

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

    # Compute confusion matrix
    cm = confusion_matrix(all_y, all_pred)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - Validation Set", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save plot
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nConfusion matrix saved to: {output_path}")

    # Optionally display plot
    # plt.show()


if __name__ == "__main__":
    main()
