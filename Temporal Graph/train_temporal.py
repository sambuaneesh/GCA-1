# coding: utf-8
"""
Train the temporal-entity graph classifier.

Usage:
  python train_temporal.py \
      --graphs processed/dgl_temporal \
      --epochs 15 --batch-size 16
"""

import argparse
import json
from pathlib import Path
from typing import List

import dgl
import torch
import torch.nn as nn
from dgl.dataloading import GraphDataLoader
from model_temporal_gnn import TemporalEntityEncoder
from sklearn.metrics import accuracy_score, f1_score


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default="processed/dgl_temporal")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--checkpoint", default="processed/temporal_entity_ckpt.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fixed class count: factual + 5 error subtypes = 6
    # (We still read label_map.json only for transparency or downstream use)
    with open(Path(args.graphs) / "label_map.json", "r", encoding="utf-8") as f:
        _ = json.load(f)
    num_classes = 6

    # load graphs + labels
    graphs = _load_graphs(args.graphs)
    labels = []
    for g in graphs:
        # y stored per node; take first value as graph label
        labels.append(int(g.ndata["y"][0].item()))

    # filter out empty graphs (rare)
    keep = [i for i, g in enumerate(graphs) if g.num_nodes() > 0]
    graphs = [graphs[i] for i in keep]
    labels = [labels[i] for i in keep]

    # splits
    n = len(graphs)
    idx = torch.randperm(n)
    n_train = int(n * 0.8)
    train_idx = idx[:n_train].tolist()
    val_idx = idx[n_train:].tolist()

    train_ds = GraphDataset([graphs[i] for i in train_idx], [labels[i] for i in train_idx])
    val_ds = GraphDataset([graphs[i] for i in val_idx], [labels[i] for i in val_idx])

    train_loader = GraphDataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = GraphDataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    in_dim = graphs[0].ndata["feat"].shape[1]
    edge_dim = (
        graphs[0].edata["e_feat"].shape[1] if graphs[0].num_edges() > 0 else in_dim
    )  # entity emb dim ~ sentence emb dim
    model = TemporalEntityEncoder(
        in_dim=in_dim, edge_dim=edge_dim, hidden_dim=args.hidden, num_layers=args.layers, num_classes=num_classes
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        total_loss = 0.0
        for bg, y in train_loader:
            bg = bg.to(device)
            y = y.to(device)
            logits = model(bg)  # [B, C]
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        # ---- validate ----
        model.eval()
        all_y, all_pred = [], []
        val_loss = 0.0
        with torch.no_grad():
            for bg, y in val_loader:
                bg = bg.to(device)
                y = y.to(device)
                logits = model(bg)
                loss = loss_fn(logits, y)
                val_loss += loss.item()
                pred = logits.argmax(dim=-1).cpu()
                all_pred.extend(pred.tolist())
                all_y.extend(y.cpu().tolist())
        acc = accuracy_score(all_y, all_pred) if all_y else 0.0
        f1 = f1_score(all_y, all_pred, average="macro") if all_y else 0.0

        print(
            f"Epoch {epoch:03d} | train loss {total_loss / max(1, len(train_loader)):.4f} | val loss {val_loss / max(1, len(val_loader)):.4f} | val acc {acc:.4f} | val macro-F1 {f1:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "in_dim": in_dim,
                    "edge_dim": edge_dim,
                    "hidden": args.hidden,
                    "layers": args.layers,
                    "num_classes": num_classes,
                },
                args.checkpoint,
            )
            print(f"  ↳ saved best to {args.checkpoint}")


if __name__ == "__main__":
    main()
