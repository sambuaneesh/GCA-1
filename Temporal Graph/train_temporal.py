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
import os
import random
from pathlib import Path
from typing import List

import dgl
import torch
import torch.nn as nn
from dgl.dataloading import GraphDataLoader
from model_temporal_gnn import TemporalEntityEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

SEED = 42


def set_all_seeds(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    ap.add_argument("--graphs", default="Temporal Graph/processed/dgl_temporal")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--checkpoint", default="Temporal Graph/processed/temporal_entity_ckpt.pt")
    ap.add_argument("--early-stop-patience", type=int, default=5, help="Early stopping patience (epochs)")
    args = ap.parse_args()

    set_all_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(Path(args.graphs) / "label_map.json", "r", encoding="utf-8") as f:
        _ = json.load(f)

    graphs = _load_graphs(args.graphs)
    if not graphs:
        raise RuntimeError("No graphs found. Check --graphs path.")

    labels = [int(g.ndata["y"][0].item()) for g in graphs]

    keep = [i for i, g in enumerate(graphs) if g.num_nodes() > 0]
    graphs = [graphs[i] for i in keep]
    labels = [labels[i] for i in keep]

    n = len(graphs)
    ggen = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(n, generator=ggen)
    n_train = int(n * 0.8)
    train_idx = idx[:n_train].tolist()
    val_idx = idx[n_train:].tolist()

    train_ds = GraphDataset([graphs[i] for i in train_idx], [labels[i] for i in train_idx])
    val_ds = GraphDataset([graphs[i] for i in val_idx], [labels[i] for i in val_idx])

    from collections import Counter

    counts = Counter([labels[i] for i in train_idx])
    num_classes = len(set(labels))
    class_w = torch.ones(num_classes, dtype=torch.float)
    for c, cnt in counts.items():
        class_w[c] = len(train_idx) / (num_classes * max(1, cnt))
    class_w = class_w.to(device)

    train_sample_weights = [class_w[labels[i]].item() for i in train_idx]
    from torch.utils.data import WeightedRandomSampler

    sampler = WeightedRandomSampler(train_sample_weights, num_samples=len(train_idx), replacement=True)

    train_loader = GraphDataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate)
    val_loader = GraphDataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    in_dim = graphs[0].ndata["feat"].shape[1]
    edge_dim = graphs[0].edata["e_feat"].shape[1] if graphs[0].num_edges() > 0 else in_dim

    model = TemporalEntityEncoder(
        in_dim=in_dim, edge_dim=edge_dim, hidden_dim=args.hidden, num_layers=args.layers, num_classes=num_classes
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=class_w)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3, verbose=True)

    class_names = ["factual", "Reasoning Error", "Incoherence", "Irrelevance", "Overreliance", "non-factual"]

    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for bg, y in train_loader:
            bg = bg.to(device)
            y = y.to(device)
            logits = model(bg)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

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

        acc_6way = accuracy_score(all_y, all_pred) if all_y else 0.0
        f1_6way = f1_score(all_y, all_pred, average="macro", zero_division=0) if all_y else 0.0
        prec_6way = precision_score(all_y, all_pred, average="macro", zero_division=0) if all_y else 0.0
        rec_6way = recall_score(all_y, all_pred, average="macro", zero_division=0) if all_y else 0.0

        all_y_binary = [0 if y == 0 else 1 for y in all_y]
        all_pred_binary = [0 if p == 0 else 1 for p in all_pred]
        acc_binary = accuracy_score(all_y_binary, all_pred_binary) if all_y_binary else 0.0
        f1_binary = f1_score(all_y_binary, all_pred_binary, average="binary", zero_division=0) if all_y_binary else 0.0
        prec_binary = (
            precision_score(all_y_binary, all_pred_binary, average="binary", zero_division=0) if all_y_binary else 0.0
        )
        rec_binary = (
            recall_score(all_y_binary, all_pred_binary, average="binary", zero_division=0) if all_y_binary else 0.0
        )

        print(
            f"Epoch {epoch:03d} | train loss {total_loss / max(1, len(train_loader)):.4f} | "
            f"val loss {val_loss / max(1, len(val_loader)):.4f}"
        )
        print(f"  6-way  | acc {acc_6way:.4f} | prec {prec_6way:.4f} | rec {rec_6way:.4f} | F1 {f1_6way:.4f}")
        print(f"  binary | acc {acc_binary:.4f} | prec {prec_binary:.4f} | rec {rec_binary:.4f} | F1 {f1_binary:.4f}")

        sched.step(f1_6way)
        if f1_6way > best_f1:
            best_f1 = f1_6way
            best_epoch = epoch
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
            print(f"  saved best to {args.checkpoint}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"  no improvement for {epochs_without_improvement} epoch(s)")
            if epochs_without_improvement >= args.early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.early_stop_patience} epochs)")
                print(f"Best validation macro-F1: {best_f1:.4f}")
                break

    print("\n=== Evaluating best checkpoint on validation set ===")
    if not Path(args.checkpoint).exists():
        raise RuntimeError(f"Checkpoint {args.checkpoint} not found after training.")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    all_y_best, all_pred_best = [], []
    with torch.no_grad():
        for bg, y in val_loader:
            bg = bg.to(device)
            y = y.to(device)
            logits = model(bg)
            pred = logits.argmax(dim=-1).cpu()
            all_pred_best.extend(pred.tolist())
            all_y_best.extend(y.cpu().tolist())

    uniq_labels = sorted(set(all_y_best))
    target_names_6 = [class_names[i] if i < len(class_names) else f"class_{i}" for i in uniq_labels]
    report_6way = classification_report(
        all_y_best,
        all_pred_best,
        labels=uniq_labels,
        target_names=target_names_6,
        zero_division=0,
        digits=4,
    )
    print(report_6way)

    all_y_best_bin = [0 if y == 0 else 1 for y in all_y_best]
    all_pred_best_bin = [0 if p == 0 else 1 for p in all_pred_best]
    uniq_labels_bin = sorted(set(all_y_best_bin))
    target_names_bin = ["factual (y=0)", "non-factual (y>0)"]
    target_names_bin = [target_names_bin[i] for i in uniq_labels_bin]
    report_binary = classification_report(
        all_y_best_bin,
        all_pred_best_bin,
        labels=uniq_labels_bin,
        target_names=target_names_bin,
        zero_division=0,
        digits=4,
    )
    print(report_binary)
    print(f"Best macro-F1 on validation during training: {best_f1:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
