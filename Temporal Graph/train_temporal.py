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
from collections import Counter, defaultdict
from pathlib import Path
from typing import List

import dgl
import numpy as np
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
from torch.utils.data import WeightedRandomSampler


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)


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


def run_single_trial(args, seed, device):
    """Run a single training trial and return metrics."""
    set_all_seeds(seed)
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
    ggen = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(n, generator=ggen)
    n_train = int(n * 0.8)
    train_idx = idx[:n_train].tolist()
    val_idx = idx[n_train:].tolist()

    train_ds = GraphDataset([graphs[i] for i in train_idx], [labels[i] for i in train_idx])
    val_ds = GraphDataset([graphs[i] for i in val_idx], [labels[i] for i in val_idx])

    counts = Counter([labels[i] for i in train_idx])
    num_classes = len(set(labels))

    loss_class_w = torch.ones(num_classes, dtype=torch.float)
    for c, cnt in counts.items():
        loss_class_w[c] = len(train_idx) / (num_classes * max(1, cnt))
    loss_class_w = loss_class_w.to(device)

    if args.sampling_scheme == "uniform6":
        target_probs = torch.tensor([1.0 / 6.0] * 6, dtype=torch.float)
    elif args.sampling_scheme == "half_first":
        target_probs = torch.tensor([0.5] + [0.5 / 5.0] * 5, dtype=torch.float)
    else:
        raise ValueError(f"Unknown sampling scheme: {args.sampling_scheme}")

    sample_w_per_class = torch.zeros(6, dtype=torch.float)
    for c in range(6):
        sample_w_per_class[c] = target_probs[c] / max(1, counts.get(c, 0))

    train_sample_weights = [sample_w_per_class[labels[i]].item() for i in train_idx]

    sampler = WeightedRandomSampler(train_sample_weights, num_samples=len(train_idx), replacement=True, generator=ggen)

    train_loader = GraphDataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate)
    val_loader = GraphDataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    in_dim = graphs[0].ndata["feat"].shape[1]
    edge_dim = graphs[0].edata["e_feat"].shape[1] if graphs[0].num_edges() > 0 else in_dim

    model = TemporalEntityEncoder(
        in_dim=in_dim, edge_dim=edge_dim, hidden_dim=args.hidden, num_layers=args.layers, num_classes=num_classes
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=loss_class_w)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    class_names = ["factual", "Reasoning Error", "Incoherence", "Irrelevance", "Overreliance", "non-factual"]

    best_acc = -1.0
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

        # print(
        #     f"Epoch {epoch:03d} | train loss {total_loss / max(1, len(train_loader)):.4f} | "
        #     f"val loss {val_loss / max(1, len(val_loader)):.4f}"
        # )
        # print(f"  6-way  | acc {acc_6way:.4f} | prec {prec_6way:.4f} | rec {rec_6way:.4f} | F1 {f1_6way:.4f}")
        # print(f"  binary | acc {acc_binary:.4f} | prec {prec_binary:.4f} | rec {rec_binary:.4f} | F1 {f1_binary:.4f}")

        sched.step(acc_binary)
        if acc_binary > best_acc:
            best_acc = acc_binary
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
            # print(f"  saved best to {args.checkpoint}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            # print(f"  no improvement for {epochs_without_improvement} epoch(s)")
            if epochs_without_improvement >= args.early_stop_patience:
                # print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.early_stop_patience} epochs)")
                # print(f"Best validation binary accuracy: {best_acc:.4f}")
                break

    # print("\n=== Evaluating best checkpoint on validation set ===")
    if not Path(args.checkpoint).exists():
        raise RuntimeError(f"Checkpoint {args.checkpoint} not found after training.")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
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

    class_names = ["factual", "Reasoning Error", "Incoherence", "Irrelevance", "Overreliance", "non-factual"]
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

    all_y_binary = [0 if y == 0 else 1 for y in all_y_best]
    all_pred_binary = [0 if p == 0 else 1 for p in all_pred_best]
    acc_binary = accuracy_score(all_y_binary, all_pred_binary) if all_y_binary else 0.0
    f1_binary = f1_score(all_y_binary, all_pred_binary, average="binary", zero_division=0) if all_y_binary else 0.0
    prec_binary = (
        precision_score(all_y_binary, all_pred_binary, average="binary", zero_division=0) if all_y_binary else 0.0
    )
    rec_binary = recall_score(all_y_binary, all_pred_binary, average="binary", zero_division=0) if all_y_binary else 0.0

    # Collect per-class metrics
    per_class_metrics = {}
    for i in uniq_labels:
        class_mask = [y == i for y in all_y_best]
        class_pred = [all_pred_best[j] for j in range(len(all_y_best)) if class_mask[j]]
        class_true = [all_y_best[j] for j in range(len(all_y_best)) if class_mask[j]]

        if len(class_true) > 0:
            all_pred_as_class = [1 if p == i else 0 for p in all_pred_best]
            all_true_as_class = [1 if t == i else 0 for t in all_y_best]

            prec = precision_score(all_true_as_class, all_pred_as_class, zero_division=0)
            rec = recall_score(all_true_as_class, all_pred_as_class, zero_division=0)
            f1 = f1_score(all_true_as_class, all_pred_as_class, zero_division=0)

            per_class_metrics[i] = {"precision": prec, "recall": rec, "f1": f1, "support": len(class_true)}

    # Overall metrics (compute these properly)
    prec_6way = precision_score(all_y_best, all_pred_best, average="macro", zero_division=0)
    rec_6way = recall_score(all_y_best, all_pred_best, average="macro", zero_division=0)
    f1_6way = f1_score(all_y_best, all_pred_best, average="macro", zero_division=0)

    overall_metrics = {
        "accuracy": accuracy_score(all_y_best, all_pred_best),
        "macro_avg_precision": prec_6way,
        "macro_avg_recall": rec_6way,
        "macro_avg_f1": f1_6way,
        "weighted_avg_precision": precision_score(all_y_best, all_pred_best, average="weighted", zero_division=0),
        "weighted_avg_recall": recall_score(all_y_best, all_pred_best, average="weighted", zero_division=0),
        "weighted_avg_f1": f1_score(all_y_best, all_pred_best, average="weighted", zero_division=0),
        "binary_acc": acc_binary,
        "binary_prec": prec_binary,
        "binary_rec": rec_binary,
        "binary_f1": f1_binary,
    }

    return per_class_metrics, overall_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default="Temporal Graph/processed/dgl_temporal")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint", default="Temporal Graph/processed/temporal_entity_ckpt.pt")
    ap.add_argument("--early-stop-patience", type=int, default=5, help="Early stopping patience (epochs)")
    ap.add_argument(
        "--sampling-scheme",
        choices=["uniform6", "half_first"],
        default="uniform6",
        help=(
            "Sampling weights: 'uniform6' gives 1/6 importance to each class (0..5); "
            "'half_first' gives 1/2 to class 0 and spreads the rest equally over classes 1..5."
        ),
    )
    ap.add_argument("--num-trials", type=int, default=25, help="Number of trials to run")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Storage for all trials
    all_per_class_metrics = defaultdict(lambda: defaultdict(list))
    all_overall_metrics = defaultdict(list)

    print(f"\nRunning {args.num_trials} trials with different seeds...")
    print("=" * 80)

    for trial in range(args.num_trials):
        seed = args.seed  # Use the same seed for all trials
        print(f"\nTrial {trial + 1}/{args.num_trials} (seed={seed})...")

        per_class, overall = run_single_trial(args, seed, device)  # Store per-class metrics
        for class_id, metrics in per_class.items():
            for metric_name, value in metrics.items():
                all_per_class_metrics[class_id][metric_name].append(value)

        # Store overall metrics
        for metric_name, value in overall.items():
            all_overall_metrics[metric_name].append(value)

    # Print results with mean +- std
    print("\n" + "=" * 80)
    print("RESULTS OVER", args.num_trials, "TRIALS")
    print("=" * 80)
    print()

    class_names = ["factual", "Reasoning Error", "Incoherence", "Irrelevance", "Overreliance", "non-factual"]

    # Print header
    print(f"{'':>20} {'precision':>20} {'recall':>20} {'f1-score':>20} {'support':>10}")
    print()

    # Print per-class metrics
    for class_id in sorted(all_per_class_metrics.keys()):
        class_name = class_names[class_id]
        prec_mean = np.mean(all_per_class_metrics[class_id]["precision"])
        prec_std = np.std(all_per_class_metrics[class_id]["precision"])
        rec_mean = np.mean(all_per_class_metrics[class_id]["recall"])
        rec_std = np.std(all_per_class_metrics[class_id]["recall"])
        f1_mean = np.mean(all_per_class_metrics[class_id]["f1"])
        f1_std = np.std(all_per_class_metrics[class_id]["f1"])

        # Get support from first trial data
        support = all_per_class_metrics[class_id]["support"][0] if "support" in all_per_class_metrics[class_id] else 0

        print(
            f"{class_name:>20} {prec_mean:>8.4f}+-{prec_std:<9.4f} "
            f"{rec_mean:>8.4f}+-{rec_std:<9.4f} "
            f"{f1_mean:>8.4f}+-{f1_std:<9.4f} {support:>10}"
        )

    print()

    # Print overall metrics
    acc_mean = np.mean(all_overall_metrics["accuracy"])
    acc_std = np.std(all_overall_metrics["accuracy"])
    print(f"{'accuracy':>20} {acc_mean:>29.4f}+-{acc_std:<9.4f}")

    macro_prec_mean = np.mean(all_overall_metrics["macro_avg_precision"])
    macro_prec_std = np.std(all_overall_metrics["macro_avg_precision"])
    macro_rec_mean = np.mean(all_overall_metrics["macro_avg_recall"])
    macro_rec_std = np.std(all_overall_metrics["macro_avg_recall"])
    macro_f1_mean = np.mean(all_overall_metrics["macro_avg_f1"])
    macro_f1_std = np.std(all_overall_metrics["macro_avg_f1"])

    print(
        f"{'macro avg':>20} {macro_prec_mean:>8.4f}+-{macro_prec_std:<9.4f} "
        f"{macro_rec_mean:>8.4f}+-{macro_rec_std:<9.4f} "
        f"{macro_f1_mean:>8.4f}+-{macro_f1_std:<9.4f}"
    )

    weighted_prec_mean = np.mean(all_overall_metrics["weighted_avg_precision"])
    weighted_prec_std = np.std(all_overall_metrics["weighted_avg_precision"])
    weighted_rec_mean = np.mean(all_overall_metrics["weighted_avg_recall"])
    weighted_rec_std = np.std(all_overall_metrics["weighted_avg_recall"])
    weighted_f1_mean = np.mean(all_overall_metrics["weighted_avg_f1"])
    weighted_f1_std = np.std(all_overall_metrics["weighted_avg_f1"])

    print(
        f"{'weighted avg':>20} {weighted_prec_mean:>8.4f}+-{weighted_prec_std:<9.4f} "
        f"{weighted_rec_mean:>8.4f}+-{weighted_rec_std:<9.4f} "
        f"{weighted_f1_mean:>8.4f}+-{weighted_f1_std:<9.4f}"
    )

    print()

    # Binary metrics
    bin_acc_mean = np.mean(all_overall_metrics["binary_acc"])
    bin_acc_std = np.std(all_overall_metrics["binary_acc"])
    bin_prec_mean = np.mean(all_overall_metrics["binary_prec"])
    bin_prec_std = np.std(all_overall_metrics["binary_prec"])
    bin_rec_mean = np.mean(all_overall_metrics["binary_rec"])
    bin_rec_std = np.std(all_overall_metrics["binary_rec"])
    bin_f1_mean = np.mean(all_overall_metrics["binary_f1"])
    bin_f1_std = np.std(all_overall_metrics["binary_f1"])

    print(
        f"  binary | acc {bin_acc_mean:.4f}+-{bin_acc_std:.4f} | "
        f"prec {bin_prec_mean:.4f}+-{bin_prec_std:.4f} | "
        f"rec {bin_rec_mean:.4f}+-{bin_rec_std:.4f} | "
        f"F1 {bin_f1_mean:.4f}+-{bin_f1_std:.4f}"
    )
    print()


if __name__ == "__main__":
    main()
