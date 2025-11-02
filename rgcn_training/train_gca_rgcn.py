import argparse
import glob
import os
from typing import List

import dgl
import torch
import torch.nn as nn
import torch.optim as optim
from dgl.dataloading import GraphDataLoader
from gca_rgcn import GCARGCN, GraphLinkPredict


def find_graph_files(data_dir: str, recursive: bool = True) -> List[str]:
    patterns = ("*.dgl", "*.bin")
    files: List[str] = []
    if recursive:
        for root, _, _ in os.walk(data_dir):
            for pat in patterns:
                files.extend(glob.glob(os.path.join(root, pat)))
    else:
        for pat in patterns:
            files.extend(glob.glob(os.path.join(data_dir, pat)))
    return sorted(set(files))


def load_graphs_from_folder(data_dir: str, recursive: bool = True) -> List[dgl.DGLGraph]:
    files = find_graph_files(data_dir, recursive=recursive)
    graphs: List[dgl.DGLGraph] = []
    for f in files:
        gs, _ = dgl.load_graphs(f)
        for g in gs:
            if "feat" not in g.ndata:
                raise ValueError(f"{f} missing ndata['feat'] (float [N, in_dim])")
            if dgl.ETYPE not in g.edata:
                raise ValueError(f"{f} missing edata[dgl.ETYPE] (long [E])")
            graphs.append(g)
    return graphs


class GraphFolderDataset(torch.utils.data.Dataset):
    def __init__(self, graphs: List[dgl.DGLGraph]):
        super().__init__()
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


class NegativeSampler:
    def __init__(self, k: int = 10):
        self.k = k

    def sample(self, pos_triplets: torch.Tensor, num_nodes: int):
        # pos_triplets: [B, 3] (long: s, r, o)
        bsz = pos_triplets.shape[0]
        neg = pos_triplets.repeat(self.k, 1)
        values = torch.randint(0, num_nodes, (bsz * self.k,), device=pos_triplets.device)
        flip = torch.rand(bsz * self.k, device=pos_triplets.device)
        # 50/50 replace head or tail
        neg[flip > 0.5, 0] = values[flip > 0.5]
        neg[flip <= 0.5, 2] = values[flip <= 0.5]
        samples = torch.cat([pos_triplets, neg], dim=0)  # [B*(k+1), 3]
        labels = torch.zeros(samples.shape[0], device=pos_triplets.device)
        labels[:bsz] = 1.0
        return samples, labels


def epoch_metrics(scores: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5):
    """
    scores: logits (pre-sigmoid)
    labels: 0/1
    """
    probs = torch.sigmoid(scores)
    pred = (probs >= threshold).float()
    correct = (pred == labels).float().sum()
    total = labels.numel()
    acc = (correct / total).item()

    tp = ((pred == 1) & (labels == 1)).float().sum().item()
    fp = ((pred == 1) & (labels == 0)).float().sum().item()
    tn = ((pred == 0) & (labels == 0)).float().sum().item()
    fn = ((pred == 0) & (labels == 1)).float().sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return acc, precision, recall, f1


def main():
    ap = argparse.ArgumentParser(description="Train GCA-ready RGCN encoder (with edge-accuracy metrics)")
    ap.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Folder with per-response graphs (.dgl/.bin)",
    )
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--h-dim", type=int, default=256)
    ap.add_argument("--out-dim", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--num-bases", type=int, default=-1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--self-loop", action="store_true", help="Enable self-loop transform")
    ap.add_argument("--neg-k", type=int, default=10, help="Negatives per positive edge")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--checkpoint-out", type=str, default="gca_rgcn.ckpt")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not search subdirectories for graphs",
    )
    ap.add_argument(
        "--auto-demo",
        action="store_true",
        help="If no graphs found, auto-generate a tiny demo dataset here",
    )
    ap.add_argument("--demo-graphs", type=int, default=4)
    ap.add_argument("--demo-nodes", type=int, default=40)
    ap.add_argument("--demo-rels", type=int, default=6)
    ap.add_argument("--demo-feat-dim", type=int, default=768)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load graphs (or optionally create a tiny demo set)
    graphs = load_graphs_from_folder(args.data_dir, recursive=not args.no_recursive)
    if len(graphs) == 0:
        if args.auto_demo:
            print(f"No graphs found in {args.data_dir}. Creating a tiny demo dataset...")
            # simple demo data — random graphs with typed edges and random features
            os.makedirs(args.data_dir, exist_ok=True)
            import random

            for i in range(args.demo_graphs):
                N, E = args.demo_nodes, args.demo_nodes * 3
                src = torch.randint(0, N, (E,))
                dst = torch.randint(0, N, (E,))
                g = dgl.graph((src, dst), num_nodes=N)
                g.ndata["feat"] = torch.randn(N, args.demo_feat_dim)
                g.edata[dgl.ETYPE] = torch.randint(0, args.demo_rels, (E,), dtype=torch.int64)
                dgl.save_graphs(os.path.join(args.data_dir, f"demo_{i:04d}.dgl"), [g])
            graphs = load_graphs_from_folder(args.data_dir, recursive=not args.no_recursive)
        else:
            raise AssertionError(
                f"No .dgl/.bin graphs found in {args.data_dir}. Add graphs or use --auto-demo to generate a tiny set"
            )

    # Infer dims and num_rels from dataset
    in_dim = graphs[0].ndata["feat"].shape[1]
    num_rels = 0
    for g in graphs:
        num_rels = max(num_rels, int(g.edata[dgl.ETYPE].max().item()) + 1)
    print(f"Loaded {len(graphs)} graphs | in_dim={in_dim} | num_rels={num_rels}")

    # Dataloader
    dataset = GraphFolderDataset(graphs)
    loader = GraphDataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model + link-pred head
    encoder = GCARGCN(
        in_dim=in_dim,
        h_dim=args.h_dim,
        out_dim=args.out_dim,
        num_rels=num_rels,
        num_layers=args.num_layers,
        num_bases=args.num_bases,
        dropout=args.drop_out if hasattr(args, "drop_out") else args.dropout,
        self_loop=args.self_loop,
    ).to(device)
    model = GraphLinkPredict(encoder).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    neg_sampler = NegativeSampler(k=args.neg_k)

    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        steps = 0

        # epoch-level metrics
        all_scores = []
        all_labels = []

        for batch_g in loader:
            batch_g = batch_g.to(device)
            feats = batch_g.ndata["feat"].to(device)
            if "norm" not in batch_g.edata:
                batch_g.edata["norm"] = dgl.norm_by_dst(batch_g).unsqueeze(1)

            src, dst = batch_g.edges()
            rel = batch_g.edata[dgl.ETYPE]
            pos_triplets = torch.stack([src, rel, dst], dim=1).to(device)

            samples, labels = neg_sampler.sample(pos_triplets, batch_g.num_nodes())

            embed = model(batch_g, feats)
            loss = model.get_loss(embed, samples, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # collect logits for metrics
            with torch.no_grad():
                scores = model.calc_score(embed, samples)  # logits
                all_scores.append(scores.detach())
                all_labels.append(labels.detach())

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(steps, 1)
        # compute epoch-level metrics
        with torch.no_grad():
            all_scores = torch.cat(all_scores, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            acc, prec, rec, f1 = epoch_metrics(all_scores, all_labels, threshold=0.5)

        print(f"Epoch {epoch:03d} | loss {avg_loss:.4f} | acc {acc:.4f} | P {prec:.4f} | R {rec:.4f} | F1 {f1:.4f}")

        # Save best checkpoint by loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt = {
                "state_dict": model.state_dict(),
                "epoch": epoch,
                # Encoder config for later loading inside GCA
                "in_dim": in_dim,
                "h_dim": encoder.h_dim,
                "out_dim": encoder.out_dim,
                "num_rels": encoder.num_rels,
                "num_layers": encoder.num_layers,
                "num_bases": encoder.num_bases,
                "dropout": encoder.dropout.p,
                "self_loop": args.self_loop,
            }
            torch.save(ckpt, args.checkpoint_out)
            print(f"  -> Saved best checkpoint to {args.checkpoint_out}")

    print("Training complete.")


if __name__ == "__main__":
    main()
