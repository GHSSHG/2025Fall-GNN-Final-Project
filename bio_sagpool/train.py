from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from .dataset import EpitopeCdr3BindingDataset
from .metrics import best_threshold_by_f1, compute_binary_metrics
from .model import SagPoolBindingNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(requested: str | None = None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_splits(n: int, *, seed: int, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n <= 0:
        raise ValueError("n must be > 0")
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("invalid split ratios")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    return train_idx, val_idx, test_idx


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    total_loss = 0.0
    ys = []
    probs = []

    for data in loader:
        data = data.to(device)
        logits = model(data)
        y = data.y.view(-1)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * y.numel()
        ys.append(y.detach().cpu().numpy())
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())

    y_true = np.concatenate(ys, axis=0)
    y_prob = np.concatenate(probs, axis=0)
    avg_loss = total_loss / float(len(loader.dataset))
    return y_true, y_prob, float(avg_loss)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/data.tsv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--aa_emb_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--pooling_ratio", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None, help="e.g. mps/cpu/cuda")
    parser.add_argument("--out_dir", type=str, default="outputs/bio_sagpool")
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars (clean logs).")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    if device.type == "cpu":
        raise RuntimeError(
            "MPS/CUDA not available. This project is configured to run on GPU (Mac MPS or CUDA). "
            "If you really need CPU, pass --device cpu and remove this guard."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_path, sep="\t")
    if not {"CDR3", "Epitope", "label"}.issubset(df.columns):
        raise ValueError(f"Expected columns: CDR3, Epitope, label. Got: {list(df.columns)}")

    epitopes = df["Epitope"].astype(str).tolist()
    cdr3s = df["CDR3"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    n = len(labels)
    train_idx, val_idx, test_idx = make_splits(n, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    dataset = EpitopeCdr3BindingDataset(epitopes=epitopes, cdr3s=cdr3s, labels=labels)

    train_ds = torch.utils.data.Subset(dataset, train_idx.tolist())
    val_ds = torch.utils.data.Subset(dataset, val_idx.tolist())
    test_ds = torch.utils.data.Subset(dataset, test_idx.tolist())

    def _count_pos(subset) -> Tuple[int, int]:
        ys = [labels[i] for i in subset.indices]  # type: ignore[attr-defined]
        pos = int(sum(ys))
        neg = int(len(ys) - pos)
        return pos, neg

    train_pos, train_neg = _count_pos(train_ds)
    val_pos, val_neg = _count_pos(val_ds)
    test_pos, test_neg = _count_pos(test_ds)

    print(f"Device: {device}")
    print(f"Total: {n} | pos={sum(labels)} neg={n-sum(labels)} pos_rate={sum(labels)/n:.4f}")
    print(f"Train: {len(train_ds)} | pos={train_pos} neg={train_neg} pos_rate={train_pos/len(train_ds):.4f}")
    print(f"Val:   {len(val_ds)} | pos={val_pos} neg={val_neg} pos_rate={val_pos/len(val_ds):.4f}")
    print(f"Test:  {len(test_ds)} | pos={test_pos} neg={test_neg} pos_rate={test_pos/len(test_ds):.4f}")

    # Class imbalance handling via pos_weight.
    pos_weight = torch.tensor([train_neg / max(train_pos, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SagPoolBindingNet(
        aa_emb_dim=args.aa_emb_dim,
        hidden_dim=args.hidden_dim,
        pooling_ratio=args.pooling_ratio,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_pr_auc = -1.0
    best_epoch = -1
    best_threshold = 0.5
    patience = 0
    ckpt_path = out_dir / "checkpoints" / "best.pt"

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0

        for data in tqdm(
            train_loader,
            desc=f"epoch {epoch}/{args.epochs}",
            leave=False,
            disable=bool(args.no_progress),
        ):
            data = data.to(device)
            logits = model(data)
            y = data.y.view(-1)
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * y.numel()
            n_seen += int(y.numel())

        train_loss = total_loss / max(n_seen, 1)

        y_true, y_prob, val_loss = collect_predictions(model, val_loader, device=device, criterion=criterion)
        threshold = 0.5
        if len(np.unique(y_true)) == 2:
            threshold = best_threshold_by_f1(y_true, y_prob)
        val_metrics = compute_binary_metrics(y_true, y_prob, loss=val_loss, threshold=threshold).as_dict() | {"n": int(len(val_ds))}

        history.append({"epoch": epoch, "train_loss": train_loss, "val": val_metrics})

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_pr_auc={val_metrics.get('pr_auc')} val_roc_auc={val_metrics.get('roc_auc')} "
            f"val_f1={val_metrics['f1']:.4f} thr={val_metrics['threshold']:.3f}"
        )

        val_pr_auc = val_metrics.get("pr_auc")
        improved = val_pr_auc is not None and float(val_pr_auc) > best_val_pr_auc
        if improved:
            best_val_pr_auc = float(val_pr_auc)
            best_epoch = epoch
            best_threshold = float(val_metrics["threshold"])
            patience = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
                ckpt_path,
            )
        else:
            patience += 1
            if patience > args.patience:
                print(f"Early stopping at epoch {epoch} (best_epoch={best_epoch}, best_val_pr_auc={best_val_pr_auc:.4f}).")
                break

    # Load best model and evaluate on test.
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    y_true, y_prob, test_loss = collect_predictions(model, test_loader, device=device, criterion=criterion)
    test_metrics = compute_binary_metrics(y_true, y_prob, loss=test_loss, threshold=best_threshold).as_dict() | {"n": int(len(test_ds))}
    results = {
        "best_epoch": best_epoch,
        "best_val_pr_auc": best_val_pr_auc,
        "best_threshold": best_threshold,
        "test": test_metrics,
        "splits": {
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "sizes": {
                "train": len(train_ds),
                "val": len(val_ds),
                "test": len(test_ds),
            },
        },
        "device": str(device),
    }

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print("Test metrics:")
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))
    print(f"Saved: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
