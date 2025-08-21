#!/usr/bin/env python3
"""
Deep Learning baselines for Wine Quality Prediction (PyTorch)

This module adds two PyTorch models to complement classical ML baselines:
- MLPClassifier: a configurable multilayer perceptron with Dropout + BatchNorm
- MultiTowerNet: a simple "multi‑tower" architecture that splits features into groups,
  learns separate towers, then concatenates and classifies.

It includes: stratified train/val/test split, StandardScaler, early stopping,
learning‑rate scheduling, and a lightweight ablation framework to toggle
regularization components. Metrics: accuracy, macro-F1, confusion matrix.

Run examples
-----------
# Basic MLP with BN + Dropout + EarlyStopping
python deep_learning_wine.py --csv data/winequality-red.csv --target quality \
    --model mlp --epochs 100 --patience 20 --hidden-sizes 128 64 --dropout 0.3 --batch-norm

# Multi‑tower net (3 towers) with ablation (disable BN + Dropout)
python deep_learning_wine.py --csv data/winequality-red.csv --target quality \
    --model multitower --towers 3 --tower-size 64 --no-batch-norm --dropout 0.0

# Reproducible ablation sweep over regularizers (results saved under runs/)
python deep_learning_wine.py --csv data/winequality-red.csv --target quality \
    --model mlp --ablate all

Author: (c) 2025 Charlie Z. (MIT License)
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset

# ----------------------
# Utils
# ----------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stratified_split(X: pd.DataFrame, y: pd.Series, test_size=0.2, val_size=0.2, seed=42):
    """Stratified train/val/test split.
    val_size is taken as a fraction of the remaining train portion.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=val_size, stratify=y_train, random_state=seed
    )
    return (X_tr, y_tr), (X_val, y_val), (X_test, y_test)


@dataclass
class DLConfig:
    input_dim: int
    n_classes: int
    hidden_sizes: List[int]
    dropout: float = 0.3
    batch_norm: bool = True
    towers: int = 0  # used for MultiTowerNet
    tower_size: int = 64


class MLPClassifier(nn.Module):
    def __init__(self, cfg: DLConfig):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            if cfg.batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if cfg.dropout and cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, cfg.n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MultiTowerNet(nn.Module):
    """A simple multi‑tower architecture.
    Splits features into `towers` roughly equal groups, each with a small MLP.
    Concatenated tower outputs go through a head MLP.
    """

    def __init__(self, cfg: DLConfig):
        super().__init__()
        assert cfg.towers >= 2, "Use at least 2 towers or switch to --model mlp"
        self.towers = nn.ModuleList()
        # compute feature splits
        splits = np.array_split(np.arange(cfg.input_dim), cfg.towers)
        self.feature_slices: List[np.ndarray] = [np.array(s, dtype=int) for s in splits]

        def make_tower(in_dim: int):
            layers: List[nn.Module] = [nn.Linear(in_dim, cfg.tower_size)]
            if cfg.batch_norm:
                layers.append(nn.BatchNorm1d(cfg.tower_size))
            layers.append(nn.ReLU())
            if cfg.dropout and cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            layers.append(nn.Linear(cfg.tower_size, cfg.tower_size))
            if cfg.batch_norm:
                layers.append(nn.BatchNorm1d(cfg.tower_size))
            layers.append(nn.ReLU())
            if cfg.dropout and cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            return nn.Sequential(*layers)

        for s in self.feature_slices:
            self.towers.append(make_tower(len(s)))

        concat_dim = cfg.tower_size * cfg.towers
        head_layers: List[nn.Module] = [nn.Linear(concat_dim, concat_dim // 2)]
        if cfg.batch_norm:
            head_layers.append(nn.BatchNorm1d(concat_dim // 2))
        head_layers += [nn.ReLU()]
        if cfg.dropout and cfg.dropout > 0:
            head_layers.append(nn.Dropout(cfg.dropout))
        head_layers.append(nn.Linear(concat_dim // 2, cfg.n_classes))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x):
        outs: List[torch.Tensor] = []
        for s, tower in zip(self.feature_slices, self.towers):
            outs.append(tower(x[:, s]))
        return self.head(torch.cat(outs, dim=1))


# ----------------------
# Data loader
# ----------------------

class NPTensorDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ----------------------
# Training loop w/ early stopping
# ----------------------

@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 20  # for early stopping
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                tcfg: TrainConfig,
                n_classes: int,
                out_dir: Path) -> Tuple[nn.Module, dict]:
    model.to(tcfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()

    best_val = float('inf')
    best_state = None
    es_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, tcfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(tcfg.device)
            yb = yb.to(tcfg.device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(tcfg.device)
                yb = yb.to(tcfg.device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                preds = logits.argmax(1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(yb.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_acc = float(accuracy_score(y_true, y_pred))
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # early stopping
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            es_counter = 0
        else:
            es_counter += 1
            if es_counter >= tcfg.patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    # save
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "best_model.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    return model, history


def evaluate(model: nn.Module, loader: DataLoader, device: str, class_names: List[str]):
    model.eval(); model.to(device)
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(yb.numpy())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    return {"accuracy": float(acc), "macro_f1": float(f1), "confusion_matrix": cm.tolist(), "report": report}


# ----------------------
# Data ingest
# ----------------------

def load_csv(csv_path: Path, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    assert target in df.columns, f"Target column '{target}' not found in CSV."
    y = df[target].astype(int)
    X = df.drop(columns=[target])
    # if any non‑numeric columns exist, one‑hot encode
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        X = pd.get_dummies(X, columns=non_numeric, drop_first=True)
    return X, y


# ----------------------
# Ablation presets
# ----------------------

ABLATIONS = {
    "baseline": {"batch_norm": True, "dropout": 0.3, "patience": 20},
    "no_dropout": {"batch_norm": True, "dropout": 0.0, "patience": 20},
    "no_batchnorm": {"batch_norm": False, "dropout": 0.3, "patience": 20},
    "no_earlystop": {"batch_norm": True, "dropout": 0.3, "patience": 10**9},
}


# ----------------------
# Main
# ----------------------

def main():
    p = argparse.ArgumentParser(description="PyTorch deep learning baselines for Wine Quality")
    p.add_argument('--csv', type=Path, required=True, help='Path to CSV (e.g., winequality-red.csv)')
    p.add_argument('--target', type=str, default='quality', help='Name of the target column')
    p.add_argument('--model', type=str, choices=['mlp', 'multitower'], default='mlp')
    p.add_argument('--hidden-sizes', type=int, nargs='+', default=[128, 64])
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--batch-norm', dest='batch_norm', action='store_true')
    p.add_argument('--no-batch-norm', dest='batch_norm', action='store_false')
    p.set_defaults(batch_norm=True)
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--ablate', type=str, choices=['none', 'all'], default='none', help='Run ablation presets')
    # multi‑tower specific
    p.add_argument('--towers', type=int, default=0, help='Number of towers (>=2 to enable)')
    p.add_argument('--tower-size', type=int, default=64)
    p.add_argument('--output', type=Path, default=Path('runs/latest'))

    args = p.parse_args()
    set_seed(args.seed)

    X, y = load_csv(args.csv, args.target)
    class_names = sorted(y.unique())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    y_idx = y.map(class_to_idx)

    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = stratified_split(X, y_idx, test_size=0.2, val_size=0.2, seed=args.seed)

    scaler = StandardScaler()
    X_trn = scaler.fit_transform(X_tr)
    X_valn = scaler.transform(X_val)
    X_ten = scaler.transform(X_te)

    train_ds = NPTensorDataset(X_trn, y_tr.values)
    val_ds = NPTensorDataset(X_valn, y_val.values)
    test_ds = NPTensorDataset(X_ten, y_te.values)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    cfg = DLConfig(
        input_dim=X_trn.shape[1],
        n_classes=len(class_names),
        hidden_sizes=args.hidden_sizes,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
        towers=args.towers,
        tower_size=args.tower_size,
    )

    # Select model
    if args.model == 'mlp' or args.towers < 2:
        model = MLPClassifier(cfg)
        model_name = 'mlp'
    else:
        assert args.towers >= 2, "--towers must be >=2 for multitower"
        model = MultiTowerNet(cfg)
        model_name = f'multitower_{args.towers}x{args.tower_size}'

    # Handle ablation sweeps
    if args.ablate == 'all':
        results = {}
        for tag, preset in ABLATIONS.items():
            print(f"\n=== Running ablation: {tag} ===")
            cfg.batch_norm = preset['batch_norm']
            cfg.dropout = preset['dropout']

            # Re‑init model each ablation
            if model_name.startswith('mlp'):
                model = MLPClassifier(cfg)
            else:
                model = MultiTowerNet(cfg)

            tcfg = TrainConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=preset['patience'],
            )
            out_dir = args.output / f"{model_name}_abl_{tag}"
            model, hist = train_model(model, train_loader, val_loader, tcfg, cfg.n_classes, out_dir)
            metrics = evaluate(model, test_loader, tcfg.device, [str(c) for c in class_names])
            with open(out_dir / 'metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            results[tag] = metrics
        # write combined summary
        (args.output / 'summaries').mkdir(parents=True, exist_ok=True)
        with open(args.output / 'summaries' / f'{model_name}_ablations.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\nAblation results:\n", json.dumps(results, indent=2))
        return

    # Single run
    tcfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )
    out_dir = args.output / model_name
    model, hist = train_model(model, train_loader, val_loader, tcfg, cfg.n_classes, out_dir)
    metrics = evaluate(model, test_loader, tcfg.device, [str(c) for c in class_names])

    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save scaler + class mapping for reproducibility
    import joblib
    joblib.dump({'scaler': scaler, 'class_to_idx': class_to_idx}, out_dir / 'artifacts.joblib')

    print("\nTest metrics:\n", json.dumps(metrics, indent=2))
    print(f"\nArtifacts saved under: {out_dir.resolve()}")


if __name__ == '__main__':
    main()
