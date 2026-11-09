"""Train and evaluate the attention-GRU encoder-decoder on a trace.

Data handling: chronological 70/30 train/test split (as in the paper scripts), the last 10% of the
training part held out for early stopping, and min-max scaling fitted on the training part only.

Usage:
    python -m kagrued.forecast --dataset fifa                    # writes results/fifa/t+{3,4,5}.csv
    python -m kagrued.forecast --dataset nasa --epochs 30 --lookback 30
    python -m kagrued.provisioning --dataset fifa --predictions results/fifa/t+3.csv --forecast AttnGRUED
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .datasets import REPO_ROOT, get_dataset, load_trace
from .model import AttentionGRUEncoderDecoder

FORECAST_COLUMN = "AttnGRUED"


def calendar_features(index: pd.DatetimeIndex) -> np.ndarray:
    """Cyclical hour-of-day and day-of-week encodings, shape (T, 4)."""
    hour = index.hour + index.minute / 60.0
    dow = index.dayofweek
    return np.stack(
        [np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24), np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)],
        axis=1,
    )


@dataclass
class MinMax:
    lo: float
    hi: float

    @classmethod
    def fit(cls, x: np.ndarray) -> MinMax:
        lo, hi = float(np.min(x)), float(np.max(x))
        return cls(lo, hi if hi > lo else lo + 1.0)

    def transform(self, x):
        return (x - self.lo) / (self.hi - self.lo)

    def inverse(self, x):
        return x * (self.hi - self.lo) + self.lo


def make_windows(features: np.ndarray, target: np.ndarray, lookback: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """X[i] = features[i : i+lookback], Y[i] = target[i+lookback : i+lookback+horizon]."""
    n = len(target) - lookback - horizon + 1
    if n <= 0:
        raise ValueError("series too short for the requested lookback and horizon")
    idx = np.arange(lookback)[None, :] + np.arange(n)[:, None]
    X = features[idx]
    Y = target[np.arange(horizon)[None, :] + np.arange(n)[:, None] + lookback]
    return X.astype(np.float32), Y.astype(np.float32)


@dataclass
class Split:
    X_train: np.ndarray
    Y_train: np.ndarray
    X_val: np.ndarray
    Y_val: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray
    scaler: MinMax


def prepare(series: pd.Series, lookback: int, horizon: int, train_frac: float = 0.7, val_frac: float = 0.1) -> Split:
    """Scale with training statistics only, then build windows for each chronological split."""
    values = series.to_numpy(dtype=float)
    cut = int(train_frac * len(values))
    scaler = MinMax.fit(values[:cut])
    scaled = scaler.transform(values)
    feats = np.concatenate([scaled[:, None], calendar_features(pd.DatetimeIndex(series.index))], axis=1)

    X, Y = make_windows(feats, scaled, lookback, horizon)
    # a window belongs to a split if its last target lies inside that split
    target_end = np.arange(len(Y)) + lookback + horizon - 1
    train = target_end < cut
    test = target_end >= cut + lookback + horizon - 1  # no input overlap with the training part
    n_val = int(val_frac * train.sum())
    train_idx = np.flatnonzero(train)
    val_idx, train_idx = train_idx[len(train_idx) - n_val :], train_idx[: len(train_idx) - n_val]
    return Split(X[train_idx], Y[train_idx], X[val_idx], Y[val_idx], X[test], Y[test], scaler)


def train_model(
    split: Split,
    horizon: int,
    hidden: int = 64,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 3,
    seed: int = 0,
    verbose: bool = True,
) -> AttentionGRUEncoderDecoder:
    torch.manual_seed(seed)
    model = AttentionGRUEncoderDecoder(split.X_train.shape[2], horizon, hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(split.X_train), torch.from_numpy(split.Y_train)),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    X_val, Y_val = torch.from_numpy(split.X_val), torch.from_numpy(split.Y_val)
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(X_val), Y_val).item() if len(X_val) else loss.item()
        if verbose:
            print(f"epoch {epoch:3d}  train_mse {loss.item():.5f}  val_mse {val:.5f}  ({time.time() - t0:.1f}s)")
        if val < best - 1e-7:
            best, best_state, bad = val, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model: AttentionGRUEncoderDecoder, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(X[i : i + batch_size])).numpy() for i in range(0, len(X), batch_size)])


def error_metrics(pred: np.ndarray, real: np.ndarray) -> dict[str, float]:
    """MAPE over non-zero targets (%), RMSE and MAE."""
    nz = real != 0
    return {
        "MAPE": float(np.mean(np.abs(pred[nz] - real[nz]) / real[nz]) * 100),
        "RMSE": float(np.sqrt(np.mean((pred - real) ** 2))),
        "MAE": float(np.mean(np.abs(pred - real))),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="fifa", choices=["fifa", "nasa"])
    ap.add_argument("--lookback", type=int, default=30, help="past minutes fed to the encoder")
    ap.add_argument("--horizon", type=int, default=5, help="future minutes predicted by the decoder")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None, help="default: results/<dataset>")
    a = ap.parse_args(argv)

    ds = get_dataset(a.dataset)
    series = load_trace(ds.trace_path)
    split = prepare(series, a.lookback, a.horizon)
    print(f"{ds.name}: {len(series)} minutes; windows train={len(split.X_train)} val={len(split.X_val)} test={len(split.X_test)}")
    model = train_model(split, a.horizon, hidden=a.hidden, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, seed=a.seed)

    pred = np.maximum(split.scaler.inverse(predict(model, split.X_test)), 0)
    real = np.rint(split.scaler.inverse(split.Y_test))
    out_dir = a.out_dir or REPO_ROOT / "results" / ds.name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in range(1, a.horizon + 1):
        rows.append({"step": f"t+{k}", **error_metrics(pred[:, k - 1], real[:, k - 1])})
        pd.DataFrame({"Real": real[:, k - 1], FORECAST_COLUMN: pred[:, k - 1]}).to_csv(out_dir / f"t+{k}.csv", index=False)
    table = pd.DataFrame(rows).set_index("step")
    table.to_csv(out_dir / "metrics.csv")
    torch.save(model.state_dict(), out_dir / "model.pt")
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print(table.to_string())
    print(f"predictions, metrics and weights written to {out_dir}")


if __name__ == "__main__":
    main()
