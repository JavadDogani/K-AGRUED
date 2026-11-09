"""The two web-server traces used in the paper and their provisioning settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Dataset:
    name: str
    folder: Path
    trace_file: str
    capacity: float  # requests one pod serves per time step
    static_pods: int  # pod count of the no-autoscaling baseline

    @property
    def trace_path(self) -> Path:
        return self.folder / self.trace_file

    def predictions_path(self, horizon: int) -> Path:
        """Bundled paper predictions for ``horizon`` steps ahead (3, 4 or 5)."""
        return self.folder / f"t+{horizon}.csv"


DATASETS: dict[str, Dataset] = {
    "fifa": Dataset("fifa", REPO_ROOT / "FIFA_dataset", "fifa98.csv", capacity=500.0, static_pods=13),
    "nasa": Dataset("nasa", REPO_ROOT / "NASA_dataset", "NASA_Results.csv", capacity=10.0, static_pods=4),
}


def get_dataset(name: str) -> Dataset:
    try:
        return DATASETS[name.lower()]
    except KeyError:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}") from None


def load_trace(path: Path | str) -> pd.Series:
    """Per-minute request counts as a float Series indexed by timestamp."""
    df = pd.read_csv(path)
    try:
        index = pd.to_datetime(df["period"], format="%m/%d/%Y %H:%M")  # format of the bundled traces
    except ValueError:
        index = pd.to_datetime(df["period"], format="mixed")
    return pd.Series(df["count"].to_numpy(dtype=float), index=index, name="count")


def load_predictions(path: Path | str) -> pd.DataFrame:
    """A ``t+k.csv`` file: column ``Real`` plus one column per forecaster."""
    return pd.read_csv(path)
