"""Trace-driven autoscaling simulation and elasticity metrics.

This is a cleaned-up, tested port of ``FIFA_dataset/Provisioning.py`` and
``NASA_dataset/provisioning.py``. With the default ``weighting="paper"`` it reproduces the numbers
those scripts print.

Policies (all scale down by removing ``ceil(0.6 * surplus)`` pods):

* ``static``    fixed number of pods (no autoscaling)
* ``proactive`` K-AGRUED policy: scale up to the forecast peak of the next 3 steps, then hold for
                3 steps; scale down to the forecast peak of the next 5 steps, then hold for 5 steps
* ``hpa``       reactive, HPA-like: same cooldowns, but driven by the load observed now
* ``previous``  proactive without look-ahead or cooldown: follows the forecast step by step

Usage:
    python -m kagrued.provisioning --dataset fifa --horizon 3 --forecast K-AGRUED
    python -m kagrued.provisioning --dataset nasa --weighting standard
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .datasets import get_dataset, load_predictions

DOWN_FRACTION = 0.6


def simulate_policy(
    signal: np.ndarray,
    capacity: float,
    up_lookahead: int,
    down_lookahead: int,
    up_cooldown: int,
    down_cooldown: int,
    down_fraction: float = DOWN_FRACTION,
) -> tuple[np.ndarray, int, int]:
    """Return (pods per step, number of scale-ups, number of scale-downs).

    At step ``j`` the policy compares the pods needed for the peak of ``signal[j:j+lookahead]``
    with the current pods. After a scale action it skips ``cooldown`` steps; the pod count is held
    constant over skipped steps.
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    pods_list = np.full(n, np.nan)
    pods = 0
    ups = downs = 0
    j = 0
    while j < n:
        need_up = math.ceil(signal[j : j + up_lookahead].max() / capacity)
        need_down = math.ceil(signal[j : j + down_lookahead].max() / capacity)
        if need_up > pods:
            pods = need_up
            ups += 1
            pods_list[j] = pods
            j += up_cooldown
        elif need_down < pods:
            pods -= math.ceil(down_fraction * (pods - need_down))
            downs += 1
            pods_list[j] = pods
            j += down_cooldown
        else:
            pods_list[j] = pods
            j += 1
    return pd.Series(pods_list).ffill().to_numpy(), ups, downs


def proactive_policy(forecast: np.ndarray, capacity: float) -> tuple[np.ndarray, int, int]:
    return simulate_policy(forecast, capacity, up_lookahead=3, down_lookahead=5, up_cooldown=3, down_cooldown=5)


def hpa_policy(observed: np.ndarray, capacity: float) -> tuple[np.ndarray, int, int]:
    return simulate_policy(observed, capacity, up_lookahead=1, down_lookahead=1, up_cooldown=3, down_cooldown=5)


def previous_study_policy(forecast: np.ndarray, capacity: float) -> tuple[np.ndarray, int, int]:
    return simulate_policy(forecast, capacity, up_lookahead=1, down_lookahead=1, up_cooldown=1, down_cooldown=1)


def _run_lengths(demand: np.ndarray) -> np.ndarray:
    """For each step, the length of the run of identical demand values ending at that step."""
    runs = np.ones(len(demand))
    for i in range(1, len(demand)):
        if demand[i] == demand[i - 1]:
            runs[i] = runs[i - 1] + 1
    return runs


def elasticity_metrics(supply: np.ndarray, demand: np.ndarray, weighting: str = "paper") -> dict[str, float]:
    """Under/over-provisioning accuracy (theta_U, theta_O) and timeshare (TU, TO), in percent.

    ``weighting="paper"`` reproduces the original scripts: each step is weighted by the length of
    the run of identical demand values ending there, and theta_O is normalised by supply. TU/TO can
    then exceed 100.

    ``weighting="standard"`` uses the SPEC RG elasticity definitions with uniform sampling: each step
    has weight 1 and both accuracies are normalised by demand, so TU + TO <= 100.
    """
    s = np.asarray(supply, dtype=float)
    d = np.asarray(demand, dtype=float)
    under = np.maximum(d - s, 0)
    over = np.maximum(s - d, 0)
    if weighting == "paper":
        w = _run_lengths(d)
        over_denominator = s
    elif weighting == "standard":
        w = np.ones_like(d)
        over_denominator = d
    else:
        raise ValueError("weighting must be 'paper' or 'standard'")
    with np.errstate(divide="ignore", invalid="ignore"):
        theta_u = np.where(d > 0, under / d, 0.0)
        theta_o = np.where(over_denominator > 0, over / over_denominator, 0.0)
    return {
        "theta_U": float(np.mean(theta_u * w) * 100),
        "theta_O": float(np.mean(theta_o * w) * 100),
        "TU": float(np.mean((d > s) * w) * 100),
        "TO": float(np.mean((s > d) * w) * 100),
    }


def elastic_speedup(baseline: dict[str, float], policy: dict[str, float]) -> float:
    """Geometric mean of baseline/policy over the four metrics (higher is better)."""
    keys = ["theta_U", "theta_O", "TU", "TO"]
    if any(policy[k] == 0 for k in keys):
        return float("inf")
    ratios = [baseline[k] / policy[k] for k in keys]
    return float(np.prod(ratios) ** (1 / len(keys)))


@dataclass
class PolicyResult:
    name: str
    metrics: dict[str, float]
    ups: int = 0
    downs: int = 0
    speedup: float = 1.0
    pods: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)


def evaluate(
    real: np.ndarray,
    forecast: np.ndarray,
    capacity: float,
    static_pods: int,
    weighting: str = "paper",
) -> list[PolicyResult]:
    """Run all four policies on one trace and return their metrics."""
    real = np.asarray(real, dtype=float)
    demand = np.ceil(real / capacity)
    static = np.full(len(real), float(static_pods))
    base = PolicyResult("static", elasticity_metrics(static, demand, weighting), pods=static)
    results = [base]
    for name, fn, signal in [
        ("proactive", proactive_policy, forecast),
        ("hpa", hpa_policy, real),
        ("previous", previous_study_policy, forecast),
    ]:
        pods, ups, downs = fn(signal, capacity)
        m = elasticity_metrics(pods, demand, weighting)
        results.append(PolicyResult(name, m, ups, downs, elastic_speedup(base.metrics, m), pods))
    return results


def results_table(results: list[PolicyResult]) -> pd.DataFrame:
    rows = [{"policy": r.name, **r.metrics, "speedup": r.speedup, "scale_ups": r.ups, "scale_downs": r.downs} for r in results]
    return pd.DataFrame(rows).set_index("policy")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="fifa", choices=["fifa", "nasa"])
    ap.add_argument("--horizon", type=int, default=3, choices=[3, 4, 5], help="which bundled t+k.csv to use")
    ap.add_argument("--predictions", default=None, help="CSV with a 'Real' column and forecast columns")
    ap.add_argument("--forecast", default="K-AGRUED", help="forecast column that drives the proactive policies")
    ap.add_argument("--weighting", default="paper", choices=["paper", "standard"])
    a = ap.parse_args(argv)

    ds = get_dataset(a.dataset)
    preds = load_predictions(a.predictions or ds.predictions_path(a.horizon))
    if a.forecast not in preds.columns:
        ap.error(f"column {a.forecast!r} not in predictions; available: {[c for c in preds.columns if c != 'Real']}")
    results = evaluate(preds["Real"].to_numpy(), preds[a.forecast].to_numpy(), ds.capacity, ds.static_pods, a.weighting)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print(f"dataset={ds.name} forecast={a.forecast} capacity={ds.capacity:g} req/pod/step weighting={a.weighting}")
    print(results_table(results).to_string())


if __name__ == "__main__":
    main()
