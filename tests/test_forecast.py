import numpy as np
import pandas as pd
import pytest

from kagrued.datasets import DATASETS, load_predictions, load_trace
from kagrued.forecast import error_metrics, make_windows, predict, prepare, train_model


def synthetic_series(n=1500, spike_at=None):
    idx = pd.date_range("2020-01-01", periods=n, freq="min")
    t = np.arange(n)
    values = 100 + 50 * np.sin(2 * np.pi * t / 60) + np.random.default_rng(0).normal(0, 2, n)
    if spike_at is not None:
        values[spike_at] = 1e6
    return pd.Series(values, index=idx)


def test_make_windows_alignment():
    f = np.arange(10, dtype=float)[:, None]
    X, Y = make_windows(f, np.arange(10, dtype=float), lookback=3, horizon=2)
    assert X.shape == (6, 3, 1) and Y.shape == (6, 2)
    assert X[0, :, 0].tolist() == [0, 1, 2] and Y[0].tolist() == [3, 4]
    assert X[-1, :, 0].tolist() == [5, 6, 7] and Y[-1].tolist() == [8, 9]


def test_make_windows_too_short():
    with pytest.raises(ValueError):
        make_windows(np.zeros((3, 1)), np.zeros(3), lookback=3, horizon=2)


def test_scaler_uses_training_data_only():
    clean = prepare(synthetic_series(), lookback=10, horizon=3)
    spiked = prepare(synthetic_series(spike_at=1400), lookback=10, horizon=3)  # spike in the test part
    assert spiked.scaler == clean.scaler


def test_splits_are_disjoint_and_ordered():
    s = prepare(synthetic_series(), lookback=10, horizon=3)
    assert len(s.X_train) > len(s.X_val) > 0 and len(s.X_test) > 0
    assert s.X_train.shape[1:] == (10, 5)
    assert s.Y_test.shape[1] == 3


def test_training_beats_naive_baseline():
    s = prepare(synthetic_series(), lookback=30, horizon=3)
    model = train_model(s, horizon=3, hidden=16, epochs=8, batch_size=64, lr=5e-3, patience=8, verbose=False)
    pred = predict(model, s.X_test)
    naive = np.repeat(s.X_test[:, -1, :1], 3, axis=1)  # repeat the last observed value
    assert np.mean((pred - s.Y_test) ** 2) < np.mean((naive - s.Y_test) ** 2)


def test_error_metrics_ignore_zero_targets_in_mape():
    m = error_metrics(np.array([1.0, 2.0, 5.0]), np.array([0.0, 2.0, 4.0]))
    assert m["MAPE"] == pytest.approx(12.5)
    assert m["MAE"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_bundled_data(name):
    ds = DATASETS[name]
    trace = load_trace(ds.trace_path)
    assert len(trace) > 80_000 and trace.index.is_monotonic_increasing
    for k in (3, 4, 5):
        preds = load_predictions(ds.predictions_path(k))
        assert {"Real", "K-AGRUED"} <= set(preds.columns)
