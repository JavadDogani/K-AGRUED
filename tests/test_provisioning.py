import numpy as np
import pytest

from kagrued.datasets import get_dataset, load_predictions
from kagrued.provisioning import (
    _run_lengths,
    elasticity_metrics,
    evaluate,
    hpa_policy,
    proactive_policy,
    results_table,
    simulate_policy,
)

# Printed by FIFA_dataset/Provisioning.py with the K-AGRUED forecast on t+3.csv.
FIFA_REFERENCE = {
    "static": (0.573040705547087, 317.2175351530245, 4.370967741935484, 442.9408602150538, 1.0, 0, 0),
    "proactive": (0.4810630006263539, 14.41952776031942, 2.612903225806452, 65.00537634408602, 4.15728791167521, 804, 813),
    "hpa": (6.062189294412595, 2.6305780459758537, 25.61290322580645, 12.60752688172043, None, 1533, 1567),
    "previous": (2.521947904027223, 2.984290042996958, 12.478494623655914, 14.580645161290324, 4.003939270775179, 2276, 2289),
}


def test_reproduces_original_fifa_script():
    ds = get_dataset("fifa")
    preds = load_predictions(ds.predictions_path(3))
    results = {r.name: r for r in evaluate(preds["Real"], preds["K-AGRUED"], ds.capacity, ds.static_pods)}
    for name, (tu, to, tU, tO, eps, ups, downs) in FIFA_REFERENCE.items():
        r = results[name]
        assert r.metrics["theta_U"] == pytest.approx(tu, rel=1e-9)
        assert r.metrics["theta_O"] == pytest.approx(to, rel=1e-9)
        assert r.metrics["TU"] == pytest.approx(tU, rel=1e-9)
        assert r.metrics["TO"] == pytest.approx(tO, rel=1e-9)
        if eps is not None:
            assert r.speedup == pytest.approx(eps, rel=1e-9)
        assert (r.ups, r.downs) == (ups, downs)


def test_proactive_scales_before_the_spike():
    load = np.array([100.0] * 10 + [1000.0] * 10)
    pods, ups, _ = proactive_policy(load, capacity=100)
    spike = 10
    assert pods[spike - 1] == 10  # already provisioned one step ahead (3-step look-ahead)
    hpa_pods, _, _ = hpa_policy(load, capacity=100)
    assert hpa_pods[spike - 1] == 1  # the reactive policy is not
    assert ups >= 2


def test_scale_down_is_gradual():
    load = np.array([1000.0] * 5 + [100.0] * 30)
    pods, _, downs = simulate_policy(load, 100, 1, 1, 1, 1)
    assert pods[4] == 10
    assert pods[5] == 10 - np.ceil(0.6 * 9)  # removes 60% of the surplus first
    assert pods[-1] == 1
    assert downs >= 2
    assert np.all(pods >= 1)


def test_metrics_zero_for_perfect_supply():
    demand = np.array([1, 2, 3, 3, 2], dtype=float)
    for weighting in ["paper", "standard"]:
        m = elasticity_metrics(demand, demand, weighting)
        assert all(v == 0 for v in m.values())


def test_standard_timeshares_are_percentages():
    rng = np.random.default_rng(0)
    demand = rng.integers(1, 10, 500).astype(float)
    supply = rng.integers(1, 10, 500).astype(float)
    m = elasticity_metrics(supply, demand, "standard")
    assert 0 <= m["TU"] + m["TO"] <= 100


def test_run_lengths():
    assert _run_lengths(np.array([1, 1, 2, 2, 2, 1])).tolist() == [1, 2, 1, 2, 3, 1]


def test_zero_demand_is_handled():
    demand = np.array([0, 0, 1, 2], dtype=float)
    supply = np.array([1, 0, 1, 1], dtype=float)
    for weighting in ["paper", "standard"]:
        m = elasticity_metrics(supply, demand, weighting)
        assert all(np.isfinite(v) for v in m.values())


def test_results_table_columns():
    real = np.linspace(100, 1000, 50)
    table = results_table(evaluate(real, real, capacity=100, static_pods=5))
    assert list(table.index) == ["static", "proactive", "hpa", "previous"]
    assert {"theta_U", "theta_O", "TU", "TO", "speedup"} <= set(table.columns)


def test_unknown_weighting():
    with pytest.raises(ValueError):
        elasticity_metrics(np.ones(3), np.ones(3), "other")
