# K-AGRUED: Proactive Container Autoscaling for Web Applications in Kubernetes

[![CI](https://github.com/JavadDogani/K-AGRUED/actions/workflows/ci.yml/badge.svg)](https://github.com/JavadDogani/K-AGRUED/actions/workflows/ci.yml)
[![Journal of Grid Computing](https://img.shields.io/badge/J.%20Grid%20Computing-2022-blue)](https://doi.org/10.1007/s10723-022-09634-x)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)

Code and experiment artifacts for the paper

> **K-AGRUED: A Container Autoscaling Technique for Cloud-Based Web Applications in Kubernetes Using an Attention-Based GRU Encoder–Decoder**<br>
> Javad Dogani, Farshad Khunjush, Mehdi Seydali<br>
> *Journal of Grid Computing*, vol. 20, article 40, 2022 · [DOI: 10.1007/s10723-022-09634-x](https://doi.org/10.1007/s10723-022-09634-x)

## Idea

Reactive autoscalers such as the Kubernetes Horizontal Pod Autoscaler (HPA) add pods only after
load has already risen, so new containers start too late. K-AGRUED is **proactive**: an
attention-based GRU encoder–decoder predicts the request rate several steps ahead, and the scaler
provisions pods for the predicted peak. Scale-up is immediate; scale-down is gradual, removing only
part of the surplus at a time to avoid oscillation.

The approach is evaluated on two real web-server traces. Prediction is compared with ARIMA, SVR,
LSTM and a plain GRU encoder–decoder (GRUED). Provisioning is compared with HPA, a previous proactive
approach and a fixed-size deployment.

## What is in this repository

1. **`kagrued/`, an installable and tested Python package:**
   - `kagrued.provisioning`: the trace-driven autoscaling simulator and elasticity metrics. It
     reproduces the numbers printed by the original paper scripts (checked in the test suite).
   - `kagrued.model` / `kagrued.forecast`: a PyTorch **re-implementation** of the attention-based
     GRU encoder–decoder, with leakage-free data preparation, training and evaluation. See
     [Forecasting](#forecasting-model) for how it relates to the paper.
2. **The original paper artifacts** in `FIFA_dataset/` and `NASA_dataset/`: the traces, the
   paper's predictions of every model, the original analysis scripts and the figures.

## Quickstart

```bash
pip install -e .                       # numpy, pandas, torch
python -m kagrued.provisioning --dataset fifa
```

```text
dataset=fifa forecast=K-AGRUED capacity=500 req/pod/step weighting=paper
           theta_U  theta_O     TU      TO  speedup  scale_ups  scale_downs
policy
static       0.573  317.218  4.371 442.941    1.000          0            0
proactive    0.481   14.420  2.613  65.005    4.157        804          813
hpa          6.062    2.631 25.613  12.608    2.875       1533         1567
previous     2.522    2.984 12.478  14.581    4.004       2276         2289
```

Other options: `--dataset nasa`, `--horizon 3|4|5` (which bundled prediction file to use),
`--forecast LSTM|GRUED|SVR|ARIMA|K-AGRUED` (which model's predictions drive the proactive policies),
`--predictions my.csv` (your own predictions) and `--weighting standard` (see below).

To train the forecasting model:

```bash
python -m kagrued.forecast --dataset fifa    # writes results/fifa/t+{1..5}.csv, metrics.csv, model.pt
python -m kagrued.provisioning --dataset fifa --predictions results/fifa/t+3.csv --forecast AttnGRUED
```

## Provisioning simulation

Each policy turns a load signal into a pod count, with a fixed per-pod capacity (500 requests per
minute for FIFA, 10 for NASA). All of them scale down by removing 60% of the surplus pods at a time.

| Policy | Driven by | Scale-up rule | Scale-down rule |
|---|---|---|---|
| `static` | — | fixed pods (13 for FIFA, 4 for NASA) | — |
| `proactive` (K-AGRUED) | forecast | to the peak of the next 3 steps, then hold 3 steps | to the peak of the next 5 steps, then hold 5 steps |
| `hpa` (reactive) | observed load | to the current load, then hold 3 steps | to the current load, then hold 5 steps |
| `previous` | forecast | to the current forecast, every step | to the current forecast, every step |

Every policy is scored against the pods actually needed, `ceil(load / capacity)`:

- **θU / θO:** under- and over-provisioning accuracy, in percent.
- **TU / TO:** timeshare spent under- or over-provisioned.
- **speedup (ε):** geometric mean of `static / policy` over these four metrics; higher is better.

**Metric weighting.** `--weighting paper` (the default) computes the metrics exactly as the paper
scripts do. Each step is weighted by the length of the run of identical demand values ending at
that step, and θO is normalised by supply. TU and TO can then exceed 100.
`--weighting standard` uses the SPEC RG elasticity definitions instead: every step has weight 1 and
both accuracies are normalised by demand. The ranking on FIFA is the same under both:

```text
python -m kagrued.provisioning --dataset fifa --weighting standard
           theta_U  theta_O     TU     TO  speedup  scale_ups  scale_downs
policy
static       0.339  254.453  2.554 96.183    1.000          0            0
proactive    0.474    7.706  2.581 26.715    3.027        804          813
hpa          3.480    2.692 15.220  9.430    1.992       1533         1567
previous     2.447    3.822 12.167 13.355    1.932       2276         2289
```

## Forecasting model

`kagrued/model.py` implements the model family described in the paper. A GRU encoder reads the last
`lookback` minutes of load plus cyclical hour-of-day and day-of-week features. A GRU decoder then
predicts the next `horizon` minutes one step at a time. At each step it attends over all encoder
states with additive (Bahdanau) attention, predicts the change from its previous value, and feeds
that prediction back in.

This is a **re-implementation from the paper's description, not the code that produced the
paper's numbers.** Layer sizes and training settings are defaults chosen here: hidden size 64,
lookback 30, horizon 5, Adam, MSE loss, up to 30 epochs with early stopping.

Data handling: chronological 70/30 train/test split, with the last 10% of the training part held
out for early stopping. Values are scaled as `log1p(load)` followed by min-max scaling, both fitted
on the training part only. Without the log, the World Cup peaks (up to 225k requests per minute)
squeeze normal load into a tiny part of the range.

**How it compares with the paper.** The bundled `t+k.csv` predictions cover 14–26 July 1998 (FIFA)
and 18–30 August 1995 (NASA), and both periods lie inside this split's test part. On those same
minutes, at t+3, the re-implementation reaches a MAPE of 12.6% on FIFA and 44.4% on NASA. The
paper's K-AGRUED predictions reach 7.7% and 37.5%. To reproduce the paper's provisioning results,
use the bundled predictions, as in the Quickstart.

## Original paper scripts

Each dataset folder has the same layout:

| Path | Contents |
|---|---|
| `fifa98.csv` / `NASA_Results.csv` | The trace, aggregated to requests per minute |
| `t+3.csv`, `t+4.csv`, `t+5.csv` | Actual load (`Real`) and the predictions of K-AGRUED and the baselines, 3, 4 and 5 minutes ahead |
| `Prediction_Model.py` | The LSTM baseline on calendar and lagged-load features (Keras; needs `pip install -e ".[paper]"`) |
| `Provisioning.py` | The original provisioning simulation (superseded by `kagrued.provisioning`) |
| `Chart_*.py`, `*.svg` | Figure scripts and the figures |

The scripts read their inputs from the current directory, so run them from inside a dataset
folder, e.g. `cd FIFA_dataset && python Provisioning.py`.

Fixes made since publication:

- `Prediction_Model.py` fitted separate scalers on the test split; it now reuses the training scalers.
- The provisioning scripts now drive the proactive policy with the `K-AGRUED` predictions (they had
  been left on `LSTM` for FIFA and `GRUED` for NASA).
- `NASA_dataset/Chart_dataset Show-NASA.py` read a file that does not exist.

The NASA provisioning script has two known quirks: at steps with zero demand its accuracy metrics
reuse the previous step's value, and it prints the HPA speedup in the "previous study" row.
`kagrued.provisioning` does not have them, so its NASA HPA θO (11.27) differs from the script's (11.48).

## Datasets

Both traces come from the [Internet Traffic Archive](https://ita.ee.lbl.gov/html/traces.html):

- **FIFA World Cup 1998:** about 1.3 billion requests to the World Cup web site, 30 April – 26 July 1998.
- **NASA-HTTP:** about 3.46 million requests to the NASA Kennedy Space Center web server,
  July and August 1995. No requests were logged from 1 Aug 14:52 to 3 Aug 04:36 because the server
  was shut down during Hurricane Erin.

## Testing

```bash
pip install -e ".[dev]"
pytest -q          # ~10 s on CPU
ruff check .
```

The tests check that the simulator reproduces the original FIFA script's output, the policy and
metric behaviour, the model and windowing code, and that the scalers never see test data.

## Citation

```bibtex
@article{dogani2022kagrued,
  title   = {K-AGRUED: A Container Autoscaling Technique for Cloud-Based Web Applications in Kubernetes Using Attention-Based GRU Encoder-Decoder},
  author  = {Dogani, Javad and Khunjush, Farshad and Seydali, Mehdi},
  journal = {Journal of Grid Computing},
  volume  = {20},
  pages   = {40},
  year    = {2022},
  doi     = {10.1007/s10723-022-09634-x}
}
```

## License

MIT. See [LICENSE](LICENSE).
