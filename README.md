# K-AGRUED: Proactive Container Autoscaling for Web Applications in Kubernetes

[![Journal of Grid Computing](https://img.shields.io/badge/J.%20Grid%20Computing-2022-blue)](https://doi.org/10.1007/s10723-022-09634-x)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Experiment artifacts for the paper

> **K-AGRUED: A Container Autoscaling Technique for Cloud-Based Web Applications in Kubernetes Using an Attention-Based GRU Encoder–Decoder**<br>
> Javad Dogani, Farshad Khunjush, Mehdi Seydali<br>
> *Journal of Grid Computing*, vol. 20, article 40, 2022 · [DOI: 10.1007/s10723-022-09634-x](https://doi.org/10.1007/s10723-022-09634-x)

## Idea

Reactive autoscalers such as the Kubernetes Horizontal Pod Autoscaler (HPA) add pods only after
load has already risen, so new containers start too late. K-AGRUED is **proactive**: an
attention-based GRU encoder–decoder predicts the request rate several steps ahead, and the scaler
provisions pods for the predicted peak. Scale-up is immediate; scale-down is gradual, removing only
part of the surplus at a time to avoid oscillation.

The approach is evaluated on two real web-server traces and compared with ARIMA, SVR, LSTM and a
plain GRU encoder–decoder (GRUED) for prediction, and with HPA, a previous proactive approach and a
fixed-size deployment for provisioning.

## Repository contents

The two folders have the same layout, one per workload trace:

| Path | Contents |
|---|---|
| `FIFA_dataset/fifa98.csv` | FIFA World Cup 1998 web-server trace, aggregated to requests per minute |
| `NASA_dataset/NASA_Results.csv` | NASA Kennedy Space Center web-server trace (Jul–Aug 1995), requests per minute |
| `*/data description.txt` | Short description of each trace |
| `*/t+3.csv`, `t+4.csv`, `t+5.csv` | Actual load (`Real`) and the predictions of K-AGRUED and the baselines, 3, 4 and 5 steps ahead |
| `*/Prediction_Model.py` | Feature engineering (calendar features + lagged load), 70/30 train/test split, and a Keras recurrent forecaster reporting MAPE, RMSE and MAE |
| `*/Provisioning.py` | Trace-driven autoscaling simulation that turns a forecast into a pod count over time |
| `*/Chart_*.py` | Scripts that regenerate the figures (prediction vs. actual, absolute-error box plots, baseline comparison) |
| `*/*.svg` | The generated figures |

### Provisioning simulation

`Provisioning.py` converts load into pods with a fixed per-pod capacity (`processrate = 500` requests
per step) and compares four policies on the same trace:

- **No autoscaling:** a fixed number of pods.
- **Proactive (K-AGRUED policy):** scales up to cover the predicted maximum of the next 3 steps;
  scales down using the next 5 steps, removing 60% of the surplus.
- **HPA-style reactive:** scales on the current observed load.
- **Previous study:** a proactive policy without the look-ahead window.

Each policy is scored with standard elasticity metrics:

- **θU / θO:** average under- and over-provisioning, as a percentage of demand.
- **TU / TO:** share of time spent under- or over-provisioned.
- **ε (elastic speedup):** improvement over the no-autoscaling baseline, computed as the
  geometric mean of the four metric ratios.

The script also counts scale-up and scale-down actions. Which forecast column drives the proactive
policy is set by the `lbl` list near the bottom of the script.

## Running

The scripts read their input files from the current directory, so run them from inside a dataset
folder:

```bash
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy scipy scikit-learn matplotlib tensorflow   # tensorflow provides keras

cd FIFA_dataset
python Provisioning.py                  # provisioning metrics for each policy
python Chart_Compare_baseline_methods.py
python Chart_AbsoluteError.py
python Prediction_Model.py              # trains the forecaster (slow: 200 epochs, batch size 1)
```

## Datasets

Both traces come from the [Internet Traffic Archive](https://ita.ee.lbl.gov/html/traces.html):

- **FIFA World Cup 1998:** about 1.3 billion requests to the World Cup web site, 30 April – 26 July 1998.
- **NASA-HTTP:** about 3.46 million requests to the NASA Kennedy Space Center web server,
  July and August 1995. No requests were logged from 1 Aug 14:52 to 3 Aug 04:36 because the server
  was shut down during Hurricane Erin.

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
