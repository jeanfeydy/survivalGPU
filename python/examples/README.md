# survivalGPU examples

Small, self-contained scripts showing how to use survivalGPU's two models —
**CoxPH** and **WCE** — with and without bootstrap resampling.

All examples fit on synthetic data (via `common.py`, which uses
survivalGPU's own simulator so the "true" coefficients / hazard ratio are
known) and run on **CPU by default**, so they work on any machine. Each
script has a comment showing how to switch to `device="cuda"` to run on a
GPU if one is available.

| Script | Model | Bootstrap |
| --- | --- | --- |
| [`01_coxph_basic.py`](01_coxph_basic.py) | CoxPH | No |
| [`02_coxph_bootstrap.py`](02_coxph_bootstrap.py) | CoxPH | Yes |
| [`03_wce_basic.py`](03_wce_basic.py) | WCE | No |
| [`04_wce_bootstrap.py`](04_wce_bootstrap.py) | WCE | Yes |

## Running

From this directory, with survivalgpu installed (`pip install -e ..` from
`python/`, or `pip install survivalgpu`):

```bash
python 01_coxph_basic.py
python 02_coxph_bootstrap.py
python 03_wce_basic.py
python 04_wce_bootstrap.py
```

## Notes

- **CoxPH** (`CoxPHSurvivalAnalysis`) fits a standard Cox Proportional
  Hazards model on `(start, stop]` intervals with fixed or time-varying
  covariates. Results (`coef_`, `std_`, `loglik_`, ...) are stored as
  attributes on the fitted object.
- **WCE** (`WCESurvivalAnalysis`) models a time-varying drug exposure with
  B-spline basis functions over a `cutoff`-long window, then fits a CoxPH
  model on the resulting features (plus any extra covariates). Use
  `model.HR(dose_a, dose_b)` to compare two exposure profiles.
- Passing `nbootstraps` (and optionally `batchsize`) to either model fits it
  once on the full data, then again on that many patient-level resamples,
  giving `bootstrap_coef_` / `bootstrap_WCE_coef_` / `bootstrap_risk_functions_`
  attributes that can be used for percentile confidence intervals — this is
  also the case where using a GPU pays off most, since all resamples are
  fitted in parallel.
