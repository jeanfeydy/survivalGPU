---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Cox Proportional Hazards

{class}`~survivalgpu.CoxPHSurvivalAnalysis` fits a standard Cox Proportional
Hazards model on `(start, stop]` intervals with fixed or time-varying
covariates. Its API is loosely based on scikit-learn/scikit-survival: build
the estimator, call `.fit()`, then read the results off the fitted object's
attributes (all of which end with an underscore).

## A simulated dataset

To keep this page self-contained, we simulate a small dataset with
{func}`~survivalgpu.simulate_dataset`: two covariates, `sex` (a 0/1 indicator)
and `biomarker` (three levels), with known "true" coefficients so that we can
check what the fit recovers.

```{code-cell} python3
import numpy as np
from survivalgpu import ConstantCovariate, simulate_dataset

TRUE_COEF = {"sex": 0.7, "biomarker": 0.4}

sex = ConstantCovariate(name="sex", values=[0, 1], weights=[1, 1], coef=TRUE_COEF["sex"])
biomarker = ConstantCovariate(
    name="biomarker", values=[0, 1, 2], weights=[1, 1, 1], coef=TRUE_COEF["biomarker"]
)

df = simulate_dataset(
    max_time=200, n_patients=300, list_covariates=[sex, biomarker], compress=True, seed=0
)

covariates = df[["sex", "biomarker"]].to_numpy(dtype=np.float64)
start = df["start"].to_numpy(dtype=np.int64)
stop = df["stop"].to_numpy(dtype=np.int64)
event = df["events"].to_numpy(dtype=np.int64)
patient = df["patients"].to_numpy(dtype=np.int64)
```

## Fitting the model

```{code-cell} python3
from survivalgpu import CoxPHSurvivalAnalysis

model = CoxPHSurvivalAnalysis(
    ties="efron",  # or "breslow"
    device="cpu",  # use device="cuda" for a GPU speedup if one is available,
    # or device=None to auto-select the best available device.
)

model.fit(covariates=covariates, stop=stop, start=start, event=event, patient=patient)

print("True coefficients:  ", list(TRUE_COEF.values()))
print("Fitted coefficients:", model.coef_[0])
print("Standard errors:    ", model.std_[0])
```

The fitted coefficients are close to the values used to simulate the data.
Results are stored as attributes on `model`: `coef_`, `std_`, `means_`,
`score_`, `loglik_`, `loglik_init_`, `sctest_init_`, `hessian_`, `imat_` and
`iter_` — see {class}`~survivalgpu.CoxPHSurvivalAnalysis` for the full list.

95% Wald confidence intervals on the hazard ratios follow directly from
`coef_`/`std_`:

```{code-cell} python3
hr = np.exp(model.coef_[0])
lower = np.exp(model.coef_[0] - 1.96 * model.std_[0])
upper = np.exp(model.coef_[0] + 1.96 * model.std_[0])
for name, hr_i, lo, hi in zip(TRUE_COEF, hr, lower, upper, strict=False):
    print(f"HR[{name}] = {hr_i:.3f}  (95% CI: {lo:.3f} - {hi:.3f})")
```

## Bootstrap resampling

Passing `nbootstraps` fits the main model once, then re-fits it on that many
patient-level resamples of the data (`patient` is required to keep intervals
from the same patient together across resamples). `batchsize` controls how
many resamples are processed at once on the device — lower it if you run out
of memory. This is also where a GPU pays off most, since all resamples are
fitted in parallel.

```{code-cell} python3
boot_model = CoxPHSurvivalAnalysis(
    ties="efron", device="cpu", nbootstraps=500, batchsize=100
)
boot_model.fit(covariates=covariates, stop=stop, start=start, event=event, patient=patient)

print("bootstrap_coef_ shape (nbootstraps, n_batch, n_covariates):", boot_model.bootstrap_coef_.shape)

boot_hr = np.exp(boot_model.bootstrap_coef_[:, 0, :])
lower, upper = np.quantile(boot_hr, [0.025, 0.975], axis=0)
for name, hr_i, lo, hi in zip(TRUE_COEF, np.exp(boot_model.coef_[0]), lower, upper, strict=False):
    print(f"HR[{name}] = {hr_i:.3f}  (bootstrap 95% CI: {lo:.3f} - {hi:.3f})")
```

The percentile bootstrap CIs above are often preferable to the Wald CIs when
the sample size is small or the likelihood isn't well approximated by a
quadratic near the optimum.

## Going further

Full API reference: {class}`~survivalgpu.CoxPHSurvivalAnalysis`.
