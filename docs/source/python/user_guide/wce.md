---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Weighted Cumulative Exposure (WCE)

{class}`~survivalgpu.WCESurvivalAnalysis` models a time-varying drug exposure
by combining it with a bank of B-spline basis functions ("atoms") over a
`cutoff`-long time window, then fits a Cox model on top of the resulting
features (plus any extra covariates). The fitted risk function
(`model.risk_function_`) describes how much a dose received `u` days ago
still contributes to the current risk.

## A simulated dataset

We simulate a single time-varying `dose` covariate that follows a
"bi_linear_scenario" risk shape (risk decreases linearly with time-since-dose
over the `cutoff` window) with a known target hazard ratio, plus a constant
`sex` covariate to illustrate mixing WCE and standard covariates in one model.
Unlike the CoxPH dataset, this one is *not* compressed, since
`WCESurvivalAnalysis` currently requires unit-length `(start, start+1]`
intervals.

```{code-cell} python3
import numpy as np
from survivalgpu import ConstantCovariate, WCECovariate, simulate_dataset

WCE_TRUE_HR = 3.0

dose = WCECovariate(
    name="dose", values=[1, 1.5, 2, 2.5, 3],
    scenario_name="bi_linear_scenario", HR_target=WCE_TRUE_HR,
)
sex = ConstantCovariate(name="sex", values=[0, 1], weights=[1, 1], coef=0.5)

df = simulate_dataset(
    max_time=200, n_patients=300, list_covariates=[dose, sex], compress=False, seed=1
)

dose_arr = df["dose"].to_numpy(dtype=np.float64)
covariates = df[["sex"]].to_numpy(dtype=np.float64)
start = df["start"].to_numpy(dtype=np.int64)
stop = df["stop"].to_numpy(dtype=np.int64)
event = df["events"].to_numpy(dtype=np.int64)
patient = df["patients"].to_numpy(dtype=np.int64)
```

## Fitting the model

```{code-cell} python3
from survivalgpu import WCESurvivalAnalysis

model = WCESurvivalAnalysis(
    cutoff=180,      # size of the exposure time window (in days)
    nknots=1,        # number of interior knots for the B-splines
    order=3,         # cubic B-splines
    constrained="right",  # risk vanishes at the cutoff
    device="cpu",    # use device="cuda" for a GPU speedup if available
)

model.fit(dose=dose_arr, stop=stop, start=start, event=event, patient=patient, covariates=covariates)

print("Coefficient for 'sex':", model.coef_[0, 0])
print("risk_function_ shape (n_batch, cutoff):", model.risk_function_.shape)
print("Information criterion (BIC by default):", model.info_criterion_[0])
```

Results are stored as attributes on `model`: `knots_`, `coef_`,
`WCE_coef_`, `risk_function_`, `std_`, `SED_`, `means_`, `score_`,
`loglik_`, `n_events_`, `info_criterion_` and more — see
{class}`~survivalgpu.WCESurvivalAnalysis` for the full list.

## Choosing the number of knots

The number of interior knots is usually not known in advance. Instead of a
single value, `nknots` accepts a list of candidates: one model is fitted per
candidate, and the one that minimizes the information criterion is selected
(`criterion="bic"` by default, or `criterion="aic"`). All the flat attributes
(`coef_`, `risk_function_`, `loglik_`, ...) then describe that selected
model, while every candidate's results are kept in parallel `*_grid_`
attributes.

```{code-cell} python3
model = WCESurvivalAnalysis(
    cutoff=180, nknots=[1, 2, 3], order=3, constrained="right", device="cpu",
)
model.fit(dose=dose_arr, stop=stop, start=start, event=event, patient=patient, covariates=covariates)

for nknots, loglik, bic in zip(
    model.nknots_candidates, model.loglik_grid_[:, 0], model.info_criterion_grid_[:, 0]
):
    print(f"nknots={nknots}: loglik={loglik:.2f}, BIC={bic:.2f}")
print("Selected number of knots:", model.best_nknots_[0])
```

## Comparing exposure profiles

`model.HR()` computes the hazard ratio between two dose "profiles" over the
`cutoff` window — for example, a constant dose of 1 for the first 30 days
versus no exposure at all:

```{code-cell} python3
exposed = np.zeros(180, dtype=np.int64)
exposed[:30] = 1
unexposed = np.zeros(180, dtype=np.int64)

hr = model.HR(exposed, unexposed)
print(f"HR(30 days of exposure vs. none) = {hr['HR']:.3f}")
print(f"(dataset was simulated with a target HR of {WCE_TRUE_HR})")
```

## Bootstrap resampling

Just like {class}`~survivalgpu.CoxPHSurvivalAnalysis`,
{class}`~survivalgpu.WCESurvivalAnalysis` accepts `nbootstraps`/`batchsize`:
the model is fitted once on the full data and `nbootstraps` times on
patient-level resamples.

When `nknots` is a list of candidates, **the number of knots is selected
again in every bootstrap replicate**:

1. the resamples are drawn once and shared by all the candidates, so that
   they are compared on exactly the same data;
2. every candidate is fitted on every resample;
3. each replicate keeps the candidate that minimizes the information
   criterion computed on its own resampled data (for the BIC, with that
   replicate's own number of events).

The bootstrap distribution therefore reflects the uncertainty on the choice
of the number of knots, and not only the uncertainty on the coefficients for
a fixed, pre-selected number of knots.

```{code-cell} python3
boot_model = WCESurvivalAnalysis(
    cutoff=180, nknots=[1, 2, 3], order=3, constrained="right",
    device="cpu", nbootstraps=100, batchsize=50,
)
boot_model.fit(dose=dose_arr, stop=stop, start=start, event=event, patient=patient, covariates=covariates)

print("Selected number of knots on the full data:", boot_model.best_nknots_[0])
values, counts = np.unique(boot_model.bootstrap_best_nknots_, return_counts=True)
for nknots, count in zip(values, counts):
    print(f"nknots={nknots} selected in {count} / {boot_model.nbootstraps} replicates")
```

`bootstrap_coef_` (covariates) and `bootstrap_risk_functions_` hold, for
every replicate, the results of the candidate selected for that replicate.
The results of every candidate on every replicate are kept in
`bootstrap_coef_grid_`, `bootstrap_WCE_coef_grid_`,
`bootstrap_risk_functions_grid_`, `bootstrap_loglik_grid_` and
`bootstrap_info_criterion_grid_`. Since different replicates may select
spline bases of different sizes, `bootstrap_WCE_coef_` is only available when
a single `nknots` value is given.

`model.HR()` automatically detects that bootstrapping was requested and
returns a percentile confidence interval for the hazard ratio alongside the
point estimate, using each replicate's selected risk function:

```{code-cell} python3
print("bootstrap_risk_functions_ shape (nbootstraps, n_batch, cutoff):", boot_model.bootstrap_risk_functions_.shape)

hr = boot_model.HR(exposed, unexposed, level=0.95)
print(f"HR(30 days of exposure vs. none) = {hr['HR']:.3f}")
print(f"95% bootstrap CI: [{hr['CI_lower']:.3f}, {hr['CI_upper']:.3f}]")
```

With a single value such as `nknots=1`, there is nothing to select: every
replicate uses that number of knots, as in a standard bootstrap.

## Going further

Full API reference: {class}`~survivalgpu.WCESurvivalAnalysis`.
