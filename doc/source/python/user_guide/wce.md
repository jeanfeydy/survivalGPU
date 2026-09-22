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
patient-level resamples. `model.HR()` automatically detects that bootstrapping
was requested and returns a percentile confidence interval for the hazard
ratio alongside the point estimate.

```{code-cell} python3
boot_model = WCESurvivalAnalysis(
    cutoff=180, nknots=1, order=3, constrained="right",
    device="cpu", nbootstraps=200, batchsize=50,
)
boot_model.fit(dose=dose_arr, stop=stop, start=start, event=event, patient=patient, covariates=covariates)

print("bootstrap_WCE_coef_ shape (nbootstraps, n_batch, n_atoms):", boot_model.bootstrap_WCE_coef_.shape)
print("bootstrap_risk_functions_ shape (nbootstraps, n_batch, cutoff):", boot_model.bootstrap_risk_functions_.shape)

hr = boot_model.HR(exposed, unexposed, level=0.95)
print(f"HR(30 days of exposure vs. none) = {hr['HR']:.3f}")
print(f"95% bootstrap CI: [{hr['CI_lower']:.3f}, {hr['CI_upper']:.3f}]")
```

## Going further

Full API reference: {class}`~survivalgpu.WCESurvivalAnalysis`.
