"""WCE (Weighted Cumulative Exposure), class API, no bootstrap, CPU.

`WCESurvivalAnalysis` models a time-varying drug exposure by combining it
with a bank of B-spline basis functions ("atoms") over a `cutoff`-long time
window, then fits a CoxPH model on top of the resulting features (plus any
extra `covariates`, such as "sex" here). The fitted risk function
(`model.risk_function_`) describes how much a dose received `u` days ago
still contributes to the current risk.

Run with:
    python 04_wce_basic.py
"""

import numpy as np
from common import WCE_TRUE_HR, wce_dataset
from survivalgpu import WCESurvivalAnalysis

data = wce_dataset()

model = WCESurvivalAnalysis(
    cutoff=180,  # size of the exposure time window (in days)
    nknots=1,  # number of interior knots for the B-splines
    order=3,  # cubic B-splines
    constrained="right",  # risk vanishes at the cutoff
    device="cpu",  # runs everywhere; use device="cuda" for a GPU speedup
    # if a CUDA-capable GPU + toolkit are available.
)

model.fit(
    dose=data["dose"],
    stop=data["stop"],
    start=data["start"],
    event=data["event"],
    patient=data["patient"],
    covariates=data["covariates"],  # the extra "sex" covariate
)

print("Coefficient for 'sex':", model.coef_[0, 0])
print("WCE B-spline weights: ", model.WCE_coef_[0])
print("risk_function_ shape (n_batch, cutoff):", model.risk_function_.shape)
print("Information criterion (BIC by default):", model.info_criterion_[0])

# The Hazard Ratio (HR) between two dose "profiles" over the cutoff window,
# e.g. a constant dose of 1 for the first 30 days vs. no exposure at all:
exposed = np.zeros(180, dtype=np.int64)
exposed[:30] = 1
unexposed = np.zeros(180, dtype=np.int64)

hr = model.HR(exposed, unexposed)
print(f"\nHR(30 days of exposure vs. none) = {hr['HR']:.3f}")
print(f"(dataset was simulated with a target HR of {WCE_TRUE_HR})")
