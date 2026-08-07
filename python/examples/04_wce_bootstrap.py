"""WCE, class API, WITH bootstrap resampling, CPU.

Just like `CoxPHSurvivalAnalysis`, `WCESurvivalAnalysis` accepts
`nbootstraps`/`batchsize`: the WCE model is fitted once on the full data and
`nbootstraps` times on patient-level resamples. The extra attributes
`model.bootstrap_WCE_coef_` and `model.bootstrap_risk_functions_` hold the
bootstrap distribution of the B-spline weights and of the risk function.

Conveniently, `model.HR()` automatically detects that bootstrapping was
requested and returns a 95% percentile confidence interval for the hazard
ratio alongside the point estimate.

Run with:
    python 05_wce_bootstrap.py
"""

import numpy as np
from common import WCE_TRUE_HR, wce_dataset
from survivalgpu import WCESurvivalAnalysis

data = wce_dataset()

model = WCESurvivalAnalysis(
    cutoff=180,
    nknots=1,
    order=3,
    constrained="right",
    device="cpu",  # runs everywhere; use device="cuda" for a GPU speedup
    # if a CUDA-capable GPU + toolkit are available. Bootstrap is the case
    # where the GPU shines most, since all resamples are fitted in parallel.
    nbootstraps=200,
    batchsize=50,
)

model.fit(
    dose=data["dose"],
    stop=data["stop"],
    start=data["start"],
    event=data["event"],
    patient=data["patient"],
    covariates=data["covariates"],
)

print("bootstrap_WCE_coef_ shape (nbootstraps, n_batch, n_atoms):")
print(" ", model.bootstrap_WCE_coef_.shape)
print("bootstrap_risk_functions_ shape (nbootstraps, n_batch, cutoff):")
print(" ", model.bootstrap_risk_functions_.shape)

exposed = np.zeros(180, dtype=np.int64)
exposed[:30] = 1
unexposed = np.zeros(180, dtype=np.int64)

# model.HR() returns "CI_lower"/"CI_upper" as soon as bootstrap was enabled:
hr = model.HR(exposed, unexposed, level=0.95)
print(f"\nHR(30 days of exposure vs. none) = {hr['HR']:.3f}")
print(f"95% bootstrap CI: [{hr['CI_lower']:.3f}, {hr['CI_upper']:.3f}]")
print(f"(dataset was simulated with a target HR of {WCE_TRUE_HR})")
