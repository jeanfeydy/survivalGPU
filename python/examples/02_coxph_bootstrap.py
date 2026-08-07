"""CoxPH, class API, WITH bootstrap resampling, CPU.

Passing `nbootstraps` to `CoxPHSurvivalAnalysis` fits the main model once
(`model.coef_`, ...) and additionally re-fits it on `nbootstraps` patient-level
resamples of the data. `batchsize` controls how many of these resamples are
processed at once on the device: lower it if you run out of memory.

The extra `model.bootstrap_coef_` attribute, of shape
(nbootstraps, n_batch, n_covariates), gives an empirical distribution of the
coefficients that can be used to build percentile confidence intervals —
often preferable to the Wald CIs from 01_coxph_basic.py when the sample size
is small or the likelihood is not well approximated by a quadratic.

Run with:
    python 02_coxph_bootstrap.py
"""

import numpy as np
from common import COXPH_TRUE_COEF, coxph_dataset
from survivalgpu import CoxPHSurvivalAnalysis

data = coxph_dataset()

model = CoxPHSurvivalAnalysis(
    ties="efron",
    device="cpu",  # runs everywhere; use device="cuda" for a GPU speedup
    # if a CUDA-capable GPU + toolkit are available. Bootstrap is the case
    # where the GPU shines most, since all resamples are fitted in parallel.
    nbootstraps=500,  # number of bootstrap resamples
    batchsize=100,  # process 100 resamples at a time on the device
)

model.fit(
    covariates=data["covariates"],
    stop=data["stop"],
    start=data["start"],
    event=data["event"],
    patient=data["patient"],  # required: defines the resampling unit
)

print("True coefficients:  ", list(COXPH_TRUE_COEF.values()))
print("Fitted coefficients:", model.coef_[0])
print("bootstrap_coef_ shape (nbootstraps, n_batch, n_covariates):")
print(" ", model.bootstrap_coef_.shape)

# Percentile bootstrap confidence intervals on the hazard ratios:
boot_hr = np.exp(model.bootstrap_coef_[:, 0, :])  # (nbootstraps, n_covariates)
lower, upper = np.quantile(boot_hr, [0.025, 0.975], axis=0)

for name, hr, lo, hi in zip(COXPH_TRUE_COEF, np.exp(model.coef_[0]), lower, upper, strict=False):
    print(f"HR[{name}] = {hr:.3f}  (bootstrap 95% CI: {lo:.3f} - {hi:.3f})")
