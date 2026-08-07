"""CoxPH, scikit-learn-style class API, no bootstrap, CPU.

This is the simplest way to fit a Cox Proportional Hazards model with
survivalGPU: build a `CoxPHSurvivalAnalysis`, call `.fit()`, then read the
results off the fitted object's attributes (all of them end with an
underscore, as in scikit-learn/scikit-survival).

Run with:
    python 01_coxph_basic.py
"""

import numpy as np
from common import COXPH_TRUE_COEF, coxph_dataset
from survivalgpu import CoxPHSurvivalAnalysis

data = coxph_dataset()

model = CoxPHSurvivalAnalysis(
    ties="efron",  # or "breslow"
    device="cpu",  # runs everywhere; use device="cuda" for a GPU speedup
    # if a CUDA-capable GPU + toolkit are available (or device=None to
    # auto-select the best one via survivalgpu.utils.default_device).
)

model.fit(
    covariates=data["covariates"],
    stop=data["stop"],
    start=data["start"],
    event=data["event"],
    patient=data["patient"],
)

print("True coefficients:  ", list(COXPH_TRUE_COEF.values()))
print("Fitted coefficients:", model.coef_[0])
print("Standard errors:    ", model.std_[0])
print("Hazard ratios:      ", np.exp(model.coef_[0]))
print("Log-likelihood:     ", model.loglik_[0])
print("Newton iterations:  ", model.iter_)

# 95% Wald confidence intervals on the hazard ratios:
lower = np.exp(model.coef_[0] - 1.96 * model.std_[0])
upper = np.exp(model.coef_[0] + 1.96 * model.std_[0])
for name, hr, lo, hi in zip(COXPH_TRUE_COEF, np.exp(model.coef_[0]), lower, upper, strict=False):
    print(f"HR[{name}] = {hr:.3f}  (95% CI: {lo:.3f} - {hi:.3f})")
