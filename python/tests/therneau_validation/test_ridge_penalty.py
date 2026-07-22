"""Standalone check for the per-covariate ridge penalty against R's survival::coxph.

The point of this test is the *selection*: `alpha` penalizes only some of the
covariates (cov_0 is left free, cov_1 and cov_2 get their own ridge strength),
mirroring what you'd write in R as `cov_0 + ridge(cov_1, theta=...) + ridge(cov_2, theta=...)`.
"""

import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import Formula, numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import DataFrame
from survivalgpu import SUPPORTED_TIES, CoxPHSurvivalAnalysis

survival = importr("survival")

# Fixed, reproducible dataset with 3 covariates:
rng = np.random.default_rng(0)
n = 80
covariates = rng.normal(size=(n, 3))
stop = rng.integers(low=1, high=30, size=n).astype(np.int64)
event = (rng.uniform(size=n) > 0.3).astype(np.int64)
event[0], event[-1] = (
    0,
    1,
)  # ensure at least one censored interval and one death

# cov_0 is left unpenalized (alpha=0), cov_1 and cov_2 each get their own
# ridge strength: this is the "not all covariates" selection we want to check.
alpha = np.array([0.0, 2.0, 5.0])


def r_coxph_ridge(ties):
    """Fits Surv(stop, event) ~ cov_0 + ridge(cov_1, theta=2) + ridge(cov_2, theta=5) in R.

    scale=FALSE: R's ridge() penalizes the standardized coefficient by
    default; survivalgpu's alpha penalizes the raw coefficient, so we match
    it against the same (unstandardized) quantity here.
    """
    with (ro.default_converter + numpy2ri.converter).context():
        data = DataFrame(
            {
                "stop": stop,
                "event": event,
                "cov_0": covariates[:, 0],
                "cov_1": covariates[:, 1],
                "cov_2": covariates[:, 2],
            }
        )
        fit = survival.coxph(
            Formula(
                "Surv(stop, event) ~ cov_0"
                f" + ridge(cov_1, theta={alpha[1]}, scale=FALSE)"
                f" + ridge(cov_2, theta={alpha[2]}, scale=FALSE)"
            ),
            data=data,
            ties=ties,
        )
    return {
        "coef_": np.array(fit.getbyname("coefficients")),
        "loglik_": np.array(fit.getbyname("loglik"))[1],
    }


def test_ridge_partial_selection():
    """Penalizing only a subset of covariates should match R's mixed formula."""
    for ties in SUPPORTED_TIES:
        model = CoxPHSurvivalAnalysis(ties=ties, alpha=alpha)
        model.fit(covariates=covariates, stop=stop, event=event)

        ref = r_coxph_ridge(ties)

        np.testing.assert_allclose(
            model.coef_.ravel(),
            ref["coef_"],
            rtol=1e-3,
            atol=1e-3,
            err_msg=f"coef_ mismatch for ties={ties!r}",
        )
        np.testing.assert_allclose(
            model.loglik_,
            ref["loglik_"],
            rtol=1e-3,
            atol=1e-3,
            err_msg=f"loglik_ mismatch for ties={ties!r}",
        )
