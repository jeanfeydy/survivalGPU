import numpy as np

from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import DataFrame
from rpy2.robjects import Formula

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri


survival = importr("survival")


def coxphfit_to_dict(surv_fit):
    return {
        "coef_": np.array(surv_fit["coefficients"]).reshape(1, -1),
        "loglik_": np.array(surv_fit["loglik"])[1],
        "loglik_init_": np.array(surv_fit["loglik"])[0],
        "hessian_": np.array(surv_fit["var"]),
        "iter_": int(surv_fit["iter"][0]),
    }


def survival_fit(*, dataset, ties, ridge) -> dict:
    """Applies the R survival package on a survivalGPU dataset."""

    df = {
        "stop": dataset.stop,
        "event": dataset.event,
    }

    if np.all(dataset.start == 0):
        formula = "Surv(stop, event) ~ "
    else:
        formula = "Surv(start, stop, event) ~ "
        df["start"] = dataset.start

    # The R survival package only supports scalar covariates:
    assert len(dataset.covariates.shape) == 2
    covars = []
    for i, cov in enumerate(dataset.covariates.T):
        df[f"cov_{i}"] = cov
        covars.append(f" cov_{i}")

    if ridge == 0:
        formula += " + ".join(covars)
    else:
        formula += f"ridge({','.join(covars)}, theta={ridge})"

    if np.any(dataset.strata != 0):
        df["mystrata"] = dataset.strata
        formula += "+ strata(mystrata)"

    if np.any(dataset.batch != 0):
        raise NotImplementedError("We should loop over batch values!")

    with (ro.default_converter + numpy2ri.converter).context():
        # Thanks rpy2!
        data_R = DataFrame(df)
        surv_fit = survival.coxph(
            Formula(formula),
            data=data_R,
            ties=ties,
        )

    return coxphfit_to_dict(surv_fit)
