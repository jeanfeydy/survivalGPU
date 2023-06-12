import pytest
import numpy as np
from numpy.testing import assert_allclose
import torch

from survivalgpu import coxph_R, CoxPHSurvivalAnalysis
from survivalgpu.utils import numpy
from survivalgpu.optimizers import newton


np.set_printoptions(precision=4)


def test_sanity_check_1(ties="breslow", alpha=0):
    data_csv = np.array(
        [
            # Time, Death, Covars
            [1, 0, -1.0, 0.0],
            [1, 0, 4.0, 4.0],
            [1, 1, 0.0, 2.0],
            [2, 0, 4.0, 2.0],
            [2, 0, 0.0, 0.0],
            [3, 1, 4.0, 1.0],
        ]
    )
    ds = {
        "stop": data_csv[:, 0].astype(np.int64),
        "event": data_csv[:, 1].astype(np.int64),
        "covariates": data_csv[:, 2:4],
    }

    models = [
        CoxPHSurvivalAnalysis(ties=ties, alpha=alpha, doscale=doscale)
        for doscale in [True, False]
    ]

    for model in models:
        model.fit(
            covariates=ds["covariates"],
            stop=ds["stop"],
            start=ds["stop"] - 1,
            event=ds["event"],
        )

    for attr in dir(models[0]):
        if attr.endswith("_") and not attr.endswith("__"):
            for m in models[1:]:
                print(attr)
                assert_allclose(
                    getattr(models[0], attr),
                    getattr(m, attr),
                    atol=1e-5,
                    rtol=4e-2,
                    err_msg=f"Attributes m.{attr} do not coincide.",
                )

    if False:
        for ties in ["efron", "breslow"]:
            for doscale in [True, False]:
                print(f"\nties = {ties}, doscale = {doscale} ========")
                res = coxph_R(
                    data,
                    "stop",
                    "death",
                    ["covar1", "covar2"],
                    bootstrap=1,
                    ties=ties,
                    doscale=doscale,
                    profile=None,
                )
                for key, item in res.items():
                    print(f"{key}:")
                    print(item)
