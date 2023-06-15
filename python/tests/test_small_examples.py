import pytest
import numpy as np
from numpy.testing import assert_allclose

from hypothesis import given
from hypothesis import strategies as st

from survivalgpu import coxph_R, CoxPHSurvivalAnalysis
from survivalgpu.utils import numpy
from survivalgpu.optimizers import newton

np.set_printoptions(precision=4)

SUPPORTED_TIES = ["breslow"]  # , "efron"]
SUPPORTED_MODES = ["unit length", "start zero"]  # , "any"]


@pytest.mark.skip()
@given(
    ties=st.sampled_from(SUPPORTED_TIES),
    alpha=st.just(0),
    mode=st.sampled_from(SUPPORTED_MODES),
)
def test_doscale_identity(*, ties, alpha, mode):
    data_csv = np.array(
        [
            # Time, Death, Covars
            [1, 0, 1.0, 0.0],
            [2, 0, -1.0, 0.0],
            [2, 0, 4.0, 4.0],
            [2, 1, 0.0, 2.0],
            [4, 0, 4.0, 2.0],
            [4, 0, 0.0, 0.0],
            [5, 1, 4.0, 1.0],
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

    if mode == "unit length":
        start = ds["stop"] - 1
    elif mode == "start zero":
        start = np.zeros_like(ds["stop"])

    for model in models:
        model.fit(
            covariates=ds["covariates"],
            stop=ds["stop"],
            start=start,
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
                    rtol=5e-2 if attr in ["imat_", "std_"] else 1e-2,
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


@pytest.mark.skip()
@given(
    n_patients=st.integers(min_value=10, max_value=20),
    n_covariates=st.integers(min_value=1, max_value=5),
    n_batch=st.integers(min_value=1, max_value=3),
    n_strata=st.integers(min_value=1, max_value=3),
    ties=st.sampled_from(SUPPORTED_TIES),
    alpha=st.floats(min_value=0.1, max_value=1),
    doscale=st.booleans(),
)
def test_modes_equality(
    *, n_patients, n_covariates, n_batch, n_strata, ties, alpha, doscale
):
    """Checks that all implementations of the CoxPH likelihood coincide when start=0, stop=1."""
    models = [
        CoxPHSurvivalAnalysis(ties=ties, alpha=alpha, mode=mode, doscale=doscale)
        for mode in SUPPORTED_MODES
    ]
    # We need at least two patients per batch to ensure identifiability
    # and thus test equality:
    n_patients = max(n_patients, 2 * n_batch)

    covariates = np.random.randn(n_patients, n_covariates)
    start = np.zeros(n_patients, dtype=np.int64)
    stop = np.ones(n_patients, dtype=np.int64)
    event = np.random.randint(0, 2, size=n_patients, dtype=np.int64)
    batch = np.random.randint(0, n_batch, size=n_patients, dtype=np.int64)
    strata = np.random.randint(0, n_strata, size=n_patients, dtype=np.int64)

    # Ensure that the problem is not degenerate:
    for k in range(n_batch):
        event[2 * k] = 0
        event[2 * k + 1] = 1
        batch[2 * k : 2 * k + 2] = k
        strata[2 * k : 2 * k + 2] = 0

    for model in models:
        model.fit(
            covariates=covariates,
            stop=stop,
            start=start,
            event=event,
            batch=batch,
            strata=strata,
        )

    for attr in dir(models[0]):
        if attr.endswith("_") and not attr.endswith("__"):
            for m in models[1:]:
                if attr in ["sctest_init_"]:  # ["hessian_", "imat_"]:
                    continue
                assert_allclose(
                    getattr(models[0], attr),
                    getattr(m, attr),
                    atol=1e-2 if attr in ["score_"] else 1e-3,
                    rtol=5e-2 if attr in ["imat_", "std_"] else 1e-2,
                    err_msg=f"Attributes m.{attr} do not coincide.",
                )
