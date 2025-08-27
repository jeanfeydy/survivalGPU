import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose
from survivalgpu import CoxPHSurvivalAnalysis

np.set_printoptions(precision=4)

SUPPORTED_TIES = ["breslow", "efron"]
SUPPORTED_MODES = ["unit length", "start zero", "any"]

challenges = [
    np.array(
        [
            # Time, Death, Covars
            [1, 0, 1.0],
            [1, 1, 0.0],
            [2, 1, 4.0],
        ]
    ),
    np.array(
        [
            # Time, Death, Covars
            [1, 0, 1.0, 0.0],
            [2, 0, -1.0, 0.0],
            [2, 0, 4.0, 4.0],
            [2, 1, 0.0, 2.0],
            [4, 0, 4.0, 2.0],
            [4, 0, 0.0, 1.0],
            [5, 1, 4.0, 1.0],
        ]
    ),
]


@given(
    ties=st.sampled_from(SUPPORTED_TIES),
    alpha=st.just(0.1),
    mode=st.sampled_from(SUPPORTED_MODES),
    example=st.integers(min_value=0, max_value=len(challenges) - 1),
)
@settings(deadline=5000)
def test_doscale_identity(*, ties, alpha, mode, example):
    """Checks that doscale=True and doscale=False give the same results on small datasets."""

    data_csv = challenges[example]
    ds = {
        "stop": data_csv[:, 0].astype(np.int64),
        "event": data_csv[:, 1].astype(np.int64),
        "covariates": data_csv[:, 2:],
    }

    models = [
        CoxPHSurvivalAnalysis(ties=ties, alpha=alpha, doscale=doscale)
        for doscale in [True, False]
    ]

    if mode == "unit length":
        start = ds["stop"] - 1
    elif mode == "start zero":
        start = np.zeros_like(ds["stop"])
    elif mode == "any":
        start = -ds["stop"]

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
                assert_allclose(
                    getattr(models[0], attr),
                    getattr(m, attr),
                    atol=1e-3,
                    rtol=5e-2 if attr in ["imat_", "std_"] else 1e-2,
                    err_msg=f"Attributes m.{attr} do not coincide.",
                )
