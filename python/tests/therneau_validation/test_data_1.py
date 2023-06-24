import numpy as np

from hypothesis import given, settings
from hypothesis import strategies as st

from survivalgpu import SUPPORTED_TIES, CoxPHSurvivalAnalysis

np.set_printoptions(precision=4)


data_csv = np.array(
    [
        # Time, Death, Covar
        [1, 1, 1],
        [1, 0, 1],
        [6, 1, 1],
        [6, 1, 0],
        [8, 0, 0],
        [9, 1, 0],
    ]
)
ds = {
    "stop": data_csv[:, 0].astype(np.int64),
    "event": data_csv[:, 1].astype(np.int64),
    "covariates": data_csv[:, 2:3].astype(np.float64),
}

final_values = {
    "breslow": {
        "coef_": 1.475285,
        "loglik_": -3.824750,
        "loglik_init_": -4.564248,
        "hessian_": 0.6341681,
        "score_": 0.0,
    },
    "efron": {
        "coef_": 1.676857,
    },
}


@given(
    ties=st.sampled_from(SUPPORTED_TIES),
    doscale=st.booleans(),
)
@settings(deadline=1000)
def test_final_value(*, ties, doscale):
    model = CoxPHSurvivalAnalysis(ties=ties, doscale=doscale)
    model.fit(
        covariates=ds["covariates"],
        stop=ds["stop"],
        event=ds["event"],
    )
    for key, value in final_values[ties].items():
        assert np.allclose(getattr(model, key), value, rtol=1e-3, atol=1e-3)


iter_values = {
    "breslow": {
        "coef_": [
            0.0,
            1.6,
            1.47272353,
            1.47528396,
            1.47528491,
            1.47528491,
        ],
        "loglik_": [
            -4.56434819,
            -3.82961962,
            -3.82475159,
            -3.82474951,
            -3.82474951,
            -3.82474951,
        ],
        "score_": [
            1.000000000,
            -7.75891712e-02,
            1.62495334e-03,
            6.05079992e-07,
            8.41399792e-14,
            0.00000000e00,
        ],
        "hessian_": [
            0.625000000,
            0.609611286,
            0.634641180,
            0.634168319,
            0.634168143,
            0.634168143,
        ],
    },
    "efron": {
        "coef_": 1.676857,
    },
}


@given(
    ties=st.sampled_from(SUPPORTED_TIES),
)
@settings(deadline=1000)
def test_iterations(*, ties):
    for maxiter in range(len(iter_values[ties]["coef_"])):
        model = CoxPHSurvivalAnalysis(ties=ties, maxiter=maxiter)
        model.fit(
            covariates=ds["covariates"],
            stop=ds["stop"],
            event=ds["event"],
        )
        for key, values in iter_values[ties].items():
            assert np.allclose(
                getattr(model, key), values[maxiter], rtol=1e-3, atol=1e-3
            ), f"maxiter={maxiter}, key={key}"


if __name__ == "__main__":
    # Use the PyTorch profiler to get a trace of the forward pass
    # and the backward pass.
    import torch
    from torch.profiler import profile, record_function, ProfilerActivity

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        test_final_value(ties="breslow", doscale=False)

    # Export to chrome://tracing
    prof.export_chrome_trace("profile.json")
