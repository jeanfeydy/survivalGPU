import numpy as np
import functools

from hypothesis import given, settings
from hypothesis import strategies as st

np.set_printoptions(precision=4)

from survivalgpu import SUPPORTED_TIES, SUPPORTED_BACKENDS, CoxPHSurvivalAnalysis
from survivalgpu.datasets import simple_dataset

from .survival_interface import survival_fit


@functools.cache
def my_dataset(
    *,
    n_covariates,
    n_patients,
    n_strata,
    max_duration,
    unit_length_intervals,
):
    return simple_dataset(
        n_covariates=n_covariates,
        n_patients=n_patients,
        n_strata=n_strata,
        max_duration=max_duration,
        unit_length_intervals=unit_length_intervals,
        ensure_one_life=True,
        ensure_one_death=True,
    )


@given(
    n_covariates=st.integers(min_value=1, max_value=10),
    n_patients=st.sampled_from([10, 100, 1000]),
    max_duration=st.sampled_from([None, 1, 10, 100]),
    ties=st.sampled_from(SUPPORTED_TIES),
    doscale=st.booleans(),
    backend=st.sampled_from(SUPPORTED_BACKENDS),
    ridge=st.sampled_from([1e-2, 1e-1, 1, 10]),
    n_batch=st.integers(min_value=1, max_value=1),
    n_strata=st.integers(min_value=1, max_value=1),
    unit_length_intervals=st.booleans(),
)
@settings(deadline=1000)
def test_survival_equality(
    *,
    n_covariates,
    n_patients,
    max_duration,
    backend,
    ties,
    doscale,
    ridge,
    n_batch,
    n_strata,
    unit_length_intervals,
):
    if max_duration is None:
        max_duration = n_patients

    # Generate a simple dataset:
    ds = my_dataset(
        n_covariates=n_covariates,
        n_patients=n_patients,
        # n_batch=n_batch,
        n_strata=n_strata,
        max_duration=max_duration,
        unit_length_intervals=unit_length_intervals,
    )

    # Fit our model:
    model = CoxPHSurvivalAnalysis(
        ties=ties,
        backend=backend,
        doscale=doscale,
        alpha=ridge,
    )
    model.fit(
        covariates=ds.covariates,
        start=ds.start,
        stop=ds.stop,
        event=ds.event,
        batch=ds.batch,
        strata=ds.strata,
    )

    # Fit the reference model:
    ref_model = survival_fit(dataset=ds, ties=ties, ridge=ridge)

    # Compare the attributes of the two models:
    for key in ref_model.keys():
        if key in ["iter_", "hessian_"]:
            continue
        if ref_model["iter_"] == 1:
            continue
        np.testing.assert_allclose(
            getattr(model, key),
            ref_model[key],
            rtol=1e-3,
            atol=1e-3,
            equal_nan=True,
            err_msg=f"key: {key}, nits: {model.maxiter} vs {ref_model['iter_']}",
        )
