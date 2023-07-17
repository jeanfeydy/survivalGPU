import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from survivalgpu import CoxPHSurvivalAnalysis, WCESurvivalAnalysis
from survivalgpu.datasets import simple_dataset, load_drugs


small_int = st.integers(min_value=1, max_value=10)


@given(
    n_covariates=small_int,
    n_patients=st.integers(min_value=30, max_value=40),
    n_batch=st.integers(min_value=1, max_value=3),
    n_strata=st.integers(min_value=1, max_value=3),
    max_duration=small_int,
)
@settings(deadline=1000)
def test_coxph_shapes(*, n_covariates, n_patients, n_batch, n_strata, max_duration):
    """Tests the shapes of the CoxPHSurvivalAnalysis attributes."""

    ds = simple_dataset(
        n_covariates=n_covariates,
        n_patients=n_patients,
        n_batch=n_batch,
        n_strata=n_strata,
        max_duration=max_duration,
        ensure_one_life=True,
        ensure_one_death=True,
        unit_length_intervals=True,
    )

    model = CoxPHSurvivalAnalysis(ties="breslow", alpha=0.01)

    model.fit(
        covariates=ds.covariates,
        start=ds.start,
        stop=ds.stop,
        event=ds.event,
        batch=ds.batch,
        strata=ds.strata,
    )

    assert model.coef_.shape == (n_batch, n_covariates)


@pytest.mark.skip()
def test_wce_shapes():
    """Tests the shapes of the WCESurvivalAnalysis attributes."""

    ds = load_drugs(n_drugs=1, n_patients=1, n_times=1)
    model = WCESurvivalAnalysis(cutoff=10, order=3, n_knots=1)
    model.fit(ds.covariates, ds.times, events=ds.events)

    assert model is not None
