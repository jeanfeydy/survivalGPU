import pytest
from hypothesis import given
from hypothesis import strategies as st

from survivalgpu import CoxPHSurvivalAnalysis, WCESurvivalAnalysis
from survivalgpu.datasets import test_dataset, load_drugs


small_int = st.integers(min_value=1, max_value=10)


# @pytest.mark.skip()
@given(
    n_covariates=small_int,
    n_patients=st.integers(min_value=2, max_value=10),
    n_batch=small_int,
    n_strata=small_int,
    max_duration=small_int,
)
def test_coxph_shapes(*, n_covariates, n_patients, n_batch, n_strata, max_duration):
    """Tests the shapes of the CoxPHSurvivalAnalysis attributes."""

    ds = test_dataset(
        n_covariates=n_covariates,
        n_patients=n_patients,
        n_batch=n_batch,
        n_strata=n_strata,
        max_duration=max_duration,
        ensure_one_life=True,
        ensure_one_death=True,
    )
    model = CoxPHSurvivalAnalysis(ties="breslow")
    model.fit(ds.covariates, stop=ds.stop, event=ds.event)

    assert model.coef_.shape == (n_batch, n_covariates)


@pytest.mark.skip()
def test_wce_shapes():
    """Tests the shapes of the WCESurvivalAnalysis attributes."""

    ds = load_drugs(n_drugs=1, n_patients=1, n_times=1)
    model = WCESurvivalAnalysis()
    model.fit(ds.covariates, ds.times, events=ds.events)

    assert model is not None
