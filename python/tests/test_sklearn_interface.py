import pytest
from hypothesis import given
from hypothesis import strategies as st

from survivalgpu import CoxPHSurvivalAnalysis, WCESurvivalAnalysis
from survivalgpu.datasets import load_drugs


small_int = st.integers(min_value=1, max_value=10)


@pytest.mark.skip()
@given(n_covariates=small_int, n_patients=small_int, n_times=small_int)
def test_coxph_shapes(*, n_covariates, n_patients, n_times):
    """Tests the shapes of the CoxPHSurvivalAnalysis attributes."""

    ds = load_drugs(
        n_covariates=n_covariates,
        n_drugs=0,
        n_patients=n_patients,
    )
    model = CoxPHSurvivalAnalysis()
    model.fit(ds.covariates, stop=ds.stop, event=ds.event)

    assert model.coef_.shape == (ds.covariates.shape[1],)


@pytest.mark.skip()
def test_wce_shapes():
    """Tests the shapes of the WCESurvivalAnalysis attributes."""

    ds = load_drugs(n_drugs=1, n_patients=1, n_times=1)
    model = WCESurvivalAnalysis()
    model.fit(ds.covariates, ds.times, events=ds.events)

    assert model is not None
