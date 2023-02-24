import pytest
from contextlib import nullcontext

from hypothesis import given
from hypothesis import strategies as st

import numpy as np
from survivalgpu.datasets import load_drugs

small_int = st.integers(min_value=1, max_value=10)


@given(
    n_covariates=st.one_of(st.just(0), small_int),
    n_drugs=st.one_of(st.just(0), small_int),
    n_patients=small_int,
    max_duration=small_int,
    max_offset=small_int,
)
def test_dataset_drugs_shapes(
    *, n_covariates, n_drugs, n_patients, max_duration, max_offset
):
    """Test dataset loading."""

    # Catch exception if n_drugs == 0 and n_covariates == 0:
    invalid = n_drugs == 0 and n_covariates == 0
    ctxt = pytest.raises(ValueError) if invalid else nullcontext()

    with ctxt:
        ds = load_drugs(
            n_covariates=n_covariates,
            n_drugs=n_drugs,
            n_patients=n_patients,
            max_duration=max_duration,
            max_offset=max_offset,
        )
    if invalid:
        return

    # The dtypes and shapes of the data arrays are all checked
    # in the dataset loading function by @typecheck.
    # For instance, the code below is redundant:
    if ds.n_covariates > 0:
        assert ds.covariates.dtype == np.float64
        assert ds.covariates.shape == (n_patients, n_covariates)

    # We only need to check the @property methods.
    assert ds.n_patients == n_patients
    assert ds.n_intervals == n_patients
    assert ds.n_covariates == n_covariates
    # N.B.: Some drugs may not be prescribed to any patient.
    assert ds.n_drugs <= n_drugs
