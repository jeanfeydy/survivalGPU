import pytest
from contextlib import nullcontext

from hypothesis import given
from hypothesis import strategies as st

import torch
import numpy as np
from survivalgpu.datasets import load_drugs
from survivalgpu.torch_datasets import torch_lexsort

small_int = st.integers(min_value=1, max_value=10)


@given(
    n_covariates=st.one_of(st.just(0), small_int),
    n_drugs=st.one_of(st.just(0), small_int),
    n_patients=small_int,
    max_duration=small_int,
    max_offset=small_int,
)
def test_dataset_drugs_shapes(
    *, n_covariates: int, n_drugs: int, n_patients: int, **kwargs
):
    """Test dataset loading."""

    # Catch exception if n_drugs == 0 and n_covariates == 0:
    invalid = n_drugs == 0 and n_covariates == 0
    ctxt = pytest.raises(ValueError) if invalid else nullcontext()

    with ctxt:
        ds = load_drugs(
            n_covariates=n_covariates, n_drugs=n_drugs, n_patients=n_patients, **kwargs
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


@given(
    n_covariates=st.one_of(st.just(0), small_int),
    n_drugs=st.one_of(st.just(0), small_int),
    n_patients=small_int,
    max_duration=small_int,
    max_offset=small_int,
)
def test_dataset_to_img(*, n_covariates: int, n_drugs: int, **kwargs):
    """Test the image export feature."""

    # Catch exception if n_drugs == 0 and n_covariates == 0:
    invalid = n_drugs == 0 and n_covariates == 0
    ctxt = pytest.raises(ValueError) if invalid else nullcontext()

    with ctxt:
        ds = load_drugs(n_covariates=n_covariates, n_drugs=n_drugs, **kwargs)
    if invalid:
        return

    # The dtypes and shapes of the data arrays are all checked
    # in the dataset loading function by @typecheck.
    # For instance, the code below is redundant:
    img = ds.to_img()

    # We expect a [H, W, 3] image with uint8 values (from 0 to 255).
    assert img.dtype == np.uint8
    assert img.ndim == 3
    assert img.shape[-1] == 3


@given(
    n_values=small_int,
    n_keys=small_int,
    n_vectors=st.integers(min_value=1, max_value=100),
    use_cuda=st.booleans(),
)
def test_torch_lexsort(*, n_values: int, n_keys: int, n_vectors: int, use_cuda: bool):
    # Make random float vector with duplicates to test if it handles floating point well
    a = torch.randint(0, n_values, (n_keys, n_vectors))

    if use_cuda and torch.cuda.is_available():
        a = a.cuda()

    ind = torch_lexsort(a)
    ind_np = torch.from_numpy(np.lexsort(a.numpy()))

    if use_cuda and torch.cuda.is_available():
        ind_np = ind_np.cuda()

    assert torch.all(ind == ind_np)
    assert torch.all(a[:, ind] == a[:, ind_np])
