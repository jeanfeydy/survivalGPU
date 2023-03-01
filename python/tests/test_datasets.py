import pytest
from contextlib import nullcontext

from hypothesis import given
from hypothesis import strategies as st

import torch
import numpy as np
from survivalgpu.datasets import load_drugs, SurvivalDataset
from survivalgpu.torch_datasets import torch_lexsort

small_int = st.integers(min_value=1, max_value=10)

if torch.cuda.is_available():
    st_device = st.sampled_from(["cpu", "cuda"])
else:
    st_device = st.just("cpu")


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
    """Tests the torch_lexsort function from torch_datasets.py."""
    # Make random float vector with duplicates to test if it handles floating point well
    a = torch.randint(0, n_values, (n_keys, n_vectors))

    if use_cuda and torch.cuda.is_available():
        a = a.cuda()

    ind = torch_lexsort(a)
    ind_np = torch.from_numpy(np.lexsort(a.numpy()))

    if use_cuda and torch.cuda.is_available():
        ind_np = ind_np.cuda()

    # N.B.: We cannot check equality for the indices because their values are ill-defined
    #       when there are duplicates.
    # assert torch.all(ind == ind_np)
    assert torch.all(a[:, ind] == a[:, ind_np])


@given(
    n_intervals=small_int,
    n_covariates=small_int,
    rescale=st.booleans(),
    device=st_device,
)
def test_scale(n_intervals: int, n_covariates: int, rescale: bool, device: str):
    """Tests the rescaling method of TorchSurvivalDataset."""

    # Create a minimal random dataset:
    rng = np.random.default_rng()
    stop = rng.integers(low=1, high=100, size=(n_intervals,))
    covariates = rng.normal(loc=0, scale=1, size=(n_intervals, n_covariates))

    dataset = SurvivalDataset(
        stop=stop,
        covariates=covariates,
    ).to_torch(device=device)

    means, scales = dataset.scale(rescale=rescale)

    assert means.shape == (n_covariates,)
    assert np.allclose(means, np.mean(covariates, axis=0), atol=1e-3)

    if rescale:
        # Check that the data has been centered:
        assert torch.allclose(
            dataset.covariates.mean(dim=0),
            torch.zeros(n_covariates, dtype=torch.float32),
            atol=1e-3,
        )

        # Check that the scales have the expected shape:
        assert scales.shape == (n_covariates,)

        # Check that the data has been rescaled:
        if n_intervals > 1:
            assert torch.allclose(
                dataset.covariates.abs().sum(dim=0),
                torch.ones(n_covariates, dtype=torch.float32),
            )
        else:
            assert torch.allclose(
                dataset.covariates.abs().sum(dim=0),
                torch.zeros(n_covariates, dtype=torch.float32),
                atol=1e-3,
            )
    else:
        assert scales is None
