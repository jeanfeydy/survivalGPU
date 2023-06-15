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

# Test the front-end "NumPy" SurvivalDataset class =======================================


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


# Test the back-end TorchSurvivalDataset class ===========================================


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
    ind_np = torch.from_numpy(np.lexsort(a.cpu().numpy()))

    if use_cuda and torch.cuda.is_available():
        ind_np = ind_np.cuda()

    # N.B.: We cannot check equality for the indices because their values are ill-defined
    #       when there are duplicates.
    # assert torch.all(ind == ind_np)
    assert torch.all(a[:, ind] == a[:, ind_np])


@given(
    n_intervals=st.integers(min_value=1, max_value=100),
    n_covariates=small_int,
    n_batches=small_int,
    n_strata=small_int,
    max_time=small_int,
    device=st_device,
)
def test_sort(
    n_intervals: int,
    n_covariates: int,
    n_batches: int,
    n_strata: int,
    max_time: int,
    device: str,
):
    """Tests the `.sort()` method of TorchSurvivalDataset."""

    # N.B.: For the sake of simplicity, we assume one interval per patient:
    n_patients = n_intervals

    # Create a minimal random dataset:
    rng = np.random.default_rng()
    stop = rng.integers(low=1, high=1 + max_time, size=(n_intervals,))
    event = rng.integers(low=0, high=2, size=(n_intervals,))
    batch = rng.integers(low=0, high=n_batches, size=(n_patients,))
    strata = rng.integers(low=0, high=n_strata, size=(n_patients,))
    covariates = rng.normal(loc=0, scale=1, size=(n_intervals, n_covariates))

    dataset = SurvivalDataset(
        stop=stop,
        event=event,
        batch=batch,
        strata=strata,
        covariates=covariates,
    ).to_torch(device=device)

    # Lexicographically sort the dataset on (batch > strata > stop > event):
    dataset.sort()

    index = (
        dataset.event
        + 2 * dataset.stop
        + 2 * (1 + max_time) * dataset.strata_intervals
        + 2 * (1 + max_time) * n_strata * dataset.batch_intervals
    )
    assert torch.all(index == torch.sort(index)[0])


@given(
    n_intervals=small_int,
    n_covariates=small_int,
    rescale=st.booleans(),
    device=st_device,
)
def test_scale(n_intervals: int, n_covariates: int, rescale: bool, device: str):
    """Tests the `.scale()` method of TorchSurvivalDataset."""

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
    assert np.allclose(means.cpu(), np.mean(covariates, axis=0), atol=1e-3)

    if rescale:
        # Check that the data has been centered:
        assert torch.allclose(
            dataset.covariates.mean(dim=0),
            torch.zeros(n_covariates, dtype=torch.float32, device=device),
            atol=1e-3,
        )

        # Check that the scales have the expected shape:
        assert scales.shape == (n_covariates,)

        # Check that the data has been rescaled:
        if n_intervals > 1:
            assert torch.allclose(
                dataset.covariates.abs().sum(dim=0),
                torch.ones(n_covariates, dtype=torch.float32, device=device),
            )
        else:
            assert torch.allclose(
                dataset.covariates.abs().sum(dim=0),
                torch.zeros(n_covariates, dtype=torch.float32, device=device),
                atol=1e-3,
            )
    else:
        assert scales is None


@given(
    n_covariates=small_int,
    device=st_device,
)
def test_count_death_simple(n_covariates: int, device: str):
    """Tests the `.count_deaths()` method of TorchSurvivalDataset on a handcrafted example."""
    stop = np.array([2, 2, 2, 2, 2, 5, 5, 6, 6, 6])
    event = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 1])
    covariates = np.zeros((len(stop), n_covariates))
    dataset = SurvivalDataset(stop=stop, event=event, covariates=covariates)
    dataset = dataset.to_torch(device).sort().count_deaths()

    assert dataset.is_sorted
    assert torch.equal(
        dataset.unique_groups,
        torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 0],
                [2, 5, 6],
            ],
            dtype=torch.int64,
            device=device,
        ),
    )
    assert dataset.n_groups == 3
    assert torch.equal(
        dataset.tied_deaths, torch.tensor([2, 1, 1], dtype=torch.int64, device=device)
    )


@given(
    n_intervals=small_int,
    n_covariates=small_int,
    n_batches=small_int,
    n_strata=small_int,
    device=st_device,
)
def test_count_death(
    n_intervals: int,
    n_covariates: int,
    n_batches: int,
    n_strata: int,
    device: str,
):
    """Tests the `.count_deaths()` method of TorchSurvivalDataset on a synthetic example."""
    rng = np.random.default_rng()
    # Create a simple dataset with 2 groups per batch and per strata:
    stop, event, batch, strata, covariates = [], [], [], [], []

    tied_deaths = []
    # Loop over the batches and stratas:
    for b in range(n_batches):
        for s in range(n_strata):
            n_total = 0  # total number of intervals for the current batch and strata
            time = 0  # start time
            n_groups = rng.integers(low=1, high=10)  # number of distinct stop times
            for _ in range(n_groups):
                # We progressively increase the time:
                time += rng.integers(low=1, high=10)
                n_censored = rng.integers(low=1, high=10)
                n_death = rng.integers(low=1, high=10)

                n_total += n_censored + n_death
                stop += [
                    time * np.ones(n_censored + n_death, dtype=np.int64),
                ]
                event += [
                    np.zeros(n_censored, dtype=np.int64),
                    np.ones(n_death, dtype=np.int64),
                ]
                tied_deaths += [n_death]

            batch += [b * np.ones(n_total, dtype=np.int64)]
            strata += [s * np.ones(n_total, dtype=np.int64)]
            covariates += [np.zeros((n_total, n_covariates), dtype=np.float64)]

    # Spice things up with a random permutation:
    perm = rng.permutation(len(np.concatenate(stop)))

    dataset = (
        SurvivalDataset(
            stop=np.concatenate(stop)[perm],
            event=np.concatenate(event)[perm],
            batch=np.concatenate(batch)[perm],
            strata=np.concatenate(strata)[perm],
            covariates=np.concatenate(covariates)[perm, :],
        )
        .to_torch(device=device)
        .sort()
        .count_deaths()
    )

    assert dataset.is_sorted
    assert dataset.n_groups == len(tied_deaths)
    assert torch.equal(
        dataset.tied_deaths,
        torch.tensor(tied_deaths, dtype=torch.int64, device=device),
    )


@given(
    n_covariates=small_int,
    device=st_device,
)
def test_prune_simple(n_covariates: int, device: str):
    """Tests the `.prune()` method of TorchSurvivalDataset on a handcrafted example."""

    # Simple example with a dummy group at time 3
    stop = np.array([1, 1, 1, 3, 3, 4, 4, 6, 6, 6, 7])
    event = np.array([0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1])
    covariates = np.zeros((len(stop), n_covariates))
    dataset = SurvivalDataset(
        start=stop - 1, stop=stop, event=event, covariates=covariates
    )
    dataset = dataset.to_torch(device).sort().count_deaths()

    # Apply the .prune() method to remove the dummy group:
    dataset = dataset.prune()

    assert dataset.is_sorted
    assert torch.equal(
        dataset.unique_groups,
        torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 4, 6, 7],
            ],
            dtype=torch.int64,
            device=device,
        ),
    )
    assert dataset.n_groups == 4
    assert torch.equal(
        dataset.tied_deaths,
        torch.tensor([1, 2, 1, 1], dtype=torch.int64, device=device),
    )
