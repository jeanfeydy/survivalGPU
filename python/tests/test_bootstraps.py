from hypothesis import given
from hypothesis import strategies as st

import torch
import numpy as np
from survivalgpu.datasets import load_drugs, SurvivalDataset
from survivalgpu.bootstrap import Resampling
from survivalgpu.group_reduction import group_reduce

from math import ceil, sqrt

small_int = st.integers(min_value=1, max_value=10)

if torch.cuda.is_available():
    st_device = st.sampled_from(["cpu", "cuda"])
else:
    st_device = st.just("cpu")


# Test the basic Resampling functionalities ==============================================


@given(
    n_patients=small_int,
    n_bootstraps=small_int,
    n_samples=small_int,
    n_intervals=small_int,
    use_cuda=st.booleans(),
)
def test_resampling_shapes(
    *,
    n_patients: int,
    n_bootstraps: int,
    n_samples: int,
    n_intervals: int,
    use_cuda: bool,
):
    """Tests the Resampling constructor."""
    indices = torch.randint(0, n_patients, size=(n_bootstraps, n_samples))
    patients = torch.randint(0, n_patients, size=(n_intervals,))

    # Make sure that we "use" all the patients:
    patients[0] = n_patients - 1

    if use_cuda and torch.cuda.is_available():
        indices = indices.cuda()
        patients = patients.cuda()

    res = Resampling(indices=indices, patient=patients)

    assert res.patient_weights.shape == (n_bootstraps, n_patients)
    assert res.patient_log_weights.shape == (n_bootstraps, n_patients)
    assert res.patient_weights.dtype == torch.float32
    assert res.patient_log_weights.dtype == torch.float32

    assert res.interval_weights.shape == (n_bootstraps, n_intervals)
    assert res.interval_log_weights.shape == (n_bootstraps, n_intervals)
    assert res.interval_weights.dtype == torch.float32
    assert res.interval_log_weights.dtype == torch.float32


@given(
    n_patients=small_int,
    n_bootstraps=small_int,
    n_samples=small_int,
    n_intervals=small_int,
    use_cuda=st.booleans(),
)
def test_resampling_single(
    *,
    n_patients: int,
    n_bootstraps: int,
    n_samples: int,
    n_intervals: int,
    use_cuda: bool,
):
    """Checks that 'resampling' a single sample works as expected."""
    unique_patient = torch.randint(0, n_patients, size=(1,)).item()
    # indices is constant: we only care about the unique patient above!
    indices = unique_patient * torch.ones(n_bootstraps, n_samples, dtype=torch.int64)
    patients = torch.randint(0, n_patients, size=(n_intervals,))

    # Make sure that we "use" all the patients:
    patients[0] = n_patients - 1

    if use_cuda and torch.cuda.is_available():
        indices = indices.cuda()
        patients = patients.cuda()

    res = Resampling(indices=indices, patient=patients)

    # Expected patient weights: zeros, except for the unique patient
    # that get a weight that is equal to n_samples.
    expected_patient_weights = torch.zeros(n_bootstraps, n_patients)
    expected_patient_weights[:, unique_patient] = n_samples

    # Expected interval weights: zeros, except for the intervals that
    # are associated to the unique patient.
    expected_interval_weights = torch.zeros(n_bootstraps, n_intervals)
    expected_interval_weights[:, patients == unique_patient] = n_samples

    if use_cuda and torch.cuda.is_available():
        expected_patient_weights = expected_patient_weights.cuda()
        expected_interval_weights = expected_interval_weights.cuda()

    assert torch.allclose(res.patient_weights, expected_patient_weights)
    assert torch.allclose(res.interval_weights, expected_interval_weights)


# Test the dataset resampling methods ====================================================


def simple_dataset(use_patient: bool, device: str):
    n_covariates = 2
    stop = np.array([2, 2, 2, 2, 2, 5, 5, 6, 6, 6])
    event = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 1])
    covariates = np.zeros((len(stop), n_covariates))
    if use_patient:
        start = stop - 1
        patient = np.array([0, 1, 2, 3, 4, 0, 1, 0, 5, 7])
        n_patients = 8
    else:
        start = None
        patient = None
        n_patients = len(stop)

    dataset = SurvivalDataset(
        start=start,
        stop=stop,
        event=event,
        patient=patient,
        covariates=covariates,
    )
    dataset = dataset.to_torch(device).sort().count_deaths()
    return dataset, n_patients


@given(
    use_patient=st.booleans(),
    device=st_device,
)
def test_original_sample_simple(use_patient: bool, device: str):
    """Tests the original sample method on a simple handcrafted dataset."""

    dataset, n_patients = simple_dataset(use_patient, device)
    res = dataset.original_sample()
    assert res.patient_weights.shape == (1, n_patients)
    assert res.interval_weights.shape == (1, len(dataset.stop))
    assert torch.allclose(res.patient_weights, torch.ones(1, n_patients, device=device))
    assert torch.allclose(
        res.interval_weights, torch.ones(1, len(dataset.stop), device=device)
    )


@given(
    n_bootstraps=small_int,
    batch_size=small_int,
    use_patient=st.booleans(),
    device=st_device,
)
def test_bootstraps_simple(
    n_bootstraps: int, batch_size: int, use_patient: bool, device: str
):
    """Tests the bootstrap method on a simple handcrafted dataset."""
    dataset, n_patients = simple_dataset(use_patient, device)

    boots = dataset.bootstraps(n_bootstraps=n_bootstraps, batch_size=batch_size)
    assert len(boots) == ceil(n_bootstraps / batch_size)
    assert sum([len(b) for b in boots]) == n_bootstraps

    for it, res in enumerate(boots):
        if it < len(boots) - 1:
            b = batch_size
        else:
            b = n_bootstraps % batch_size
            if b == 0:
                b = batch_size

        assert len(res) == b
        assert res.patient_weights.shape == (b, n_patients)
        assert res.interval_weights.shape == (b, len(dataset.stop))
        assert torch.allclose(
            res.patient_weights.sum(dim=1),
            n_patients * torch.ones(b, device=device),
        )


@given(
    n_groups=small_int,
    n_intervals=small_int,
    n_bootstraps=st.one_of(small_int, st.just(1000), st.just(10000)),
    device=st_device,
)
def test_bootstraps_stratification_1(
    n_groups: int, n_intervals: int, n_bootstraps: int, device: str
):
    """Checks that stratification works as expected."""

    # Stop, event and covariates don't really matter here:
    stop = np.random.randint(1, 10, size=(n_intervals,))
    event = np.random.randint(0, 2, size=(n_intervals,))
    covariates = np.zeros((n_intervals, 1))

    # Batch is a random vector that defines at most n_groups separate groups:
    batch = np.random.randint(0, n_groups, size=(n_intervals,))

    # Wrap the data in a TorchSurvivalDataset object:
    dataset = SurvivalDataset(
        stop=stop,
        event=event,
        covariates=covariates,
        batch=batch,
    )
    dataset = dataset.to_torch(device).sort().count_deaths()

    # Retrieve our bootstraps in a single Resampling object:
    boots = dataset.bootstraps(n_bootstraps=n_bootstraps, batch_size=n_bootstraps)[0]

    # Simple check on the shapes, as in test_bootstraps_simple():
    assert boots.patient_weights.shape == (n_bootstraps, n_intervals)

    # Check that the total number of samples per group is preserved ----------------------

    batch = torch.from_numpy(batch).to(device=device)
    # Compute the original number of patients per strata:
    weight_per_strata = (
        torch.bincount(batch, minlength=n_groups).tile((n_bootstraps, 1)).float()
    )

    # Compute the total weight per strata:
    new_weight_per_strata = group_reduce(
        values=boots.patient_weights,
        groups=batch.view(1, -1).tile((n_bootstraps, 1)),
        reduction="sum",
        output_size=n_groups,
        backend="pyg",
    )

    assert torch.allclose(weight_per_strata, new_weight_per_strata)

    # Check that every patient has an equal probability of being sampled -----------------
    if n_bootstraps >= 1000:
        # each cell of boots.patient_weights is a random variable with expected
        # mean value of 1 and finite variance that depends on the number of patients
        # per group. (A patient that is alone is always going to get picked, with
        # weight=1, whereas a patient in a more populous groups may experience
        # a wider range of fortunes.)
        # In any case, according to the central limit theorem,
        # we expect that the average empirical probas over n_bootstraps will
        # be equal to 1 + Cst * N(0,1) / sqrt(n_bootstraps)
        probas = boots.patient_weights.mean(dim=0)  # (n_intervals,)

        # We can reasonably expect that Cst ~ 1, and ask with >99% certainty
        # that the error falls in the confidence interval +- 3/sqrt(n_boostraps):
        assert torch.allclose(
            probas, torch.ones_like(probas), atol=3 / sqrt(n_bootstraps)
        )
