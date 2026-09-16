from math import ceil, sqrt

import numpy as np
import torch
from hypothesis import given
from hypothesis import strategies as st
from survivalgpu import (
    ConstantCovariate,
    CoxPHSurvivalAnalysis,
    simulate_dataset,
)
from survivalgpu.bootstrap import Resampling
from survivalgpu.datasets import SurvivalDataset
from survivalgpu.group_reduction import group_sum

small_int = st.integers(min_value=1, max_value=10)

if torch.cuda.is_available():
    st_device = st.sampled_from(["cpu", "cuda"])
else:
    st_device = st.just("cpu")


# Test the basic Resampling functionalities ==============================================


@given(
    n_patients=small_int,
    nbootstraps=small_int,
    n_samples=small_int,
    n_intervals=small_int,
    use_cuda=st.booleans(),
)
def test_resampling_shapes(
    *,
    n_patients: int,
    nbootstraps: int,
    n_samples: int,
    n_intervals: int,
    use_cuda: bool,
):
    """Tests the Resampling constructor."""
    indices = torch.randint(0, n_patients, size=(nbootstraps, n_samples))
    patients = torch.randint(0, n_patients, size=(n_intervals,))

    # Make sure that we "use" all the patients:
    patients[0] = n_patients - 1

    if use_cuda and torch.cuda.is_available():
        indices = indices.cuda()
        patients = patients.cuda()

    res = Resampling(indices=indices, patient=patients)

    assert res.patient_weights.shape == (nbootstraps, n_patients)
    assert res.patient_counts.shape == (nbootstraps, n_patients)
    assert res.patient_weights.dtype in (torch.float32, torch.float64)
    assert res.patient_counts.dtype == torch.int64

    assert res.interval_weights.shape == (nbootstraps, n_intervals)
    assert res.interval_weights.dtype in (torch.float32, torch.float64)


@given(
    n_patients=small_int,
    nbootstraps=small_int,
    n_samples=small_int,
    n_intervals=small_int,
    use_cuda=st.booleans(),
)
def test_resampling_single(
    *,
    n_patients: int,
    nbootstraps: int,
    n_samples: int,
    n_intervals: int,
    use_cuda: bool,
):
    """Checks that 'resampling' a single sample works as expected."""
    unique_patient = torch.randint(0, n_patients, size=(1,)).item()
    # indices is constant: we only care about the unique patient above!
    indices = unique_patient * torch.ones(
        nbootstraps, n_samples, dtype=torch.int64
    )
    patients = torch.randint(0, n_patients, size=(n_intervals,))

    # Make sure that we "use" all the patients:
    patients[0] = n_patients - 1

    if use_cuda and torch.cuda.is_available():
        indices = indices.cuda()
        patients = patients.cuda()

    res = Resampling(indices=indices, patient=patients)

    # Expected patient weights: zeros, except for the unique patient
    # that get a weight that is equal to n_samples.
    expected_patient_weights = torch.zeros(nbootstraps, n_patients)
    expected_patient_weights[:, unique_patient] = n_samples

    # Expected interval weights: zeros, except for the intervals that
    # are associated to the unique patient.
    expected_interval_weights = torch.zeros(nbootstraps, n_intervals)
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
        # N.B.: patient ids must be a dense 0..n_patients-1 range, with no
        # gaps, since n_patients is inferred as max(patient) + 1.
        patient = np.array([0, 1, 2, 3, 4, 0, 1, 0, 5, 6])
        n_patients = 7
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
    assert torch.allclose(
        res.patient_weights, torch.ones(1, n_patients, device=device)
    )
    assert torch.allclose(
        res.interval_weights, torch.ones(1, len(dataset.stop), device=device)
    )


@given(
    nbootstraps=small_int,
    batchsize=small_int,
    use_patient=st.booleans(),
    device=st_device,
)
def test_bootstraps_simple(
    nbootstraps: int, batchsize: int, use_patient: bool, device: str
):
    """Tests the bootstrap method on a simple handcrafted dataset."""
    dataset, n_patients = simple_dataset(use_patient, device)

    boots = list(
        dataset.bootstraps(nbootstraps=nbootstraps, batchsize=batchsize)
    )
    assert len(boots) == ceil(nbootstraps / batchsize)
    assert sum([len(b) for b in boots]) == nbootstraps

    for it, res in enumerate(boots):

        print("it:", it, "len(res):", len(res))
        if it < len(boots) - 1:
            b = batchsize
        else:
            b = nbootstraps % batchsize
            if b == 0:
                b = batchsize

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
    nbootstraps=st.one_of(small_int, st.just(1000), st.just(10000)),
    device=st_device,
)
def test_bootstraps_stratification_1(
    n_groups: int, n_intervals: int, nbootstraps: int, device: str
):
    """Checks that stratification works as expected."""

    # Stop, event and covariates don't really matter here:
    rng = np.random.default_rng()
    stop = rng.integers(1, 10, size=(n_intervals,))
    event = rng.integers(0, 2, size=(n_intervals,))
    covariates = np.zeros((n_intervals, 1))

    # Batch is a random vector that defines at most n_groups separate groups:
    batch = rng.integers(0, n_groups, size=(n_intervals,))

    # Wrap the data in a TorchSurvivalDataset object:
    dataset = SurvivalDataset(
        stop=stop,
        event=event,
        covariates=covariates,
        batch=batch,
    )
    dataset = dataset.to_torch(device).sort().count_deaths()

    # Retrieve our bootstraps in a single Resampling object:
    boots = next(
        dataset.bootstraps(nbootstraps=nbootstraps, batchsize=nbootstraps)
    )

    # Simple check on the shapes, as in test_bootstraps_simple():
    assert boots.patient_weights.shape == (nbootstraps, n_intervals)

    # Check that the total number of samples per group is preserved ----------------------

    batch = torch.from_numpy(batch).to(device=device)
    # Compute the original number of patients per strata:
    weight_per_strata = (
        torch.bincount(batch, minlength=n_groups)
        .tile((nbootstraps, 1))
        .float()
    )

    # Compute the total weight per strata:
    new_weight_per_strata = group_sum(
        values=boots.patient_weights,
        groups=batch.view(1, -1).tile((nbootstraps, 1)),
        output_size=n_groups,
    )

    assert torch.allclose(weight_per_strata, new_weight_per_strata)

    # Check that every patient has an equal probability of being sampled -----------------
    if nbootstraps >= 1000:
        # each cell of boots.patient_weights is a random variable with expected
        # mean value of 1 and finite variance that depends on the number of patients
        # per group. (A patient that is alone is always going to get picked, with
        # weight=1, whereas a patient in a more populous groups may experience
        # a wider range of fortunes.)
        # In any case, according to the central limit theorem,
        # we expect that the average empirical probas over nbootstraps will
        # be equal to 1 + Cst * N(0,1) / sqrt(nbootstraps)
        probas = boots.patient_weights.mean(dim=0)  # (n_intervals,)

        # We can reasonably expect that Cst ~ 3, and ask with >99% certainty
        # that the error falls in the confidence interval +- 3/sqrt(n_boostraps):
        assert torch.allclose(
            probas, torch.ones_like(probas), atol=10 / sqrt(nbootstraps)
        )


# Test CoxPHSurvivalAnalysis.bootstrap() end-to-end ======================================


def _simple_coxph_arrays():
    """Raw numpy arrays for a small, well-conditioned simulated CoxPH dataset.

    Uses survivalGPU's simualator to test bootstrap resemple
    """
    sex = ConstantCovariate(
        name="sex", values=[0, 1], weights=[1, 1], coef=0.7
    )
    biomarker = ConstantCovariate(
        name="biomarker", values=[0, 1, 2], weights=[1, 1, 1], coef=0.4
    )
    df = simulate_dataset(
        max_time=50,
        n_patients=40,
        list_covariates=[sex, biomarker],
        compress=False,
        seed=0,
    )
    covariates = df[["sex", "biomarker"]].to_numpy(dtype=np.float64)
    start = df["start"].to_numpy(dtype=np.int64)
    stop = df["stop"].to_numpy(dtype=np.int64)
    event = df["events"].to_numpy(dtype=np.int64)
    patient = df["patients"].to_numpy(dtype=np.int64)
    return covariates, stop, start, event, patient


def test_coxph_bootstrap_uneven_batchsize_does_not_crash():
    """nbootstraps % batchsize != 0 must not crash.

    Regression test: the bootstrap loop used to do
    `torch.stack(bootstrap_coef).view(...)`, which requires every chunk to
    have the same size -- but the last chunk is smaller than the others
    whenever nbootstraps isn't a multiple of batchsize.
    """
    covariates, stop, start, event, patient = _simple_coxph_arrays()
    n_covariates = covariates.shape[1]

    model = CoxPHSurvivalAnalysis(nbootstraps=6, batchsize=4)
    model.fit(
        covariates=covariates,
        stop=stop,
        start=start,
        event=event,
        patient=patient,
    )

    assert model.bootstrap_coef_.shape == (6, 1, n_covariates)
    assert np.all(np.isfinite(model.bootstrap_coef_))


def test_coxph_bootstrap_index_reuse_gives_identical_coef():
    """Replaying the same index chunks reproduces the same bootstrap_coef_."""
    covariates, stop, start, event, patient = _simple_coxph_arrays()

    model1 = CoxPHSurvivalAnalysis(nbootstraps=5, batchsize=2)
    model1.fit(
        covariates=covariates,
        stop=stop,
        start=start,
        event=event,
        patient=patient,
        keep_bootstrap_indices=True,
    )
    saved_indices = model1.bootstrap_indices_

    model2 = CoxPHSurvivalAnalysis(nbootstraps=5, batchsize=2)
    model2.fit(
        covariates=covariates,
        stop=stop,
        start=start,
        event=event,
        patient=patient,
        bootstrap_indices=saved_indices,
    )

    assert np.allclose(model1.bootstrap_coef_, model2.bootstrap_coef_)

    # Contrast: two independently-drawn (non-shared) fits should differ --
    # the space of possible resamples is large enough that a coincidental
    # exact match is negligible.
    model3 = CoxPHSurvivalAnalysis(nbootstraps=5, batchsize=2)
    model3.fit(
        covariates=covariates,
        stop=stop,
        start=start,
        event=event,
        patient=patient,
    )

    assert not np.allclose(model1.bootstrap_coef_, model3.bootstrap_coef_)


def test_coxph_bootstrap_indices_are_patient_sized_not_interval_sized():
    """`bootstrap_indices_` chunks must be (batchsize, n_patients), not (batchsize, n_intervals).

    Guards against accidentally reverting to storing full `Resampling`
    objects (which hold (B, n_intervals) tensors) instead of the compact,
    pre-expansion (B, n_patients) index tensors -- the whole point of the
    `bootstrap_indices()`/`bootstrap()` split is to keep this cheap.
    """
    covariates, stop, start, event, patient = _simple_coxph_arrays()
    n_patients = len(np.unique(patient))
    batchsize = 4

    model = CoxPHSurvivalAnalysis(nbootstraps=6, batchsize=batchsize)
    model.fit(
        covariates=covariates,
        stop=stop,
        start=start,
        event=event,
        patient=patient,
        keep_bootstrap_indices=True,
    )

    assert model.bootstrap_indices_[0].shape == (batchsize, n_patients)
    assert model.bootstrap_indices_[0].shape != (batchsize, len(stop))
