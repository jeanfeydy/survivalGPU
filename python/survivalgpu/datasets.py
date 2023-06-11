"""Implements the logic related to dataset management, on the CPU side.

We provide:

- SurvivalDataset, a class that encapsulates observation intervals, covariates and doses.
  It performs minor checks (e.g. that observation intervals for a patient do not overlap)
  and provides a display routine. All expensive computations are performed once the
  dataset has been sent to the GPU and turned into a TorchSurvivalDataset object 
  with the .to_torch() method.

- load_drugs(), a method that generates synthetic data that we use in our tutorials
  and benchmarks.
"""


import numpy as np
import torch

from matplotlib import pyplot as plt
from .typecheck import typecheck, Optional, Callable, Union
from .typecheck import Int, Real
from .typecheck import UInt8Array, Int64Array, Float64Array
from .typecheck import TorchDevice

from .torch_datasets import TorchSurvivalDataset


def contains_duplicates(X):
    seen = set()
    seen_add = seen.add
    for x in X:
        if x in seen or seen_add(x):
            return True
    return False


class SurvivalDataset:
    """A dataset for survival analysis.

    We observe I intervals:
    - if `patient is None`, we assume that each patient is observed with a single interval,
      i.e. patient = [0, 1, ..., I-1].
    - if `patient` is provided as an integer array of shape (I,),
      we assume that the i-th interval correspond to the `patient[i]`-th patient.

    For each interval, we may also observe C covariates: these may correspond to
    constant features if the interval is observed for a single patient, or to
    time-varying features if multiple intervals are used to describe a single patient.

    Finally, we may observe D drug intakes at arbitrary time points.
    These are represented using:
    - `dose`: the dose of the drug at time `dose_time`.
    - `dose_time`: the time at which the dose was taken.
    - `dose_patient`: the patient to which the dose belongs, as an integer label.
    - `dose_drug`: the drug to which the dose belongs, as an integer label.

    Please note that the optional "strata" and "batch" attributes can be used to
    split the dataset into multiple groups: these arrays of shape
    (n_patients,) contain group labels for each patient.
    More precisely:
    - batches are used to split the dataset into multiple disjoint subsets.
      These do not interact with each other: we train a separate model for each batch.
    - within a batch, we may split the dataset into multiple strata.
      These sub-groups contribute to separate terms in e.g. the Cox PH likelihood.
      Patients that belong to one strata are never compared to patients that belong
      to another strata, but a trained model will try to get the best possible
      performance on all strata that belong to a batch.

    Attributes:
        stop (int64 (I,) array): the end time of each interval.
        start (int64 (I,) array): the start time of each interval.
        event (int64 (I,) array): the event type at the end of each interval.
        patient (int64 (I,) array): the patient to which each interval belongs.
        strata (int64 (P,) array): the strata to which each patient belongs.
        batch (int64 (P,) array): the batch to which each patient belongs.
        covariates (float64 (I,C) array): the covariates of each interval.
        dose (float64 (D,) array): the dose of each drug.
        dose_time (int64 (D,) array): the time at which each dose was taken.
        dose_patient (int64 (D,) array): the patient to which each dose belongs.
        dose_drug (int64 (D,) array): the drug to which each dose belongs.
    """

    @typecheck
    def __init__(
        self,
        *,
        stop: Int64Array["intervals"],
        start: Optional[Int64Array["intervals"]] = None,
        event: Optional[Int64Array["intervals"]] = None,
        patient: Optional[Int64Array["intervals"]] = None,
        strata: Optional[Int64Array["patients"]] = None,
        batch: Optional[Int64Array["patients"]] = None,
        covariates: Optional[Float64Array["intervals covariates"]] = None,
        dose: Optional[Float64Array["doses"]] = None,
        dose_time: Optional[Int64Array["doses"]] = None,
        dose_patient: Optional[Int64Array["doses"]] = None,
        dose_drug: Optional[Int64Array["doses"]] = None,
    ):
        if covariates is None and dose is None:
            raise ValueError(
                "At least one of `covariates` and `dose` must be provided."
            )

        if any(x is not None for x in (dose, dose_time, dose_patient, dose_drug)):
            if not all(x is not None for x in (dose, dose_time, dose_patient)):
                raise ValueError(
                    "If any of `dose`, `dose_time`, `dose_patient` or `dose_drug` "
                    "is provided, then `dose`, `dose_time` and `dose_patient` must all be provided."
                )

        # Checks for start ---------------------------------------------------------------
        # Default value for start is 0: all intervals start at time 0.
        if start is None:
            start = np.zeros_like(stop)

        # Check that the intervals are )start < stop].
        if np.any(start >= stop):
            raise ValueError("Start times must be < stop times.")

        # Checks for event ---------------------------------------------------------------
        # Default value for event is 1: all intervals correspond to death, without censoring.
        if event is None:
            event = np.ones_like(stop)
        if np.any((event != 0) & (event != 1)):
            raise ValueError("Event values must be 0 (survival) or 1 (death).")

        # Checks for patient -------------------------------------------------------------
        # Default value for patient is [0, 1, 2, ...]: we observe one interval per patient.
        if patient is None:
            patient = np.arange(stop.shape[0])
        elif contains_duplicates(patient):
            # If some patients are observed with several intervals,
            # we must check that they do not overlap.

            # We sort the intervals by patient, then by start time.
            order = np.lexsort((start, patient))
            sorted_patient = patient[order]
            sorted_start = start[order]
            sorted_stop = stop[order]

            # We have already checked that (start < stop).
            # Now, the sorted intervals reads something like:
            # Patient Start  Stop
            # 0       0      3
            # 0       3      5
            # 1       0      6
            # 1       2      4
            # We write a test that compares the start times of the "next" interval
            # ([3, 0, 2]) with the stop time of the current interval ([3, 5, 6]),
            # while also checking that the patient is the same between the two lines
            # ([True, False, True]).
            # Our condition to detect overlapping intervals is:
            # - There is a cell where the start time is < the stop time.
            #   In our example: [False, True, True]
            overlap = sorted_start[1:] < sorted_stop[:-1]
            # - The corresponding patient is the same.
            #   In our example: [True, False, True]
            same_patient = sorted_patient[1:] == sorted_patient[:-1]
            # In our example: [False, False, True] -> we raise an error.
            if np.any(overlap & same_patient):
                raise ValueError("Overlapping intervals for the same patient.")

            # We must also check that there is at most one event per patient,
            # and that the event is the last interval for that patient.
            sorted_event = event[order]
            # In our example: [False, True, False]
            last_per_patient = sorted_patient[1:] != sorted_patient[:-1]
            # The last interval of the list always counts as the last interval
            # of the last patient:
            last_per_patient = np.concatenate((last_per_patient, [True]))

            if np.any(sorted_event & ~last_per_patient):
                raise ValueError(
                    "Events can only occur for the last interval of a patient."
                )

        # TODO: decide what to do with missing values in the covariates.

        # Please note that we do not check what happens with the doses:
        # we allow for dose_times < start, dose_times > stop, etc.

        self.stop = stop
        self.start = start
        self.event = event
        self.patient = patient
        self.covariates = covariates
        self.dose = dose
        self.dose_time = dose_time
        self.dose_patient = dose_patient
        self.dose_drug = dose_drug

        # N.B.: we check for strata and batch once "self.patient" has been set.
        # Checks for strata -------------------------------------------------------------
        # Default value for strata is [0, 0, 0, ...]: all patients belong to the same stratum.
        if strata is None:
            strata = np.zeros((self.n_patients,), dtype=np.int64)

        if strata.shape != (self.n_patients,):
            raise ValueError(
                "Strata must be a vector of length n_patients = max(patient) + 1. "
                f"Got {strata.shape} instead of {self.n_patients}."
            )

        # Checks for batch --------------------------------------------------------------
        # Default value for batch is [0, 0, 0, ...]: all patients belong to the same batch.
        if batch is None:
            batch = np.zeros((self.n_patients,), dtype=np.int64)

        if batch.shape != (self.n_patients,):
            raise ValueError(
                "Batch must be a vector of length n_patients = max(patient) + 1. "
                f"Got {batch.shape} instead of {self.n_patients}."
            )

        self.strata = strata
        self.batch = batch

    @typecheck
    def to_torch(self, device: TorchDevice) -> TorchSurvivalDataset:
        """Converts the dataset to a TorchSurvivalDataset.

        This new representation is meant to be used internally by the library,
        with e.g. faster sorting on the GPU.
        """

        def to_int(x):
            return torch.tensor(x, dtype=torch.int64, device=device)

        def to_float(x):
            return torch.tensor(x, dtype=torch.float32, device=device)

        return TorchSurvivalDataset(
            stop=to_int(self.stop),
            start=to_int(self.start),
            event=to_int(self.event),
            patient=to_int(self.patient),
            strata=to_int(self.strata),
            batch=to_int(self.batch),
            covariates=to_float(self.covariates),
        )

    @property
    @typecheck
    def n_patients(self) -> int:
        """Number of patients that are referenced in the dataset."""
        return int(np.max(self.patient) + 1)

    @property
    @typecheck
    def n_intervals(self) -> int:
        """Number of intervals that are referenced in the dataset."""
        return self.stop.shape[0]

    @property
    @typecheck
    def n_covariates(self) -> int:
        """Number of covariates that are referenced in the dataset."""
        return 0 if self.covariates is None else self.covariates.shape[1]

    @property
    @typecheck
    def n_drugs(self) -> int:
        """Number of drugs that are referenced in the dataset."""
        if self.dose_drug is None:
            return 0
        elif len(self.dose_drug) == 0:
            return 0
        else:
            return int(np.max(self.dose_drug) + 1)

    @property
    @typecheck
    def n_doses(self) -> int:
        """Number of drug doses that are referenced in the dataset."""
        return 0 if self.dose is None else self.dose.shape[0]

    @property
    @typecheck
    def min_time(self) -> int:
        """First time value in the dataset."""
        tmin = np.min(self.start)
        if self.dose_time is not None and len(self.dose_time) > 0:
            tmin = min(tmin, np.min(self.dose_time))
        return int(tmin)

    @property
    @typecheck
    def max_time(self) -> int:
        """Last time value in the dataset."""
        tmax = np.max(self.stop)
        if self.dose_time is not None and len(self.dose_time) > 0:
            tmax = max(tmax, np.max(self.dose_time))
        return int(tmax)

    @typecheck
    def to_img(self, pixel_size: int = 1) -> UInt8Array["H W 3"]:
        """Return a graphical representation of the dataset as a (H, W, 3) RGB uint8 array."""
        total_covariates = self.n_covariates + self.n_drugs
        min_time = self.min_time
        max_time = self.max_time

        margin = 4

        # colors = ["Purples", "Blues", "Greens", "Oranges", "Reds"]
        colors = ["Blues"]
        covar_maps = [
            plt.get_cmap(colors[i % len(colors)]) for i in range(self.n_covariates)
        ]
        drug_maps = [plt.get_cmap(colors[i % len(colors)]) for i in range(self.n_drugs)]

        # Normalize the covariates.
        if self.covariates is None:
            covariates = None
        else:
            covariates = self.covariates.copy()
            covariates -= covariates.min(axis=0)
            max_cov = covariates.max(axis=0)
            max_cov[max_cov == 0] = 1
            covariates /= max_cov

            # covariates >= 0.2 to ensure that we never blend with the white background.
            covariates = 0.2 + 0.8 * covariates

        # The patients are stacked vertically, with a margin of 2 pixels between them.
        # We add an extra column to account for the "death bar" of the last patient:
        img = 255 * np.ones(
            (
                margin + self.n_patients * (total_covariates + margin),
                2 + max_time - min_time,
                3,
            ),
            dtype=np.uint8,
        )

        # Paint the covariates and a gray background for the doses:
        for i in range(self.n_intervals):
            start = 0 if self.start is None else self.start[i]
            stop = self.stop[i]
            patient = i if self.patient is None else self.patient[i]

            offset_y = margin + patient * (total_covariates + margin)
            # The intervals are always of the form )start, stop],
            # so we need to add 1 to the start.
            offset_x = 1 + start - min_time

            # Paint the covariates on )start, stop]:
            for j in range(self.n_covariates):
                covar = covariates[i, j]
                img[offset_y + j, offset_x : 1 + stop - min_time, :] = covar_maps[j](
                    covar, bytes=True
                )[:3]

            # Paint a gray background for the doses on )start, stop]:
            if False:
                img[
                    offset_y + self.n_covariates : offset_y + total_covariates,
                    offset_x : 1 + stop - min_time,
                    :,
                ] = 200

            # Paint the death events as a red line at stop+1:
            # N.B.: self.event is None => event = 1 for all intervals.
            if self.event is None or self.event[i] == 1:
                img[
                    offset_y - 1 : 1 + offset_y + total_covariates,
                    1 + stop - min_time,
                    1:,
                ] = 0

        # Paint the doses:
        for i in range(self.n_doses):
            dose = self.dose[i]
            time = self.dose_time[i]
            patient = self.dose_patient[i]
            # N.B.: self.dose_drug is None => drug = 0 for all doses.
            drug = 0 if self.dose_drug is None else self.dose_drug[i]

            offset_y = (
                margin
                + patient * (total_covariates + margin)
                + self.n_covariates
                + drug
            )
            offset_x = time - min_time

            img[offset_y, offset_x, :] = drug_maps[drug](dose, bytes=True)[:3]

        return np.kron(img, np.ones((pixel_size, pixel_size, 1), dtype=np.uint8))


# Virtual dataset ========================================================================


@typecheck
def consecutive_doses(
    *,
    start: Int,
    stop: Int,
    covariates: Optional[Float64Array["covariates"]],
    dose: Optional[Float64Array["doses"]],
    dose_time: Optional[Int64Array["doses"]],
    dose_drug: Optional[Int64Array["doses"]],
    poison_covariates: Union[Real, Float64Array["covariates"]] = 0.5,
    poison_dose: Real = 0.5,
    poison_time: Int = 1,
):
    """Simple risk model for a drug that kills with two consecutive doses.

    The signature of this function corresponds to what is expected by `load_drugs`.
    Given the medical record of a patient, this function returns a premature death
    at time t if:
    - All the `covariates` are >= the `poison_covariates` threshold (True if `covariates` is None).
    - The patient has received at least two doses of the drug of interest,
      that corresponds to `dose_drug == 0` (the other drugs have no influence on the risk).
    - The patient has received a `dose >= poison_dose` at time t.
    - The previous `dose` received was also `>= poison_dose`, and received in the
      interval `[t - poison_time, t)`.
    """
    # Condition for death: either no covariate, or all covariates >= covariates_threshold.
    if covariates is None:
        at_risk = True
    else:
        at_risk = np.all(covariates >= poison_covariates)

    if dose is None:
        if covariates is None:
            raise ValueError("Either dose or covariates must be provided.")
        return at_risk, stop

    if not at_risk:
        return False, stop

    # We only focus on the first drug:
    doses = dose[dose_drug == 0]
    times = dose_time[dose_drug == 0]

    # Condition for death: at least two doses for the drug of interest
    if not (len(doses) >= 2):
        return False, stop

    # Condition for death: two consecutive doses
    consecutive = times[1:] <= times[:-1] + poison_time

    # Condition for death: the two consecutive doses are above a threshold
    thresh = (doses[:-1] >= poison_dose) & (doses[1:] >= poison_dose)

    # Condition for death: the 2nd dose is observed after the start of the time interval
    after_start = times[1:] > start

    candidates = consecutive & thresh & after_start
    if not np.any(candidates):
        return False, stop
    else:
        premature_stop = times[1:][candidates][0]
        return True, premature_stop


# Random doses, with a simple risk model: -----------------------------------------------
@typecheck
def load_drugs(
    *,
    n_covariates: int = 0,
    n_drugs: int = 1,
    n_patients: int = 1,
    max_duration: int = 1,
    max_offset: int = 0,
    seed: Optional[int] = None,
    risk_model: Callable = consecutive_doses,
) -> SurvivalDataset:
    """Create a virtual dataset for testing using a simple risk model.

    Args:
        n_covariates (int, optional): number of constant covariates. Defaults to 0.
        n_drugs (int, optional): number of drugs to test. Defaults to 1.
        n_patients (int, optional): number of patients. Defaults to 1.
        max_duration (int, optional): max size of the time interval per subject. Defaults to 1.
        max_offset (int, optional): max offset of the time interval window per subject. Defaults to 0.
        seed (int, optional): random seed that can be specified explicitly for the sake
            of reproducibility. Defaults to None: a new seed is generated at each call.
        risk_model (Callable, optional): function that is applied to the covariates
            and drug consumption history of each patient to decide whether or not
            "death" (= the event of interest) happened, and at which time.
            This function that takes as input:
            - start (int): the start of the time interval.
            - stop (int): the suggested end of the time interval.
            - covariates: a float64 (n_covariates,) array of covariates.
            - dose: a float64 (n_doses,) array of doses.
            - dose_time: a int64 (n_doses,) array of dose times.
            - dose_drug: a int64 (n_doses,) array of dose drugs.
            and returns:
            - event: a boolean indicating whether the patient died in this interval.
            - premature_stop: the time at which the patient died, or the end of the interval
                if the patient did not die.
            Defaults to `consecutive_doses`.


    Returns:
        SurvivalDataset: contains arrays that describe a drug consumption dataset.
    """

    rng = np.random.default_rng(seed)

    # Default time windows:
    start = rng.integers(low=0, high=max_offset + 1, size=n_patients)
    stop = start + rng.integers(low=1, high=max_duration + 1, size=n_patients)
    # event = rng.integers(low=0, high=2, size=n_patients)
    event = np.zeros(n_patients, dtype=np.int64)
    patient = np.arange(n_patients)
    if n_covariates > 0:
        covariates = rng.random(size=(n_patients, n_covariates))
    else:
        covariates = None

    # Random doses - we could try more interesting models:
    if n_drugs == 0:
        dose = None
        dose_time = None
        dose_patient = None
        dose_drug = None

    else:
        dose = []
        dose_time = []
        dose_patient = []
        dose_drug = []

        for p in range(n_patients):
            # Generate doses for all drugs using a simple model:
            for d in range(n_drugs):
                t = start[p]
                while True:
                    new_dose = rng.random()
                    t += 1 + rng.integers(10)
                    if t > stop[p]:
                        break
                    dose.append(new_dose)
                    dose_time.append(t)
                    dose_patient.append(p)
                    dose_drug.append(d)

        # Cast the dose descriptions as contiguous arrays:
        dose = np.array(dose, dtype=np.float64)
        dose_time = np.array(dose_time, dtype=np.int64)
        dose_patient = np.array(dose_patient, dtype=np.int64)
        dose_drug = np.array(dose_drug, dtype=np.int64)

        # Generate an index array to slice the doses by patient:
        n_doses = np.bincount(dose_patient, minlength=n_patients)
        assert n_doses.shape == (n_patients,)

        # N.B.: Append a zero to make the slicing easier:
        dose_indices = np.cumsum(n_doses)
        dose_indices = np.concatenate([[0], dose_indices])
        assert dose_indices.shape == (n_patients + 1,)
        assert dose_indices[-1] == len(dose)

    # Use an arbitrary risk model to decide if and when the patient dies:
    for p in range(n_patients):
        if covariates is None:
            covariates_p = None
        else:
            covariates_p = covariates[p]

        if dose is None:
            dose_p = None
            dose_time_p = None
            dose_drug_p = None
        else:
            s, e = dose_indices[p], dose_indices[p + 1]
            dose_p = dose[s:e]
            dose_time_p = dose_time[s:e]
            dose_drug_p = dose_drug[s:e]

        death, t = risk_model(
            start=start[p],
            stop=stop[p],
            covariates=covariates_p,
            dose=dose_p,
            dose_time=dose_time_p,
            dose_drug=dose_drug_p,
        )
        event[p] = death
        stop[p] = t

    return SurvivalDataset(
        start=start,
        stop=stop,
        event=event,
        patient=patient,
        covariates=covariates,
        dose=dose,
        dose_time=dose_time,
        dose_patient=dose_patient,
        dose_drug=dose_drug,
    )


@typecheck
def simple_dataset(
    *,
    n_covariates: int = 0,
    n_patients: int = 1,
    n_batch: int = 1,
    n_strata: int = 1,
    max_duration: int = 1,
    ensure_one_life: bool = False,
    ensure_one_death: bool = False,
    unit_length_intervals: bool = False,
):
    # N.B.: For the sake of simplicity, we assume one interval per patient:
    n_intervals = n_patients

    # Create a minimal random dataset:
    rng = np.random.default_rng()
    stop = rng.integers(low=1, high=1 + max_duration, size=(n_intervals,))
    event = rng.integers(low=0, high=2, size=(n_intervals,))
    batch = rng.integers(low=0, high=n_batch, size=(n_patients,))
    strata = rng.integers(low=0, high=n_strata, size=(n_patients,))
    covariates = rng.normal(loc=0, scale=1, size=(n_intervals, n_covariates))

    if ensure_one_life:
        event[0] = 0

    if ensure_one_death:
        event[-1] = 1

    if unit_length_intervals:
        start = stop - 1
    else:
        start = None

    dataset = SurvivalDataset(
        start=start,
        stop=stop,
        event=event,
        batch=batch,
        strata=strata,
        covariates=covariates,
    )
    return dataset


if __name__ == "__main__":
    import time
    import imageio

    ds = load_drugs(
        n_covariates=1, n_drugs=1, n_patients=20, max_duration=100, max_offset=50
    )
    imageio.imwrite("output_test_dataset.png", ds.to_img(pixel_size=10))

    for n_patients in [10, 100, 1000, 10000]:
        clock = time.time()
        ds = load_drugs(
            n_covariates=2,
            n_drugs=1,
            n_patients=n_patients,
            max_duration=365,
            max_offset=3650,
        )
        print(
            f"Generated a dataset with {n_patients:8,} patients in {time.time() - clock:6.2f} seconds."
        )
