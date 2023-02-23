import numpy as np
from matplotlib import pyplot as plt
from .typecheck import typecheck, Optional, Int64Array, Float64Array


def contains_duplicates(X):
    seen = set()
    seen_add = seen.add
    for x in X:
        if x in seen or seen_add(x):
            return True
    return False


class SurvivalDataset:
    """A dataset for survival analysis.

    Attributes:
        doses (Optional[RealArray["drugs", "patients", "times"]]): doses of drugs.
    """

    @typecheck
    def __init__(
        self,
        stop: Int64Array["intervals"],
        start: Optional[Int64Array["intervals"]] = None,
        event: Optional[Int64Array["intervals"]] = None,
        patient: Optional[Int64Array["intervals"]] = None,
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

    @property
    def n_patients(self) -> int:
        """Number of patients that are referenced in the dataset."""
        return np.max(self.patient) + 1

    @property
    def n_intervals(self) -> int:
        """Number of intervals that are referenced in the dataset."""
        return self.stop.shape[0]

    @property
    def n_covariates(self) -> int:
        """Number of covariates that are referenced in the dataset."""
        return 0 if self.covariates is None else self.covariates.shape[1]

    @property
    def n_drugs(self) -> int:
        """Number of drugs that are referenced in the dataset."""
        return 0 if self.dose_drug is None else np.max(self.dose_drug) + 1

    @property
    def n_doses(self) -> int:
        """Number of drug doses that are referenced in the dataset."""
        return 0 if self.dose is None else self.dose.shape[0]

    @property
    def min_time(self) -> int:
        """First time value in the dataset."""
        tmin = np.min(self.start)
        if self.dose_time is not None:
            tmin = min(tmin, np.min(self.dose_time))
        return tmin

    @property
    def max_time(self) -> int:
        """Last time value in the dataset."""
        tmax = np.max(self.stop)
        if self.dose_time is not None:
            tmax = max(tmax, np.max(self.dose_time))
        return tmax

    def to_img(self, pixel_size: int=1):
        """Return a graphical representation of the dataset as a (H, W, 3) RGB uint8 array."""
        total_covariates = self.n_covariates + self.n_drugs
        min_time = self.min_time
        max_time = self.max_time

        margin = 2

        colors = ["Purples", "Blues", "Greens", "Oranges", "Reds"][::-1]
        covar_maps = [
            plt.get_cmap(colors[i % len(colors)]) for i in range(self.n_covariates)
        ]
        drug_maps = [plt.get_cmap(colors[i % len(colors)]) for i in range(self.n_drugs)]

        # Normalize the covariates.
        covariates = self.covariates.copy()
        covariates -= covariates.min(axis=0)
        max_cov = covariates.max(axis=0)
        max_cov[max_cov == 0] = 1
        covariates /= max_cov
        # Rescale the covariates to [0.2, 0.8] to have nice pastel colors:
        covariates = 0.2 + 0.6 * covariates

        # The patients are stacked vertically, with a margin of 2 pixels between them.
        # We add an extra column to account for the "death bar" of the last patient:
        img = 255 * np.ones(
            (
                self.n_patients * (total_covariates + margin),
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

            offset_y = patient * (total_covariates + margin)
            # The intervals are always of the form )start, stop],
            # so we need to add 1 to the start.
            offset_x = 1 + start - min_time

            # Paint the covariates on )start, stop]:
            for j in range(self.n_covariates):
                covar = covariates[i, j]
                img[offset_y + j, offset_x : 1 + stop - min_time, :] = covar_maps[j](covar, bytes=True)[
                    :3
                ]

            # Paint a gray background for the doses on )start, stop]:
            img[
                offset_y + self.n_covariates : offset_y + total_covariates,
                offset_x : 1 + stop - min_time,
                :,
            ] = 200

            # Paint the death events as a black line at stop+1:
            # N.B.: self.event is None => event = 1 for all intervals.
            if self.event is None or self.event[i] == 1:
                img[offset_y : offset_y + total_covariates, 1 + stop - min_time, :] = 0

        # Paint the doses:
        for i in range(self.n_doses):
            dose = self.dose[i]
            time = self.dose_time[i]
            patient = self.dose_patient[i]
            # N.B.: self.dose_drug is None => drug = 0 for all doses.
            drug = 0 if self.dose_drug is None else self.dose_drug[i]

            offset_y = patient * (total_covariates + margin) + self.n_covariates + drug
            offset_x = time - min_time

            img[offset_y, offset_x, :] = drug_maps[drug](dose, bytes=True)[:3]

        return np.kron(img, np.ones((pixel_size, pixel_size, 1), dtype=np.uint8))


# Virtual dataset ========================================================================

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
) -> SurvivalDataset:
    """Create a virtual dataset for testing using a simple risk model.

    Args:
        n_covariates (int, optional): number of constant covariates. Defaults to 0.
        n_drugs (int, optional): number of drugs to test. Defaults to 1.
        n_patients (int, optional): number of patients. Defaults to 1.
        n_times (int, optional): number of sampling times. Defaults to 1.

    Returns:
        SurvivalDataset: contains arrays that describe a drug consumption dataset with:
            - doses: a (Drugs, Patients, Times) float32 Tensor.
            - times: a (Times,) int32 Tensor with the sampling times.
            - events: a (Patients, Times) int32 Tensor whose values are equal to
                0 if everything is all right, i.e. the patient is still "alive",
                1 if the patient is "dying" at this exact moment,
                2+ if this consumption happened after the event of interest,
                   and should therefore be removed from the survival analysis.

              Note that we deliberately include drug consumption data "after death"
              as these may be relevant to a permutation test.
    """

    rng = np.random.default_rng(seed)

    # Default time windows:
    start = rng.integers(low=0, high=max_offset + 1, size=n_patients)
    stop = start + rng.integers(low=1, high=max_duration + 1, size=n_patients)
    event = rng.integers(low=0, high=2, size=n_patients)
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
            for d in range(n_drugs):
                t = start[p]
                while True:
                    t += 1 + rng.poisson(4)
                    if t > stop[p]:
                        break
                    dose.append(rng.random())
                    dose_time.append(t)
                    dose_patient.append(p)
                    dose_drug.append(d)

        # Cast the dose descriptions as contiguous arrays:
        dose = np.array(dose, dtype=np.float64)
        dose_time = np.array(dose_time, dtype=np.int64)
        dose_patient = np.array(dose_patient, dtype=np.int64)
        dose_drug = np.array(dose_drug, dtype=np.int64)

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


if __name__ == "__main__":
    import imageio

    ds = load_drugs(
        n_covariates=10, n_drugs=5, n_patients=10, max_duration=100, max_offset=50
    )
    imageio.imwrite("output_test_dataset.png", ds.to_img(pixel_size=10))
