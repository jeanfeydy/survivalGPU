import numpy as np
import imageio
from matplotlib import pyplot as plt
from typing import NamedTuple, Optional, Tuple
from .typing import typecheck, Int64Array, Float64Array


class SurvivalDataset:
    """A dataset for survival analysis.

    Attributes:
        doses (Optional[RealArray["drugs", "patients", "times"]]): doses of drugs.
    """

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
        return np.max(self.patient) - 1

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
        return 0 if self.dose is None else np.max(self.dose_drug) - 1

    @property
    def n_doses(self) -> int:
        """Number of drug doses that are referenced in the dataset."""
        return 0 if self.dose is None else self.dose.shape[0]

    @property
    def min_time(self) -> int:
        """First time value in the dataset."""
        return 0 if self.start is None else np.min(self.start)

    @property
    def max_time(self) -> int:
        """Last time value in the dataset."""
        return np.max(self.stop)

    def to_img(self, path: str):
        """Save the dataset as a png file."""
        total_covariates = self.n_covariates + self.n_drugs
        min_time = self.min_time
        max_time = self.max_time
        
        margin=2

        colors = ['Purples', 'Blues', 'Greens', 'Oranges', 'Reds']
        covar_maps = [plt.get_cmap(colors[i % len(colors)]) for i in range(self.n_covariates)]
        drug_maps = [plt.get_cmap(colors[i % len(colors)]) for i in range(self.n_drugs)]

        # Normalize the covariates.
        covariates = self.covariates.copy()
        covariates -= covariates.min(axis=0)
        max_cov = covariates.max(axis=0)
        max_cov[max_cov == 0] = 1
        covariates /= max_cov


        # The patients are stacked vertically, with a margin of 2 pixels between them.
        # We add an extra column to account for the "death bar" of the last patient:
        img = np.ones((self.n_patients * (total_covariates + margin), 1 + max_time - min_time, 3,), dtype=np.uint8)

        # Paint the covariates and a gray background for the doses:
        for i in range(self.n_intervals):
            start = 0 if self.start is None else self.start[i]
            stop = self.stop[i]
            patient = i if self.patient is None else self.patient[i]

            offset_y = patient * (total_covariates + margin)
            # The intervals are always of the form )start, stop], 
            # so we need to add 1 to the start.
            offset_x = 1 + start - min_time

            # Paint the covariates:
            for j in range(self.n_covariates):
                covar = covariates[i, j]
                img[offset_y + j, offset_x:stop - min_time, :] = covar_maps[j](covar)[:3]

            # Paint a gray background for the doses:
            img[offset_y+self.n_covariates:offset_y+total_covariates, offset_x:stop - min_time, :] = 200

            # Paint the death events as a black line:
            # N.B.: self.event is None => event = 1 for all intervals.
            if self.event is None or self.event[i] == 1:
                img[offset_y:offset_y + total_covariates, 1 + stop - min_time, :] = 0

        # Paint the doses:
        for i in range(self.n_doses):
            dose = self.dose[i]
            time = self.dose_time[i]
            patient = self.dose_patient[i]
            # N.B.: self.dose_drug is None => drug = 0 for all doses.
            drug = 0 if self.dose_drug is None else self.dose_drug[i]

            offset_y = patient * (total_covariates + margin) + self.n_covariates + drug
            offset_x = 1 + time - min_time

            img[offset_y, offset_x, :] = drug_maps[drug](dose)[:3]

        imageio.imwrite(img, path)


# Virtual dataset ========================================================================

# Random doses, with a simple risk model: -----------------------------------------------
@typecheck
def load_drugs(
    *,
    n_drugs: int = 1,
    n_patients: int = 1,
    n_times: int = 1,
    device: str = "cpu",
) -> SurvivalDataset:
    """Create a virtual dataset for testing using a simple risk model.

    Args:
        n_drugs (int, optional): number of drugs to test. Defaults to 1.
        n_patients (int, optional): number of patients. Defaults to 1.
        n_times (int, optional): number of sampling times. Defaults to 1.
        n_device (str, optional): device where the data should be stored. Defaults to "cpu".

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

    # Sampling times = [0, 1, ..., Times-1]:
    sampling_times = torch.arange(n_times, dtype=torch.int32, device=device)

    # Random doses - we could try more interesting models:
    doses = 1.0 * (torch.rand(n_drugs, n_patients, n_times, device=device) < 0.5)
    assert doses.shape == (n_drugs, n_patients, n_times)

    # events = 0  if everything is all right,
    #          1  if the patient is dying at this exact moment
    #          2+ if it is too late
    events = torch.zeros(n_patients, n_times, dtype=torch.int32, device=device)

    # Simple risk model: two consecutive doses of drug 0 -> 80% death.
    danger = doses[0, :, :]  # (Patients, Times)
    events[:, 2:] = torch.logical_and(danger[:, :-2] > 0.5, danger[:, 1:-1] > 0.5)
    events[:] = torch.logical_and(
        events, torch.rand(n_patients, n_times, device=device) > 0.2
    )
    # Event = 1 only for the first time.
    # All subsequent cells have value >= 2 and should be discarded
    events[:, 1:] += 2 * events[:, :-1].cumsum(axis=-1)
    assert events.shape == (n_patients, n_times)

    return SurvivalDataset(
        doses=doses,
        start=sampling_times,
        stop=sampling_times + 1,
        event=events,
    )
