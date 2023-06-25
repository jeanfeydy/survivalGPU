"""Implements the compute-intensive parts of dataset management using PyTorch.

We provide a TorchSurvivalDataset object with methods that implement:
- The re-ordering of data by lexicographical order on (batch > strata > stop > event).
- A re-scaling of the input covariates, for the sake of numerical stability.
- The pre-computation of basic statistics on at-risk sets.
- The removal of groups that do not contribute to the CoxPH objective
  (e.g. because they do not include any death event).

"""


import torch
import numpy as np
from matplotlib import pyplot as plt
from .typecheck import (
    typecheck,
    Optional,
    Callable,
    Union,
    List,
    Tuple,
    TorchDevice,
    Literal,
)
from .typecheck import Int, Real
from .typecheck import Int64Tensor, Float32Tensor
from .bootstrap import Resampling


@typecheck
def torch_lexsort(a: Int64Tensor["keys indices"]) -> Int64Tensor["indices"]:
    """PyTorch implementation of np.lexsort.

    N.B.: This function relies on the fact that torch.unique implements
    a lexicographical sort in the background. This is not a fully documented
    behaviour, so it might break in the future: testing is important!
    """
    # To be consistent with numpy, we flip the keys (sort by last row first):
    a_unq, inv = torch.unique(a.flip(0), dim=-1, sorted=True, return_inverse=True)
    return torch.argsort(inv)


class TorchSurvivalDataset:
    """Object that holds data for the compute-intensive parts of the CoxPH solvers.

    Attributes:
        stop ((I,) int64 Tensor): stop times for the intervals.
        start ((I,) int64 Tensor): start times for the intervals.
        event ((I,) int64 Tensor): event indicators for the intervals.
        patient ((I,) int64 Tensor): patient indices for the intervals.
        strata ((P,) int64 Tensor): strata indices for the patients.
        batch ((P,) int64 Tensor): batch indices for the patients.
        covariates ((I,D) float32 Tensor): covariates for the intervals.
        batch_intervals ((I,) int64 Tensor): batch indices for the intervals.
        strata_intervals ((I,) int64 Tensor): strata indices for the intervals.
        is_sorted (bool): whether the data is sorted by lexicographical order on (batch > strata > stop > event).
        device (torch.device or str): torch device (cpu, cuda...) where the data is stored.
        n_patients (int): number of patients that are referenced in the dataset.
        n_batch (int): number of batches that are referenced in the dataset.
        n_covariates (int): number of covariates that are referenced in the dataset.
        group ((I,) int64 Tensor): group indices for the intervals, with values in [0, n_groups-1].
          These correspond to the T unique values of (batch, strata, stop) that are defined for the intervals.
        n_groups (int): number of independent groups defined above.
        unique_groups ((3,n_groups) int64 Tensor): (batch, strata,  stop) values for the groups that are referenced in the dataset.
          These correspond to the group indices defined above.
        tied_deaths ((n_groups,) int64 Tensor): number of tied deaths for each group.
    """

    @typecheck
    def __init__(
        self,
        *,
        stop: Int64Tensor["intervals"],
        start: Int64Tensor["intervals"],
        event: Int64Tensor["intervals"],
        patient: Int64Tensor["intervals"],
        strata: Int64Tensor["patients"],
        batch: Int64Tensor["patients"],
        covariates: Float32Tensor["intervals covariates"],
    ):
        self.stop = stop
        self.start = start
        self.event = event
        self.patient = patient
        self.strata = strata
        self.batch = batch
        self.covariates = covariates

        self.batch_intervals = self.batch[self.patient]
        self.strata_intervals = self.strata[self.patient]

        assert self.batch_intervals.shape == self.stop.shape
        assert self.strata_intervals.shape == self.stop.shape

        self.is_sorted = False

    @property
    @typecheck
    def device(self) -> TorchDevice:
        """Torch device (cpu, cuda...) where the data is stored."""
        return self.event.device

    @property
    @typecheck
    def n_patients(self) -> int:
        """Number of patients that are referenced in the dataset."""
        return int(self.patient.max() + 1)

    @property
    @typecheck
    def n_intervals(self) -> int:
        """Number of intervals that are referenced in the dataset."""
        return self.stop.shape[0]

    @property
    @typecheck
    def n_batch(self) -> int:
        """Number of batches that are referenced in the dataset."""
        return int(self.batch.max() + 1)

    @property
    @typecheck
    def n_covariates(self) -> int:
        """Number of covariates that are referenced in the dataset."""
        return 0 if self.covariates is None else self.covariates.shape[1]

    def sort(self):
        """Re-orders the input arrays by lexicographical order on (batch > strata > stop > event)."""
        # N.B.: the numpy convention is to sort by the last row first.
        keys = torch.stack(
            (self.event, self.stop, self.strata_intervals, self.batch_intervals)
        )
        ind = torch_lexsort(keys)

        # Re-order the arrays of length n_intervals:
        self.stop = self.stop[ind]
        self.start = self.start[ind]
        self.event = self.event[ind]
        self.patient = self.patient[ind]
        self.covariates = self.covariates[ind, :]
        self.batch_intervals = self.batch_intervals[ind]
        self.strata_intervals = self.strata_intervals[ind]
        self.is_sorted = True

        # N.B.: self.strata and self.batch are not re-ordered, because they are
        #       arrays of length n_patients.
        return self

    @typecheck
    def scale(
        self, *, rescale: bool
    ) -> Tuple[Float32Tensor["covariates"], Optional[Float32Tensor["covariates"]]]:
        """Computes the mean and scale (= L1 norm) of each covariate.

        If rescale is False, we simply return the means and None.
        If rescale is True, we return the means, the scales, and normalize the columns of
        self.covariates.
        """
        D = self.n_covariates
        means = self.covariates.mean(dim=0)  # (I,D) -> (D,)
        if not rescale:
            return means, None
        else:
            # For the sake of numerical stability, we may normalize the covariates
            # This should have zero impact on the CoxPH objective:
            self.covariates = self.covariates - means.view(1, D)
            # Use the L1 norms for scale as in the R survival package:
            scales = self.covariates.abs().sum(dim=0)  # (I,D) -> (D,)
            scales[scales == 0] = 1  # Rare case of a constant covariate
            scales = 1 / scales
            self.covariates = self.covariates * scales.view(1, D)
            return means, scales

    def count_deaths(self):
        """Compute basic statistics for each 'at risk' set: labels and number of deaths."""

        # Make sure that (batch > strata > stop > event) is lexicographically sorted:
        assert self.is_sorted, "The dataset must be sorted before counting deaths."

        # Count the number of death times:
        self.unique_groups, self.group = torch.unique_consecutive(
            torch.stack((self.batch_intervals, self.strata_intervals, self.stop)),
            return_inverse=True,
            dim=-1,
        )
        # For instance, for a simple dataset with 1 batch, 1 strata and 3 unique stop times:
        # - self.batch_intervals  = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # - self.strata_intervals = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # - self.stop             = [2, 2, 2, 2, 2, 5, 5, 6, 6, 6],
        # unique_groups is (3,T), e.g.
        # [[0, 0, 0],
        #  [0, 0, 0],
        #  [2, 5, 6]]
        # and self.group is (I,), e.g. [0, 0, 0, 0, 0, 1, 1, 2, 2, 2]
        assert self.unique_groups.shape[0] == 3
        self.n_groups = self.unique_groups.shape[1]  # in our example, T = 3

        # For each group, count the number of (possibly tied) deaths:
        self.tied_deaths = torch.bincount(
            self.group[self.event == 1], minlength=self.n_groups
        )
        # self.tied_deaths is (T,), e.g. [2, 1, 1]
        # if self.event == [0, 0, 0, 1, 1, 0, 1, 0, 0, 1].
        return self

    @typecheck
    def prune(
        self, *, mode: Optional[Literal["unit length", "start zero", "any"]] = None
    ):
        """Filters out the intervals that have no impact on the CoxPH likelihood.

        This may be a common occurence in some datasets where the raw
        "start-stop times" sample every single month or year, including
        times where no event occurs.
        This optimization should have zero impact on the final result of a CoxPH fit.
        We leave it enabled by default - but feel free to comment it out if needed.

        Please note that the means and scales are not affected by this filtering pass,
        as we stick to the conventions of the R survival package.
        """
        # Make sure that (batch > strata > stop > event) is lexicographically sorted:
        assert self.is_sorted, "The dataset must be sorted before pruning."
        # Make sure that the deaths have been counted by looking for the tied_deaths attribute:
        assert hasattr(
            self, "tied_deaths"
        ), "Please apply `.count_deaths()` before pruning."

        if mode is None:
            # Case 1: the intervals are (stop-1, stop].
            #         -> we remove all the times where no one dies.
            if torch.all(self.stop == self.start + 1):
                mode = "unit length"
            # Case 2: the intervals are (0, stop].
            #         -> we remove all the intervals that get censored before the first death.
            #         -> the intervals that get censored right after the n-th death
            #            get their time values updated to the time of the n-th death.
            elif torch.all(self.start == 0):
                mode = "start zero"
            else:
                mode = "any"

        if mode in ["unit length", "start zero"]:
            if mode == "unit length":
                # Filter out the groups that have no deaths:
                mask = self.tied_deaths[self.group] > 0

            elif mode == "start zero":
                if False:
                    # Get an id per independent group:
                    _, batch_strata_group = torch.unique_consecutive(
                        self.unique_groups[0:2],
                        return_inverse=True,
                        dim=-1,
                    )

                # Build a "new_stop" vector that contains the time of the most recent
                # death at any given time for a given (batch, strata, stop),
                # within a given (batch, strata) group:
                assert self.tied_deaths.shape == (self.unique_groups.shape[1],)

                current_stop = 0
                current_batch_strata = self.unique_groups[0:2, 0]
                new_stop = torch.zeros_like(self.unique_groups[2])

                # We loop over the unique values of (batch, strata, stop)
                # and the associated numbers of deaths:
                for i, (b_s_s, deaths) in enumerate(
                    zip(self.unique_groups.T, self.tied_deaths)
                ):
                    # Reset the counter if we are in a new (batch, strata):
                    if not torch.equal(current_batch_strata, b_s_s[0:2]):
                        current_stop = 0
                        current_batch_strata = b_s_s[0:2]
                    # Update the counter if we meet a new "death time":
                    if deaths > 0:
                        current_stop = b_s_s[2]
                    new_stop[i] = current_stop

                if False:
                    print("Pruning!")
                    print("batch: ", self.batch_intervals)
                    print("strata:", self.strata_intervals)
                    print("stop:  ", self.stop)
                    print("death: ", self.event)
                    print("n stop:", new_stop[self.group])

                # Update the stop times to these new values:
                self.stop = new_stop[self.group]

                # If new_stop==0, this means that the interval has been censored
                # before the first death in the current (batch, strata)
                # -> we can prune it safely.
                mask = self.stop > 0

            # Prune out useless lines:
            assert mask.shape == self.stop.shape
            assert mask.shape == self.start.shape
            self.stop = self.stop[mask]
            self.start = self.start[mask]
            self.event = self.event[mask]
            self.patient = self.patient[mask]
            self.covariates = self.covariates[mask, :]
            self.batch_intervals = self.batch_intervals[mask]
            self.strata_intervals = self.strata_intervals[mask]

            # Don't forget to re-count the deaths:
            self.count_deaths()
            # This updates self.group, self.unique_groups, self.n_groups and self.tied_deaths
            # We can now check that the filtering worked as expected:
            assert (self.tied_deaths > 0).all()

        return self

    @typecheck
    def original_sample(self) -> Resampling:
        """Returns a Resampling object that corresponds to the original sample."""
        indices = torch.arange(
            self.n_patients,
            dtype=torch.int64,
            device=self.device,
        )
        indices = indices.view(1, -1)
        return Resampling(
            indices=indices,
            patient=self.patient,
        )

    @typecheck
    def bootstraps(
        self,
        *,
        n_bootstraps: int,
        batch_size: Optional[int],
        stratify: bool = True,
    ) -> List[Resampling]:
        """Returns a list of Resampling objects that correspond to bootstrap samples.

        This method generates bootstrap sampling indices which are similar to:
        indices = torch.randint(P, (B, P)),
        where P is the number of patients and B is the batch size.
        These correspond to a random bootstrap sample, e.g.
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  -> "miracle", we retrieve the original sample!
             [3, 3, 1, 2, 8, 6, 0, 0, 8, 9]]  -> "genuinely random" bootstrap sample

        Note that we draw these indices using a random number generator
        that respects the batch and stratification constraints.
        For instance, with P=10 as above, if:
        - batch  = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
        - strata = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        This defines three groups of patients that should not mix in the bootstrap samples:
        - group=0: [0, 1, 2]
        - group=1: [3, 4, 5, 6]
        - group=2: [7, 8, 9]
        Consequently, with B=2 bootstraps, typical indices would be:
        - indices = [[0, 1, 2 ; 3, 4, 5, 6 ; 7, 8, 9],  -> "miracle", we retrieve the original sample!
                     [0, 0, 1 ; 4, 6, 4, 4 ; 8, 8, 8]]  -> "genuinely random" bootstrap sample

        Note that if stratify=False, we only stratify according to the `batch` vector:
        "group=0" is separated from "group=1" and "group=2", but "group=1" and "group=2"
        may mix together in the bootstrap samples.


        Args:
            n_bootstraps (int): The number of bootstrap samples to generate.
            batch_size (int): The number of bootstrap samples that should be handled
                simultaneously by the CoxPH optimizer.
            stratify (bool): If True, the bootstrap samples are stratified according
                to the values of the `batch` and `strata` vectors.
                Otherwise, we only stratify according to the `batch` vector.
                Defaults to True.
        """
        P = self.n_patients

        # Step 1: create stratifications groups for the sampling =========================
        if stratify:
            strata = torch.stack((self.batch, self.strata), dim=1)
        else:
            strata = self.batch.view(-1, 1)

        # At this point, strata.T may look like - with batch+strata:
        # [[0, 0, 0; 1, 1, 1, 1; 1, 1, 1],
        #  [0, 0, 0; 0, 0, 0, 0; 1, 1, 1]]
        # or, for a more complex example - with batch only:
        # [[0, 3, 1, 0, 1, 1, 2, 0, 3]]
        strata_unique, strata_indices, strata_counts = torch.unique(
            strata,
            return_inverse=True,
            return_counts=True,
            dim=0,
        )
        # strata_unique looks like:
        # [[0, 0], [1, 0], [1, 1]]
        # or:
        # [0, 1, 2, 3]  (torch.unique sorts the values)

        # strata_indices looks like:
        # [0, 0, 0; 1, 1, 1, 1; 2, 2, 2]
        # or:
        # [0, 3, 1, 0, 1, 1, 2, 0, 3]

        # strata_counts looks like:
        # [3, 4, 3]
        # or:
        # [3, 3, 1, 2]

        # Step 2: compute the sampling offsets and cardinals =============================
        # We group in contiguous intervals the patient indices that belong
        # to a given strata, and store this information in strata_values:

        strata_id, strata_values = torch.sort(strata_indices)
        # strata_values looks like:
        # [0, 1, 2; 3, 4, 5, 6; 7, 8, 9]
        # or:
        # [0, 3, 7; 2, 4, 5; 6; 1, 8]

        # strata_id looks like:
        # [0, 0, 0; 1, 1, 1, 1; 2, 2, 2]
        # or:
        # [0, 0, 0; 1, 1, 1; 2; 3, 3]

        # Then, we want store the cardinal of each strata in a vector strata_cardinal,
        # as well as offsets that we use below for indexing purposes:
        strata_cumsums = strata_counts.cumsum(dim=0)
        strata_cumsums = torch.cat(
            (torch.zeros_like(strata_cumsums[:1]), strata_cumsums)
        )
        strata_offset = strata_cumsums[strata_id]
        # strata_offset looks like:
        # [0, 0, 0; 3, 3, 3, 3; 7, 7, 7]
        # or:
        # [0, 0, 0; 3, 3, 3; 6; 7, 7]

        strata_cardinal = strata_counts[strata_id]
        # strata_cardinal looks like:
        # [3, 3, 3; 4, 4, 4, 4; 3, 3, 3]
        # or:
        # [3, 3, 3; 3, 3, 3; 1; 2, 2]

        assert strata_indices.shape == (P,)
        assert strata_offset.shape == (P,)
        assert strata_values.shape == (P,)
        assert strata_cardinal.shape == (P,)

        # Using this information, we can now generate bootstrap samples that respect
        # the stratification by using the formula:
        # bootstrap_indices = strata_values[strata_offset + floor(X * strata_cardinal)]
        # where X follows a uniform distribution in [0, 1).
        # Indeed, floor(X * strata_cardinal) will be a uniform distribution
        # in [0, strata_cardinal - 1], so that strata_offset + floor(X * strata_cardinal)
        # will be a uniform distribution in [strata_offset, strata_offset + strata_cardinal - 1],
        # i.e. strata_values[...] will be a uniform distribution in the set of patient
        # ids that belong to the strata.

        bootstrap_list = []
        if batch_size is None:
            batch_size = n_bootstraps
        for s in range(0, n_bootstraps, batch_size):
            B = min(batch_size, n_bootstraps - s)
            rnd = torch.rand(
                (B, P),
                dtype=torch.float32,
                device=self.device,
            )
            bootstrap_indices = strata_values[
                strata_offset + (rnd * strata_cardinal).long()
            ]

            bootstrap_list.append(
                Resampling(
                    indices=bootstrap_indices,
                    patient=self.patient,
                )
            )
        return bootstrap_list
