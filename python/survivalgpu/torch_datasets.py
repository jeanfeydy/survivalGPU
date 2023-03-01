import torch
import numpy as np
from matplotlib import pyplot as plt
from .typecheck import typecheck, Optional, Callable, Union, Tuple
from .typecheck import Int, Real
from .typecheck import Int64Tensor, Float32Tensor


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
        keys = torch.stack((self.event, self.stop, self.strata, self.batch))
        ind = torch_lexsort(keys)

        # Re-order the arrays of length n_intervals:
        self.stop = self.stop[ind]
        self.start = self.start[ind]
        self.event = self.event[ind]
        self.patient = self.patient[ind]
        self.covariates = self.covariates[ind, :]
        self.batch_intervals = self.batch_intervals[ind]
        self.strata_intervals = self.strata_intervals[ind]

        # N.B.: self.strata and self.batch are not re-ordered, because they are
        #       arrays of length n_patients.

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
