import torch
import numpy as np
from matplotlib import pyplot as plt
from .typecheck import typecheck, Optional, Callable, Union
from .typecheck import Int, Real
from .typecheck import Int64Tensor, Float32Tensor

@typecheck
def torch_lexsort(a : Int64Tensor["keys indices"]) -> Int64Tensor["indices"]:
    """PyTorch implementation of np.lexsort."""
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
        self.covariates = self.covariates[ind]
        self.batch_intervals = self.batch_intervals[ind]
        self.strata_intervals = self.strata_intervals[ind]

        # N.B.: self.strata and self.batch are not re-ordered, because they are
        #       arrays of length n_patients.

