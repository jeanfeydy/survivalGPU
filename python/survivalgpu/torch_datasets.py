import numpy as np
from matplotlib import pyplot as plt
from .typecheck import typecheck, Optional, Callable, Union
from .typecheck import Int, Real
from .typecheck import Int64Tensor, Float32Tensor


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
        """Re-orders the input arrays by lexicographical order on (batch, strata, stop, event)."""
        pass
