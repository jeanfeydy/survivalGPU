import torch
from .typecheck import typecheck, Float32Tensor, Int64Tensor
from .group_reduction import group_reduce


def stable_log(x, eps=1e-8):
    return (x + eps).log()


class Resampling:
    """Holds the pre-computed sample weights that are required for efficient bootstrapping.

    This class interacts with the CoxPH objective functions that are implemented in
    coxph_likelihood.py. It contains the information that is required to compute
    B separate values that correspond to independent bootstrap samples over the P patients.

    This code can handle time-varying covariates, that are encoded via multiple
    "intervals" per patients. As a consequence, the important quantities are both
    stored per patient and per interval. I denotes the total number of intervals in
    the dataset: it is typically larger than P.

    Attributes:
        patient_weights (B, P) float32 Tensor:
            The number of times each patient is drawn in each bootstrap sample.
            This is typically a tensor of integers that sum up to P, such as:
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [2, 1, 1, 2, 0, 0, 1, 0, 2, 1]]
        patient_log_weights (B, P) float32 Tensor:
            The pre-computed logarithms of the patient weights above.
        interval_weights (B, I) float32 Tensor:
            The number of times each interval is present in each bootstrap sample.
            Intervals that correspond to the same patient hold the same value.
        interval_log_weights (B, I) float32 Tensor:
            The pre-computed logarithms of the interval weights above.
    """

    @typecheck
    def __init__(
        self,
        *,
        indices: Int64Tensor["bootstraps samples"],
        event: Int64Tensor["intervals"],
        patient: Int64Tensor["intervals"],
    ):
        """
        We perform efficient bootstrapping using (B, I) arrays of "weights"
        where B is the number of bootstraps and I is the number of intervals.
        The original sample corresponds to weights = [1, ..., 1],
        while other values typically include 0's (for samples that are not drawn)
        and higher integer values (for samples that are drawn several times).

        This object computes all the relevant groups and weights from the following inputs:
        - indices, a (B, S) int64 Tensor
          where B is the number of bootstraps and S is the number of samples per bootstrap.
          The values of indices should all fall in [0, P-1] where P is the number of patients.
          Typical choices are:
          - indices = torch.arange(P) = [[0, 1, 2, 3, ..., P-1]],
            which corresponds to the original sample.
          - indices = torch.randint(P, (B, P)),
            which corresponds to a random bootstrap sample, e.g.
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  -> "miracle", we retrieve the original sample!
             [3, 3, 1, 2, 8, 6, 0, 0, 8, 9]]  -> "genuinely random" bootstrap sample
          - indices = [[0]],
            which corresponds to just picking the first patient, for debugging purposes.
          Note that these indices are usually drawn using a random number generator
          that respects the batch and stratification constraints.
          For instance, with S=P=10 as above, if:
          - batch  = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
          - strata = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
          This defines three groups of patients that should not mix in the bootstrap samples:
          - group=0: [0, 1, 2]
          - group=1: [3, 4, 5, 6]
          - group=2: [7, 8, 9]
          Consequently, with B=2 bootstraps, typical indices would be:
          - indices = [[0, 1, 2 ; 3, 4, 5, 6 ; 7, 8, 9],  -> "miracle", we retrieve the original sample!
                       [0, 0, 1 ; 4, 6, 4, 4 ; 8, 8, 8]]  -> "genuinely random" bootstrap sample

        - event, a (I,) int64 Tensor that indicates the event type for each interval.
        - patient, a (I,) int64 Tensor that indicates the patient for each interval.
        """
        B, S = indices.shape
        P = patient.max() + 1
        I = event.shape[0]

        # Step 1: compute the patient weights --------------------------------------------
        # Compute the numbers of occurrences of each patient index in the
        # rows of bootstrap_indices:
        sample_weights = torch.ones(B, P, dtype=torch.float32, device=indices.device)
        self.patient_weights = group_reduce(
            values=sample_weights,
            groups=indices,
            reduction="sum",
            output_size=P,
            backend="pyg",
        ).to(
            device=indices.device,
            dtype=torch.float32,
        )
        # Equivalent to:
        # self.patient_weights
        # = torch.stack([torch.bincount(b_ind, minlength=P) for b_ind in indices])
        #
        # self.patient_weights is (B,P) with lines that sum up to P, e.g.
        # [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #  [2, 1, 1, 2, 0, 0, 1, 0, 2, 1]]

        # Pre-compute the logarithms of the weights:
        # TODO: We are currently adding a small value to prevent NaN.
        #       This is not very clean...
        self.patient_log_weights = stable_log(self.patient_weights)  # (B,P), e.g.
        # [[ 0, 0, 0,  0,   0,   0, 0,    0,  0, 0],
        #  [.7, 0, 0, .7,-inf,-inf, 0, -inf, .7, 0]]

        assert self.patient_weights.shape == (B, P)
        assert self.patient_log_weights.shape == (B, P)

        # Step 2: compute the interval weights -------------------------------------------
        self.interval_weights = self.patient_weights[:, indices]
        self.interval_log_weights = stable_log(self.interval_weights)

        assert self.interval_weights.shape == (B, I)
        assert self.interval_log_weights.shape == (B, I)
