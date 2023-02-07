# ======================================================================================
# ==================== CoxPH log-likelihood, KeOps implementation ======================
# ======================================================================================
#
# Second implementation of the convex CoxPH objective, using KeOps LazyTensors.
# This implementation is work in progress for bootstrap > 1.
# Currently, it should be sub-optimal when there are not many ties :-(
#

# Use NumPy for basic array manipulation:
import numpy as np

# Use PyTorch for fast array manipulations (on the GPU):
import torch

# Use KeOps (www.kernel-operations.io) for computing e.g. the WCE features:
from pykeops.torch import LazyTensor


# Helper functions to create a block-diagonal mask in KeOps:
def ranges_slices(batch):
    """Helper function for the diagonal ranges function."""
    Ns = batch.bincount()
    indices = Ns.cumsum(0)
    ranges = torch.cat((0 * indices[:1], indices))
    ranges = (
        torch.stack((ranges[:-1], ranges[1:])).t().int().contiguous().to(batch.device)
    )
    slices = (1 + torch.arange(len(Ns))).int().to(batch.device)

    return ranges, slices


def diagonal_ranges(batch_x=None, batch_y=None):
    """Encodes the block-diagonal structure associated to a batch vector."""

    if batch_x is None and batch_y is None:
        return None  # No batch processing
    elif batch_y is None:
        batch_y = batch_x  # "symmetric" case

    ranges_x, slices_x = ranges_slices(batch_x)
    ranges_y, slices_y = ranges_slices(batch_y)

    return ranges_x, slices_x, ranges_y, ranges_y, slices_y, ranges_x


def coxph_objective_keops(
    *,
    N,
    B,
    T,
    x,
    deaths,
    weights,
    log_weights,
    ties,
    tied_deaths,
    tied_dead_weights,
    cluster_indices,
    backend="keops",
):
    """Please look at coxph_scatter.coxph_objective_torch for documentation."""

    # Create the arrays of offsets for ties:
    if ties == "breslow":
        offsets = torch.zeros(T).to(x.device)  # (T,)
        offsets_id = torch.arange(T).int().to(x.device)  # (T,)
        weight_factor = tied_dead_weights  # (B,T)

    elif ties == "efron":
        offsets = []
        offsets_id = []
        weight_factors = []
        for i, (tt, dw) in enumerate(zip(tied_deaths, tied_dead_weights.T)):
            t = max(1, tt)
            offsets += [np.log(k) - np.log(t) for k in range(1, t + 1)]
            offsets_id += [i] * t
            weight_factors += [dw / t] * t  # (B,) * t

        offsets = torch.FloatTensor(offsets).to(x.device)
        # offsets is (Tties,), e.g. [-0.7, 0, 0, 0]
        offsets_id = torch.IntTensor(offsets_id).to(x.device)
        # offsets_id is (Tties,), e.g. [0, 0, 1, 2]
        weight_factor = torch.stack(weight_factors, dim=1)
        # weight_factor is (B,Tties), e.g.
        # [[ 1,  1, 1, 1],
        #  [.5, .5, 1, 1]]

    else:
        raise ValueError(
            f"Incorrect value for ties ('{ties}'), should be either 'breslow' or 'efron'."
        )

    # Number of log-sum-exp reductions that we have to compute:
    Tties = len(offsets)  # = T for Breslow, sum(nties) for Efron
    # Generate the ranges for our block-diagonal reduction:
    ranges = diagonal_ranges(offsets_id, cluster_indices)

    def negloglikelihood(params):
        scores = x @ params.T  # (N,D) @ (D,B) = (N,B)
        weighted_scores = scores + log_weights.T  # (N,B)

        # Encoding as KeOps LazyTensors:
        offsets_i = LazyTensor(offsets.view(Tties, 1, 1))
        weighted_scores_j = LazyTensor(weighted_scores.view(1, N, B))
        deaths_j = LazyTensor(deaths.view(1, N, 1).float())

        log_weightedrisk_ij = offsets_i * deaths_j + weighted_scores_j  # (Tties, N, B)
        # Apply our block-diagonal mask:
        log_weightedrisk_ij.ranges = ranges

        # Perform the log-sum-exp computations in parallel:
        LSEs = log_weightedrisk_ij.logsumexp(dim=1)  # (Tties, B)

        # Finally, compute both terms in the CoxPH objective:
        # The linear term:
        lin = (weights.T.view(N, B) * scores.view(N, B) * deaths.view(N, 1)).sum(
            0
        )  # (B,)
        # The log-sum-exp term:
        lse = (weight_factor.T.view(Tties, 1) * LSEs.view(Tties, B)).sum(0)  # (B,)
        return lse - lin

    return negloglikelihood
