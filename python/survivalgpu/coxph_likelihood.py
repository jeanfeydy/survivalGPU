# ======================================================================================
# ==================== CoxPH log-likelihood, PyTorch implementation ====================
# ======================================================================================
#
# First implementation of the convex CoxPH objective, using either:
# - Vanilla PyTorch - but please note that as of PyTorch 1.11, this code is
#   not GPU-compatible (https://github.com/pytorch/pytorch/issues/74770).
#   This will be fixed with PyTorch 1.12.
# - The PyTorch-Scatter package (https://github.com/rusty1s/pytorch_scatter)
#   for PyTorch-Geometric (https://pytorch-geometric.readthedocs.io/).
#   This implementation is probably the fastest one right now,
#   since it can rely on a CSR/"ranges" representation of the summation indices
#   while being more specialized than KeOps LazyTensors (which are sub-optimal
#   for the very "thin" block-sparsity masks that appear in the CoxPH log-likelihood).

# Import numpy to compute logarithms
import numpy as np

# Use PyTorch for fast array manipulations (on the GPU):
import torch

from .group_reduction import SlicedSummation, group_logsumexp


from .typecheck import typecheck, Optional, Literal
from .typecheck import Float32Tensor
from .bootstrap import Bootstrap



@typecheck
def coxph_objective_unit_intervals(
    coef: Float32Tensor["batches intervals"],
    *,
    dataset,  #: TorchSurvivalDataset, omitted to avoid circular import
    ties: Literal["efron", "breslow"],
    backend: Literal["torch", "pyg", "coo", "csr"],
    bootstrap: Bootstrap,
) -> Float32Tensor["batches"]:

    # Create the arrays of offsets for ties:
    if ties == "breslow":
        # The Breslow approximation is fairly straightforward,
        # with summation groups that correspond to the time "clusters"
        # of people "at risks" at any given time:
        group = dataset.group  # (N,)
        n_groups = dataset.n_groups  # T

        # TODO: BELOW!!!

        # With the Breslow approximation, the multiplicative factor
        # in front of the log-sum-exp term is equal to
        # (Sum_{dead at t} w[i]) = tied_dead_weights
        weight_factor = tied_dead_weights.view(B, T)  # (B,T)

    elif ties == "efron":
        # The Efron approximation handles "survivors" and "dying subjects"
        # differently (in every cluster of people "at risk").
        # To handle this, we build 2*T summation "groups":
        group_indices = 2 * cluster_indices + deaths
        ngroups = 2 * T
        # If cluster_indices is equal to:
        # [0, 0, 0, 0, 0, 1, 1, 1, 2, 2]
        # And if deaths is equal to:
        # [0, 0, 0, 1, 1, 0, 0, 1, 0, 1]
        # Then group_indices is equal to:
        # [0, 0, 0, 1, 1, 2, 2, 3, 4, 5]

        # With the Breslow approximation and weights that come from bootstrapping,
        # the multiplicative factor in front of the log-sum-exp term is equal to 1.
        # weight_factor = tied_dead_weights.view(B, T) # 1


def coxph_objective_torch(
    *,
    N,  # int
    B,  # int
    T,  # int
    risk_scores,  # (B,D) -> (B,N) function
    # x,  # (N,D) tensor
    deaths,  # (N,) tensor
    weights,  # (B,N) tensor
    log_weights,  # (B,N) tensor
    ties,  # string, "efron" or "breslow"
    tied_deaths,  # (T,) tensor
    tied_dead_weights,  # (B,T) tensor
    cluster_indices,  # (N,) tensor
    backend="torch",  # string, either "torch", "pyg", "coo" or "csr"
):
    """Implements the negative log-likelihood of the Cox Proportional Hazards model.

    This code runs several evaluations of the log-likelihood in parallel,
    that correspond to different values of the "importance weights"
    that are associated to the samples. This is to enable fast cross-validation
    with bootstrapping.

    We follow the exact same conventions as in the R "survival" package,
    as implemented at: https://github.com/therneau/survival/blob/master/src/coxfit6.c
    in the function coxfit6_iter(...).

    Namely, if:
    - b denotes the linear model's parameters, a (D,) vector.
    - x[i] denotes the features of the i-th sample, a (D,) vector.
    - w[i] denotes the importance weight of the i-th sample, a non-negative number.
    - r[i] = w[i] * exp(dot(x[i], b)) denotes the weighted risk of the i-th sample.

    Then, with the Breslow convention, the neg-log-likelihood is equal to:

    - Sum_{all dead samples} w[i] * dot(x[i], b)
    + Sum_{death times t} (
        (Sum_{dead at t} w[i])
        *
        log( Sum_{observed at t} r[i] )
        )

    With the Efron convention, the neg-log-likelihood is equal to:

    - Sum_{all dead samples} w[i] * dot(x[i], b)
    + Sum_{death times t} (
        (Sum_{dead at t} w[i]) / {number of deaths at t}
        *
        Sum_{k=1}^{number of deaths at t} (
            log(
                Sum_{survived at t} r[i]
                +
                (k / {number of deaths at t})
                *
                Sum_{dead at t} r[i]
                )
            )
        )

    As of today, we assume that the weights w[i] are integer numbers
    used for copy-free bootstrapping and thus simplify the Efron expression as:

    - Sum_{all dead samples} w[i] * dot(x[i], b)
    + Sum_{death times t} (
        Sum_{k=1}^{Sum_{dead at t} w[i]} (
            log(
                Sum_{survived at t} r[i]
                +
                (k / {Sum_{dead at t} w[i]})
                *
                Sum_{dead at t} r[i]
                )
            )
        )

    All the log-sum-exp computations are performed in a numerically stable way,
    by applying the max-factorization trick (https://en.wikipedia.org/wiki/LogSumExp)
    on the weighted scores:
        log(r[i]) = log(w[i]) + dot(x[i], b)


    Args:
        N (int): number of samples.
        B (int): batch size, i.e. the number of evaluations to run in parallel.
            Typically, it is equal to the number of bootstraps.
        T (int): number of interesting "death times".
        x ((N,D) tensor): input features for the samples.
        deaths ((N,) tensor): indicator function for the "dead" (=1) and
            "surviving" (=0) samples.
        weights ((B,N) tensor): importance weights for the samples.
            We use weights instead of data copies to implement bootstrap cross-validation
            as efficiently as possible.
        ties (string): convention to handle tied deaths. Either "efron" or "breslow".
        tied_dead_weights ((B,T) tensor): total weights of the dead samples per death time.
        cluster_indices ((N,) vector): "batch vector" that indicates the id of the
            "death time" that is associated to each sample.

    Raises:
        ValueError: if ties is not in ["efron", "breslow"].

    Returns:
        negloglikelihood (function): a convex function that takes as input a batch
            of parameters "beta" (encoded as a (B,D) tensor) and returns
            a (B,) vector of values.
    """

    # Create the arrays of offsets for ties:
    if ties == "breslow":
        # The Breslow approximation is fairly straightforward,
        # with summation groups that correspond to the time "clusters"
        # of people "at risks" at any given time:
        group_indices = cluster_indices  # (N,)
        ngroups = T

        # With the Breslow approximation, the multiplicative factor
        # in front of the log-sum-exp term is equal to
        # (Sum_{dead at t} w[i]) = tied_dead_weights
        weight_factor = tied_dead_weights.view(B, T)  # (B,T)

    elif ties == "efron":
        # The Efron approximation handles "survivors" and "dying subjects"
        # differently (in every cluster of people "at risk").
        # To handle this, we build 2*T summation "groups":
        group_indices = 2 * cluster_indices + deaths
        ngroups = 2 * T
        # If cluster_indices is equal to:
        # [0, 0, 0, 0, 0, 1, 1, 1, 2, 2]
        # And if deaths is equal to:
        # [0, 0, 0, 1, 1, 0, 0, 1, 0, 1]
        # Then group_indices is equal to:
        # [0, 0, 0, 1, 1, 2, 2, 3, 4, 5]

        # With the Breslow approximation and weights that come from bootstrapping,
        # the multiplicative factor in front of the log-sum-exp term is equal to 1.
        # weight_factor = tied_dead_weights.view(B, T) # 1

    else:
        raise ValueError(
            f"Incorrect value for ties ('{ties}'), should be either 'breslow' or 'efron'."
        )

    if backend in ["torch", "pyg", "coo"]:
        groups = group_indices.repeat(B, 1)
        # groups is (B,N), indicates the time that is associated to each sample e.g.
        # [[0, 0, 0, 0, 0, 1, 1, 1, 2, 2],
        #  [0, 0, 0, 0, 0, 1, 1, 1, 2, 2]]

    elif backend == "csr":
        cluster_sizes = torch.bincount(group_indices, minlength=ngroups)
        groups = torch.cat(
            (torch.zeros_like(cluster_sizes[:1]), cluster_sizes.cumsum(dim=0))
        )
        groups = groups.view(1, ngroups + 1).repeat(B, 1)
        # groups is (B, T+1) with Breslow, (B, 2*T+1) with Efron.

    def negloglikelihood(params):
        # nonlocal weight_factor

        # Compute the risk scores associated to the N feature vectors
        # by our current estimate of the parameters.
        # For linear scores "dot(beta, x[i])", this corresponds to
        # scores = params @ x.T, (B,D) @ (D,N) = (B,N)
        scores = risk_scores(params)  # (B,N)
        # And add the logarithms of the weights, so that
        # exp(weighted_scores) = w_i * exp(beta . x_i):
        weighted_scores = scores + log_weights  # (B,N)

        # The linear term in the CoxPH objective - (B,):
        lin = (weights.view(B, N) * scores.view(B, N) * deaths.view(1, N)).sum(1)

        if ties == "breslow":
            # groups_scores is (B,T)
            group_scores = group_logsumexp(
                values=weighted_scores, groups=groups, output_size=T, backend=backend
            )
            # The log-sum-exp term in the CoxPH log-likelihood - (B,):
            lse = (weight_factor * group_scores.view(B, T)).sum(1)

        elif ties == "efron":
            # groups_scores is (B,T*2)
            group_scores = group_logsumexp(
                values=weighted_scores,
                groups=groups,
                output_size=T * 2,
                backend=backend,
            )
            # We reshape it as a (B,T,2) array that contains, for every batch b
            # and every death time t, the log-sum-exp values that correspond
            # to "survivors" (= group_scores[b,t,0]) and
            # "tied deaths" (= group_scors[b,t,1]).
            group_scores = group_scores.view(B, T, 2)

            # To implement the Efron rule efficiently, we need to sort the B*T
            # groups by increasing number of deaths.
            # Please note that at this point, we mix together times that come
            # from different batches.
            # Please also note that since the tied_dead_weights come from bootstraps,
            # tied deaths are extremely likely to happen.
            order = tied_dead_weights.view(B * T).argsort()
            sorted_dead_weights = tied_dead_weights.view(B * T)[order]  # (B*T,)
            sorted_group_scores = group_scores.view(B * T, 2)[order, :]  # (B*T, 2)

            # We compute the "slice indices" that correspond to sorted_dead_weights:
            bincounts = torch.bincount(sorted_dead_weights.long())
            # bincounts is (Max_tied_deaths+1,).
            # It looks like:
            # [4, 5, 1, 0, 3, 0, 0, 1],  (shape = (8,))
            # i.e. there are:
            # - 4 times where no one dies,
            # - 5 times where there is a single death (= no ties),
            # - 1 time with 2 tied deaths,
            # - 3 times with 4 tied deaths,
            # - 1 time with 7 tied deaths.
            slice_indices = torch.cumsum(bincounts, dim=0).long()
            # slice_indices is (Max_tied_deaths+1,).
            # It looks like:
            # [4, 9, 10, 10, 13, 13, 13, 14],  (shape = (8,))

            # Our buffer for the time-wise values:
            slices = [torch.zeros_like(sorted_group_scores[:, 0])]  # (B*T,)
            for it, slice_start in enumerate(slice_indices):
                sliced_scores = sorted_group_scores[slice_start:, :]  # (#ties > it, 2)
                sliced_dead_weights = sorted_dead_weights[slice_start:]  # (#ties > it,)
                # sliced_scores[:,1] = sliced_scores[:,1] + np.log(it+1) - sliced_dead_weights.log()
                sliced_scores = torch.stack(
                    (
                        sliced_scores[:, 0].clamp(min=-(10**6)),
                        sliced_scores[:, 1]
                        + np.log(it + 1)
                        - sliced_dead_weights.log(),
                    ),
                    dim=1,
                )

                new_scores = sliced_scores.logsumexp(dim=-1)
                slices.append(new_scores)

                # The PyTorch autograd engine does not support in-place operations,
                # so we have to use a custom operator to implement the update:
                # sorted_scores[slice_start:] = sorted_scores[slice_start:] + new_scores
                # in a differentiable way.

            sorted_scores = SlicedSummation.apply(slice_indices, *slices)

            # We now need to re-sort
            time_scores = torch.zeros_like(sorted_group_scores[:, 0])  # (B*T,)
            time_scores[order] = sorted_scores

            # The log-sum-exp term in the CoxPH log-likelihood - (B,):
            lse = time_scores.view(B, T).sum(1)

        # lin and lse are (B,)
        ret_value = lse - lin  # (B,) values, computed in parallel
        return ret_value

    return negloglikelihood
