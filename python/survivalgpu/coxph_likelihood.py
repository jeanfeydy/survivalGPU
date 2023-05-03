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

from .group_reduction import group_reduce, SlicedSummation, group_logsumexp


from .typecheck import typecheck, Callable, Literal
from .typecheck import Float32Tensor
from .bootstrap import Resampling


@typecheck
def coxph_objective_unit_intervals(
    *,
    dataset,  #: TorchSurvivalDataset, omitted to avoid circular import
    ties: Literal["efron", "breslow"],
    backend: Literal["torch", "pyg", "coo", "csr"],
    bootstrap: Resampling,
) -> Callable[[Float32Tensor["batch_size covariates"]], Float32Tensor["batch_size"]]:
    """Implements the CoxPH objective in the case where `stop == start + 1`.

    Since we follow the survival convention and assume that all intervals are of
    the form `(start, stop]` with integer time values for `start` and `stop`,
    the condition above ensures that the intervals used to describe our dataset
    overlap if and only if they share the same 'stop' time.

    This simplifies some computations, compared with the general implementation of
    coxph_objective_any_intervals().

    This function evaluates `batch_size` instances of the CoxPH objective in parallel:
    we return one scalar value per row of the `coef` Tensor.
    We assume that `batch_size == len(bootstrap) * dataset.n_batch`:

    - If `batch_size == dataset.n_batch`, the vector of coefficients `coef[i]` will be
      associated to the subset of patients such that `dataset.batch == i`.
    - If `batch_size == len(bootstrap) * dataset.n_batch`, the vector of coefficients
      `coef[i]` will be associated to the subset of patients such that
      `dataset.batch == i % dataset.n_batch`.
      In other words, the `(batch_size, covariates)` Tensor of coefficients `coef`
      is interpreted as a `(n_bootstraps, dataset.n_batch, covariates)` Tensor.
    """

    # Pre-processing ---------------------------------------------------------------------
    # For each bootstrap and value of (batch, strata), aggregate the
    # "total weights for dead samples" at each time point.
    # These are required as multiplicative factors by the Efron and Breslow approximations.

    # Recall that bootstrap.interval_weights is a (n_bootstraps, n_intervals)
    # Tensor of int64 that records the number of occurences of each interval.

    # Compute the total weight of dead samples for every event time:
    dead_weights = bootstrap.interval_weights[:, dataset.event == 1]
    # dead_weights is (n_bootstraps, n_death_intervals), e.g.
    # [[1, 1, 1, 1],
    #  [2, 0, 1, 1]]

    # Recall that dataset.group is a (n_intervals,) Tensor of int64 that records
    # the T unique values of (batch, strata, stop):
    dead_cluster_indices = dataset.group[dataset.event == 1].repeat(len(bootstrap), 1)
    # dead_cluster_indices is (n_bootstraps, n_death_intervals), e.g.
    # [[0, 0, 1, 2],
    #  [0, 0, 1, 2]]

    tied_dead_weights = group_reduce(
        values=dead_weights,
        groups=dead_cluster_indices.long(),
        reduction="sum",
        output_size=dataset.n_groups,
        backend="pyg",
    )
    # Equivalent to:
    # tied_dead_weights = torch.bincount(cluster_indices[deaths == 1],
    #                     weights=weights.view(-1)[deaths == 1],
    #                     minlength=T)
    #
    # tied_dead_weights is (n_bootstraps,n_times), e.g.
    # [[2, 1, 1],
    #  [2, 1, 1]]
    assert tied_dead_weights.shape == (len(bootstrap), dataset.n_groups)

    # Create the summation groups --------------------------------------------------------
    if ties == "breslow":
        # The Breslow approximation is fairly straightforward,
        # with summation groups that correspond to the time "clusters"
        # of people "at risks" at any given time:
        group = dataset.group  # (n_intervals,)
        n_groups = dataset.n_groups  # n_times

        # With the Breslow approximation, the multiplicative factor
        # in front of the log-sum-exp term is equal to
        # (Sum_{dead at t} w[i]) = tied_dead_weights
        # weight_factor is (n_bootstraps, n_times):
        weight_factor = tied_dead_weights.view(len(bootstrap), n_groups)

    elif ties == "efron":
        # The Efron approximation handles "survivors" and "dying subjects"
        # differently (in every cluster of people "at risk").
        # To handle this, we build 2*n_times summation "groups":
        group = 2 * dataset.group + dataset.event
        n_groups = 2 * dataset.n_groups
        # If dataset.group is equal to:
        # [0, 0, 0, 0, 0, 1, 1, 1, 2, 2]
        # And if dataset.event is equal to:
        # [0, 0, 0, 1, 1, 0, 0, 1, 0, 1]
        # Then group is equal to:
        # [0, 0, 0, 1, 1, 2, 2, 3, 4, 5]

        # With the Breslow approximation and weights that come from bootstrapping,
        # the multiplicative factor in front of the log-sum-exp term is equal to 1.
        # -> there is no need to define a weight_factor variable.

    # Format the "group" vector as required by our backend for group-wise summations:
    if backend in ["torch", "pyg", "coo"]:
        group = group.repeat(len(bootstrap), 1)
        # group is (n_bootstrap,n_intervals),
        # and indicates the summation group that is associated to each interval e.g.
        # [[0, 0, 0, 0, 0, 1, 1, 1, 2, 2],
        #  [0, 0, 0, 0, 0, 1, 1, 1, 2, 2]]

    elif backend == "csr":
        # We assume that group looks like:
        # [0, 0, 1, 1, 1, 3, 4, 4, ...]
        assert dataset.is_sorted
        assert (group[1:] >= group[:-1]).all()

        cluster_sizes = torch.bincount(group, minlength=n_groups)
        assert cluster_sizes.shape == (n_groups,)
        group = torch.cat(
            (torch.zeros_like(cluster_sizes[:1]), cluster_sizes.cumsum(dim=0))
        )
        group = group.view(1, n_groups + 1).repeat(len(bootstrap), 1)
        # groups is (n_bootstraps, n_times+1) with Breslow,
        #           (n_bootstraps, 2*n_times + 1) with Efron.

    def negloglikelihood(
        scores: Float32Tensor["batch_size intervals"],
    ) -> Float32Tensor["batch_size"]:
        # TODO: below!!!
        B, D = coef.shape
        if B != len(bootstrap) * dataset.n_batch:
            raise ValueError(
                f"The number of rows {B} of the `coef` Tensor "
                f"should be equal to the number of bootstrap samples {len(bootstrap)}"
                f"times the number of batches {dataset.n_batch} "
                "that are referenced in `dataset.batch`."
            )

        coef = coef.view(len(bootstrap), dataset.n_batch, D)

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
