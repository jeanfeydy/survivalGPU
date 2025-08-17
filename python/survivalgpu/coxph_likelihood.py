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
    Sum_{k=0}^{number of deaths at t - 1} (
        log(
            Sum_{observed at t} r[i]
            -
            (k / {number of deaths at t})
            *
            Sum_{dead at t} r[i]
            )
        )
    )

We assume that the weights w[i] are integer numbers used for copy-free bootstrapping,
which implies that:
Sum_{dead at t} w[i] = {number of deaths at t}.

This simplifies the Efron expression as:

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

"""

# ======================================================================================
# ==================== CoxPH log-likelihood, PyTorch implementation ====================
# ======================================================================================
#
# Our main implementation of the convex CoxPH objective, using vanilla PyTorch.
# The PyTorch-Scatter package (https://github.com/rusty1s/pytorch_scatter)
# for PyTorch-Geometric (https://pytorch-geometric.readthedocs.io/)
# was considered for a long time, but eventually dropped to keep dependencies minimal.

# Import numpy to compute logarithms
import numpy as np

# Use PyTorch for fast array manipulations (on the GPU):
import torch

from .bootstrap import Resampling
from .group_reduction import (
    first_in_segment,
    group_logsumexp,
    group_sum,
    keys_to_segments,
    logdiffexp,
    segment_logcumsumexp,
)
from .typecheck import Callable, Float32Tensor, Int64Tensor, Literal, typecheck


@typecheck
def _compute_unique_batch_strata_time(
    *,
    batch: Int64Tensor["intervals"],
    strata: Int64Tensor["intervals"],
    start: Int64Tensor["intervals"],
    stop: Int64Tensor["intervals"],
) -> tuple[Int64Tensor["3 times"], Int64Tensor["intervals"], Int64Tensor["intervals"]]:
    """Collapses the intervals of the dataset into unique (batch, strata, time) values.

    .. testcode::

        import torch
        from survivalgpu.coxph_likelihood import _compute_unique_batch_strata_time

        batch = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1])
        strata = torch.tensor([0, 0, 0, 1, 1, 1, 0, 0, 0])
        start = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0])
        stop = torch.tensor([1, 2, 3, 1, 2, 3, 1, 2, 3])

        unique_batch_strata_time, index_start, index_stop = _compute_unique_batch_strata_time(
            batch=batch,
            strata=strata,
            start=start,
            stop=stop,
        )
        print(unique_batch_strata_time)

    .. testoutput::

        tensor([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]])

    .. testcode::

        print(index_start)

    .. testoutput::

        tensor([0, 0, 0, 4, 4, 4, 8, 8, 8])

    .. testcode::

        print(index_stop)

    .. testoutput::

        tensor([ 1,  2,  3,  5,  6,  7,  9, 10, 11])

    """

    I = batch.shape[0]

    # batch and strata define independent groups,
    # while start and stop refer to time values.
    # Working in parallel over (batch, strata) groups,
    # we need to sort by time (handling both start and stop)
    # and find the number of unique (batch, strata, time) values.
    batch_strata_start = torch.stack((batch, strata, start), dim=0)
    batch_strata_stop = torch.stack((batch, strata, stop), dim=0)
    assert batch_strata_start.shape == (3, I)
    assert batch_strata_stop.shape == (3, I)

    batch_strata_start_stop = torch.cat(
        (batch_strata_start, batch_strata_stop), dim=1
    )
    assert batch_strata_start_stop.shape == (3, 2 * I)

    # Find the unique (batch, strata, time) values
    unique_batch_strata_time, inverse_indices = torch.unique(
        batch_strata_start_stop, sorted=True, return_inverse=True, dim=1
    )
    T = unique_batch_strata_time.shape[1]
    assert unique_batch_strata_time.shape == (3, T)
    assert T <= 2 * I

    assert inverse_indices.shape == (2 * I,)
    assert unique_batch_strata_time.shape[1] <= 2 * I

    index_start = inverse_indices[:I]
    index_stop = inverse_indices[I:]

    return unique_batch_strata_time, index_start, index_stop


@typecheck
def _compute_time_data(
    *,
    interval_counts: Int64Tensor["bootstraps intervals"],
    interval_weights: Float32Tensor["bootstraps intervals"],
    interval_weighted_scores: Float32Tensor["bootstraps intervals"],
    index_start: Int64Tensor["intervals"],
    index_stop: Int64Tensor["intervals"],
    event: Int64Tensor["intervals"],
    T: int,
) -> tuple[
    Int64Tensor["bootstraps times 2 2"],
    Float32Tensor["bootstraps times 2 2"],
    Float32Tensor["bootstraps times 2 2"]
]:
    """Aggregates the interval data ("bootstrap" counts, weights, scores) into time data.

    Recall that for each interval (start, stop], we have:

     - an integer count of "bootstrap" occurrences,
     - a float weight >= 0,
     - a weighted score that corresponds to log(risk) = log(weight) + dot(beta, x).

    We aggregate these values into a "time-indexed" data table:
    for every bootstrap b, data for interval (start, stop] is aggregated at locations
    [b, stop, 0, event] and [b, start, 1, event],
    where event == 0 if the interval is "censored" and event == 1 if it ends with an event.

    The reduction for counts and weights is a sum,
    while the reduction for weighted scores is a log-sum-exp.

    For each one of our three "tables" (counts, weights, weighted scores),
    bootstrap index b and time t, the table values correspond to the following aggregations:

    - table[b, t, 0, 0]: intervals that stop at time t, without an event.
    - table[b, t, 0, 1]: intervals that stop at time t, with an event.
    - table[b, t, 1, 0]: intervals that start at time t, without an event.
    - table[b, t, 1, 1]: intervals that start at time t, with an event.


    .. warning::

        With our (start, stop] convention,
        data for the "stop" time actually happens at time "stop",
        whereas data for the "start" time actually happens at time "start + epsilon".

        Along the 3rd dimension of the time data table, we thus make sure that
        the "stop" index is 0 and the "start" index is 1,
        not the other way around.

        This convention ensures that once we flatten the table as
        a (B, T * 4) tensor (for a cumsum in the Breslow/Efron implementations),
        time data is ordered properly.


    .. testcode::

        import torch
        from survivalgpu.coxph_likelihood import _compute_time_data

        interval_counts = torch.tensor([[1, 2, 3], [2, 3, 4]])
        interval_weights = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        interval_weighted_scores = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        index_start = torch.tensor([0, 1, 2])
        index_stop = torch.tensor([1, 3, 3])
        event = torch.tensor([0, 1, 1])
        T = 4  # Number of unique time points

        time_counts, time_weights, time_weighted_scores = _compute_time_data(
            interval_counts=interval_counts,
            interval_weights=interval_weights,
            interval_weighted_scores=interval_weighted_scores,
            index_start=index_start,
            index_stop=index_stop,
            event=event,
            T=T,
        )

        print(time_counts.view(2, -1))

    .. testoutput::

        tensor([[0, 0, 1, 0, 1, 0, 0, 2, 0, 0, 0, 3, 0, 5, 0, 0],
                [0, 0, 2, 0, 2, 0, 0, 3, 0, 0, 0, 4, 0, 7, 0, 0]])


    .. testcode::

        print(time_weights.view(2, -1))

    .. testoutput::

        tensor([[0., 0., 1., 0., 1., 0., 0., 2., 0., 0., 0., 3., 0., 5., 0., 0.],
                [0., 0., 2., 0., 2., 0., 0., 3., 0., 0., 0., 4., 0., 7., 0., 0.]])

    .. testcode::

        print(time_weighted_scores.view(2, -1))

    .. testoutput::

        tensor([[  -inf,   -inf, 1.0000,   -inf, 1.0000,   -inf,   -inf, 2.0000,   -inf,
                   -inf,   -inf, 3.0000,   -inf, 3.3133,   -inf,   -inf],
                [  -inf,   -inf, 2.0000,   -inf, 2.0000,   -inf,   -inf, 3.0000,   -inf,
                   -inf,   -inf, 4.0000,   -inf, 4.3133,   -inf,   -inf]])

    """
    B, I = interval_counts.shape

    # index_start and index_stop have values in [0, T-1]
    assert ((index_start >= 0) & (index_start < T)).all()
    assert ((index_stop >= 0) & (index_stop < T)).all()
    assert max(index_start.max(), index_stop.max()) == T - 1

    full_index = torch.cat(
        (
            (4 * index_stop + event),
            (4 * index_start + 2 + event),
        ),
        dim=0,
    )
    assert full_index.shape == (2 * I,)

    time_counts = group_sum(
        values=torch.cat((interval_counts,) * 2, dim=1),
        groups=full_index,
        output_size=T * 4,
    ).view(B, T, 2, 2)

    time_weights = group_sum(
        values=torch.cat((interval_weights,) * 2, dim=1),
        groups=full_index,
        output_size=T * 4,
    ).view(B, T, 2, 2)

    time_weighted_scores = group_logsumexp(
        values=torch.cat((interval_weighted_scores,) * 2, dim=1),
        groups=full_index,
        output_size=T * 4,
    ).view(B, T, 2, 2)

    return time_counts, time_weights, time_weighted_scores




@typecheck
def _intervals_to_time_data(
    *,
    scores: Float32Tensor["bootstraps intervals"],
    interval_counts: Int64Tensor["bootstraps intervals"],
    interval_weights: Float32Tensor["bootstraps intervals"],
    interval_log_weights: Float32Tensor["bootstraps intervals"],
    batch: Int64Tensor["intervals"],
    strata: Int64Tensor["intervals"],
    start: Int64Tensor["intervals"],
    stop: Int64Tensor["intervals"],
    event: Int64Tensor["intervals"],
) -> tuple[
    Int64Tensor["bootstraps times 2 2"],  # Counts of intervals at each time
    Float32Tensor["bootstraps times 2 2"],  # Weights of intervals at each time
    Float32Tensor["bootstraps times 2 2"],  # Weighted scores at each time
    Int64Tensor["3 times"],  # Unique (batch, strata, time) values
]:
    """Aggregates interval data into a table indexed by time.

    The format of the output is the same as in _compute_time_data(),
    plus the unique (batch, strata, time) values.

    .. testcode::

        import torch
        from survivalgpu.coxph_likelihood import _intervals_to_time_data

        time_counts, time_weights, time_weighted_scores, unique_batch_strata_time = (
            _intervals_to_time_data(
                scores=torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
                interval_counts=torch.tensor([[1, 1, 3], [2, 2, 1]]),
                interval_weights=torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
                interval_log_weights=torch.tensor(
                    [[0.0, 0.6931, 1.0986], [0.6931, 1.0986, 1.3863]]
                ),
                batch=torch.tensor([0, 0, 0]),
                strata=torch.tensor([0, 0, 0]),
                start=torch.tensor([0, 0, 0]),
                stop=torch.tensor([1, 2, 2]),
                event=torch.tensor([0, 1, 1]),
            )
        )

        print(time_counts.view(2, -1))

    .. testoutput::

        tensor([[0, 0, 1, 4, 1, 0, 0, 0, 0, 4, 0, 0],
                [0, 0, 2, 3, 2, 0, 0, 0, 0, 3, 0, 0]])

    .. testcode::

        print(time_weights.view(2, -1))

    .. testoutput::

        tensor([[0., 0., 1., 5., 1., 0., 0., 0., 0., 5., 0., 0.],
                [0., 0., 2., 7., 2., 0., 0., 0., 0., 7., 0., 0.]])

    .. testcode::

        print(time_weighted_scores.view(2, -1))

    .. testoutput::

        tensor([[  -inf,   -inf, 1.0000, 4.3179, 1.0000,   -inf,   -inf,   -inf,   -inf,
                 4.3179,   -inf,   -inf],
                [  -inf,   -inf, 2.6931, 5.6300, 2.6931,   -inf,   -inf,   -inf,   -inf,
                 5.6300,   -inf,   -inf]])

    .. testcode::

        print(unique_batch_strata_time)

    .. testoutput::

        tensor([[0, 0, 0],
                [0, 0, 0],
                [0, 1, 2]])

    """
    B, I = scores.shape

    unique_batch_strata_time, index_start, index_stop = _compute_unique_batch_strata_time(
        batch=batch,
        strata=strata,
        start=start,
        stop=stop,
    )
    T = unique_batch_strata_time.shape[1]
    assert unique_batch_strata_time.shape == (3, T)
    assert index_start.shape == (I,)
    assert index_stop.shape == (I,)

    # log(r[b,i]) = log(w[b,i]) + dot(beta[b], x[i])
    interval_weighted_scores = interval_log_weights + scores
    assert interval_weighted_scores.shape == (B, I)

    time_counts, time_weights, time_weighted_scores = _compute_time_data(
        interval_counts=interval_counts,
        interval_weights=interval_weights,
        interval_weighted_scores=interval_weighted_scores,
        index_start=index_start,
        index_stop=index_stop,
        event=event,
        T=T,
    )

    return (
        time_counts,
        time_weights,
        time_weighted_scores,
        unique_batch_strata_time,
    )


@typecheck
def _compute_time_log_risks(
    *,
    time_weighted_scores: Float32Tensor["bootstraps times 2 2"],
    unique_batch_strata_time: Int64Tensor["3 times"],
) -> Float32Tensor["bootstraps times"]:
    """Computes the log-risk over the full risk set of observed patients at each time point.

    .. warning::

        Currently, this is float32-based, which may lead to numerical errors
        when the number of time points T is very large (e.g. T > 10k).

    .. testcode::

        import torch
        from survivalgpu.coxph_likelihood import _compute_time_log_risks

        # First "bootstrap" corresponds to:
        #  - one interval (0, 1] with a score of 1 and no event,
        #  - one interval (0, 2] with a score of 2 and an event.
        # We expect the log-risks to be:
        #  - at time 0: -inf (no risk set),
        #  - at time 1: log(e^1 + e^2) = 2.3133
        #  - at time 2: log(e^2) = 2.0000
        #
        # Second "bootstrap" corresponds to:
        #  - one interval (0, 2] with a score of 1 and no event,
        #  - one interval (1, 2] with a score of 3 and an event.
        # We expect the log-risks to be:
        #  - at time 0: -inf (no risk set),
        #  - at time 1: log(e^1) = 1.0000
        #  - at time 2: log(e^1 + e^3) = 3.1269
        #
        # We also add an empty strata at the end.

        z = -float("inf")
        time_log_risks = _compute_time_log_risks(
            time_weighted_scores=torch.tensor(
                [
                    [
                        [[z, z], [1.0, 2.0]],
                        [[1.0, z], [z, z]],
                        [[z, 2.0], [z, z]],
                        [[z, z], [z, z]],
                    ],
                    [
                        [[z, z], [1.0, z]],
                        [[z, z], [z, 3.0]],
                        [[1.0, 3.0], [z, z]],
                        [[z, z], [z, z]],
                    ],
                ]
            ),
            unique_batch_strata_time=torch.tensor(
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 1, 2, 4],
                ]
            ),
        )
        print(time_log_risks)

    .. testoutput::

        tensor([[  -inf, 2.3133, 2.0000,   -inf],
                [  -inf, 1.0000, 3.1269,   -inf]])
    """

    B, T, _, _ = time_weighted_scores.shape

    # Reduce over the "no event / event" dimension
    time_risk_updates = time_weighted_scores.logsumexp(dim=-1)
    assert time_risk_updates.shape == (B, T, 2)

    # Define summation groups by (batch, strata)
    batch_strata_segments = keys_to_segments(unique_batch_strata_time[:2])
    assert batch_strata_segments.shape == (T,)

    # TODO: when T > 10k, we should probably switch to float64 here...
    if T > 10000:
        # Raise a warning if T is too large
        import warnings
        warnings.warn(
            "T is larger than 10k, we should implement a float64 backend for numerical stability.",
            stacklevel=1,
        )

    # Recall that with our convention, the "stop" index is 0 and the "start" index is 1
    # along the 3rd dimension of our time data tables.
    time_risk_set_stop = segment_logcumsumexp(
        values=time_risk_updates[:, :, 0],
        segments=batch_strata_segments,
    )
    time_risk_set_start = segment_logcumsumexp(
        values=time_risk_updates[:, :, 1],
        segments=batch_strata_segments,
    )

    # At time t, the "weighted risk" over the risk set
    #    Sum_{observed at t} r[i]
    # is equal to the difference:
    #    Sum_{started at time < t} r[i]
    #  - Sum_{stopped at time < t} r[i]
    # For the sake of numerical stability, we compute this difference
    # in the log-domain:
    time_log_risks = logdiffexp(time_risk_set_start, time_risk_set_stop)
    assert time_log_risks.shape == (B, T)

    # We shift these log-risks to the right by one time step,
    # in order to compensate for the "< t" condition above.
    time_log_risks = torch.cat(
        (
            -float("inf") * torch.ones_like(time_log_risks[:, :1]),
            time_log_risks[:, :-1],
        ),
        dim=1,
    )
    assert time_log_risks.shape == (B, T)

    # N.B.: This shift fills the first time step of every segment
    #       with a very small value (theoretically equal to -inf
    #       since every interval appears once in "start" and once in "stop").
    #       This is not a problem, since the first time step
    #       of every segment can only correspond to a "start" time,
    #       not a "stop" time, and therefore does not contribute
    #       to the log-sum-exp term of the CoxPH objective.

    return time_log_risks


@typecheck
def _breslow_efron_logsumexp_term(
    *,
    time_counts: Int64Tensor["bootstraps times 2 2"],
    time_weights: Float32Tensor["bootstraps times 2 2"],
    time_weighted_scores: Float32Tensor["bootstraps times 2 2"],
    unique_batch_strata_time: Int64Tensor["3 times"],
    ties: Literal["efron", "breslow"],
    n_batches: int,
) -> Float32Tensor["bootstraps {n_batches}"]:
    """Computes the log-sum-exp term for the Breslow or Efron approximation.

    The format of the input is explained in the docstring of _compute_time_data().

    .. testcode::

        import torch
        from survivalgpu.coxph_likelihood import _breslow_efron_logsumexp_term

        # Very simple example with just two times, 0 and 1.
        unique_batch_strata_time = torch.tensor(
            [
                [0, 0],
                [0, 0],
                [0, 1],
            ]
        )
        # Bootstrap 1 has:
        # - 1 interval (0, 1] that is censored with a weighted score of 2,
        # - 1 interval (0, 1] that is an event with a weighted score of 3.
        # Bootstrap 2 has:
        # - 3 intervals (0, 1] that are censored with a weighted score of 1,
        # - 2 intervals (0, 1] that correspond to an event with a weighted score of 0.
        time_counts = torch.tensor(
            [
                [[[0, 0], [1, 1]], [[1, 1], [0, 0]]],
                [[[0, 0], [3, 2]], [[3, 2], [0, 0]]],
            ]
        )
        time_weights = torch.tensor(
            [
                [[[0.0, 0.0], [1.0, 1.0]], [[1.0, 1.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [3.0, 2.0]], [[3.0, 2.0], [0.0, 0.0]]],
            ]
        )
        z = -float("inf")
        time_weighted_scores = torch.tensor(
            [
                [[[z, z], [2.0, 3.0]], [[2.0, 3.0], [z, z]]],
                [[[z, z], [1.0, 0.0]], [[1.0, 0.0], [z, z]]],
            ]
        )

        # The contribution at time 0 is 0, since there are no deaths at that time.
        # At time 1, with the Breslow approximation for ties, we expect:
        # - for bootstrap 1, 1 * log(exp(2) + exp(3)) = 3.3133
        # - for bootstrap 2, 2 * log(exp(1) + exp(0)) = 2.6265
        print(
            _breslow_efron_logsumexp_term(
                time_counts=time_counts,
                time_weights=time_weights,
                time_weighted_scores=time_weighted_scores,
                unique_batch_strata_time=unique_batch_strata_time,
                ties="breslow",
                n_batches=1,
            )
        )

    .. testoutput::

        tensor([[3.3133],
                [2.6265]])

    """

    B, T, _, _ = time_counts.shape

    time_log_risks =  _compute_time_log_risks(
        time_weighted_scores=time_weighted_scores,
        unique_batch_strata_time=unique_batch_strata_time,
    )
    assert time_log_risks.shape == (B, T)

    # Summation groups by (batch, strata)
    segments = keys_to_segments(unique_batch_strata_time[:2])
    assert segments.shape == (T,)

    # In every segment, the very first time should be a "start", not a "stop".
    # We make sure that the corresponding weight is 0.
    first_in_segment_mask = first_in_segment(segments)
    assert first_in_segment_mask.shape == (T,)

    # Recall that with our convention, the "stop" index is 0 and the "start" index is 1
    # along the 3rd dimension of our time data tables.
    assert (time_weights[:, first_in_segment_mask, 0, :] == 0.).all()

    # The weight factor is usually equal to the number of dead samples at each time,
    # but we use time_weights instead of time_counts to handle
    # cases were the user provided varying weights for each patient.

    # Compute the factor "(Sum_{dead at t} w[i])"
    # 0 on 3rd dimension corresponds to "stop" time,
    # 1 on 4th dimension corresponds to "event".
    dead_weights = time_weights[:, :, 0, 1]
    assert dead_weights.shape == (B, T)

    if ties == "breslow":
        # The Breslow approximation is straightforward:
        # we sum the log-risks over the risk set at each time,
        # weighted by cumulative weight of intervals that stop at that time.

        # Recall that we are computing:
        # + Sum_{death times t} (
        #     (Sum_{dead at t} w[i])
        #     *
        #     log( Sum_{observed at t} r[i] )
        #   )
        time_contributions = dead_weights * time_log_risks

        # When dead_weights == 0, the contribution is 0, even if time_log_risks is -inf.
        # If we don't mask things out, we would end up with -inf * 0 == NaN.
        time_contributions = torch.where(
            dead_weights != 0,
            time_contributions,
            torch.zeros_like(time_contributions),
        )
        assert time_contributions.shape == (B, T)

        # We sum batch-wise over the time contributions:
        logsumexp_term = group_sum(
            values=time_contributions,
            groups=unique_batch_strata_time[0],
            output_size=n_batches,
        )

    elif ties == "efron":
        # The Efron approximation is more complex:
        # we sum the log-risks over the risk set at each time,
        # weighted by cumulative weight of intervals that stop at that time,
        # divided by the number of deaths at that time.

        # Recall that we are computing:
        # + Sum_{death times t} (
        #     (Sum_{dead at t} w[i]) / {number of deaths at t}
        #     *
        #     Sum_{k=0}^{number of deaths at t - 1} (
        #         log(
        #             Sum_{observed at t} r[i]
        #             -
        #             (k / {number of deaths at t})
        #             *
        #             Sum_{dead at t} r[i]
        #         )
        #     )
        #   )

        msg = "Very soon!"
        raise NotImplementedError(msg)

    return logsumexp_term





@typecheck
def coxph_objective(
    *,
    dataset,  #: TorchSurvivalDataset, omitted to avoid circular import
    ties: Literal["efron", "breslow"],
    bootstrap: Resampling,
    mode: Literal["unit length", "start zero", "any"],
) -> Callable[[Float32Tensor["bootstraps intervals"]], Float32Tensor["batch_size"]]:
    """Implements the CoxPH objective.

    Depending on the value of "mode", we use different optimizations:

      - mode == "unit length" corresponds to the case where `stop == start + 1`.

        Since we follow the survival convention and assume that all intervals are of
        the form `(start, stop]` with integer time values for `start` and `stop`,
        the condition above ensures that the intervals used to describe our dataset
        overlap if and only if they share the same 'stop' time.

      - mode == "start zero" corresponds to the case where `start == 0`.

        This implies that all risk sets correspond to successive subsets of the dataset,
        with computations that can be handled by a cumsum.

      - mode == "any" corresponds to the general case, where we have no assumption
        on the values of `start` and `stop`.

    The objective function evaluates `batch_size` instances of the CoxPH
    neg-log-likelihood in parallel.
    It takes as input a collection of scores (presumably computed via a dot
    product between a vector of parameters and a vector of covariates)
    and returns a vector of length `batch_size == len(bootstrap) * dataset.n_batch`
    which is identified with len(bootstrap) vectors of length data.n_batch,
    concatenated with each other.

    .. testcode::

        import survivalgpu

        print(1 + 1)

    .. testoutput::

        2

    """
    B = len(bootstrap)  # Number of bootstraps to process in parallel
    I = dataset.n_intervals  # Number of intervals in the dataset
    E = dataset.n_event_intervals  # Number of event intervals in the dataset

    if I == 0:
        msg = "The dataset is empty (dataset.n_intervals == 0)."
        raise ValueError(msg)

    # Pre-processing ---------------------------------------------------------------------
    # For each bootstrap and value of (batch, strata), aggregate the
    # "total weights for dead samples" at each time point.
    # These are required as multiplicative factors by the Efron and Breslow approximations.

    # Recall that bootstrap.interval_weights is a (n_bootstraps, n_intervals)
    # Tensor of int64 that records the number of occurrences of each interval.
    assert bootstrap.interval_weights.shape == (B, I)

    # Compute the total weight of dead samples for every event time:
    dead_weights = bootstrap.interval_weights[:, dataset.event == 1]
    assert dead_weights.shape == (B, E)
    # dead_weights is (n_bootstraps, n_event_intervals), e.g.
    # [[1, 1, 1, 1],
    #  [2, 0, 1, 1]]

    # Recall that dataset.group is a (n_intervals,) Tensor of int64 that records
    # the T unique values of (batch, strata, stop):
    assert dataset.group.shape == (I,)

    # Select the indices of the "death" intervals
    dead_cluster_indices = dataset.group[dataset.event == 1].long()
    assert dead_cluster_indices.shape == (E,)
    # dead_cluster_indices is (n_death_intervals,), e.g.
    # [0, 0, 1, 2]

    tied_dead_weights = group_sum(
        values=dead_weights,
        groups=dead_cluster_indices,
        output_size=dataset.n_groups,
    )
    # Equivalent to:
    # tied_dead_weights = torch.bincount(cluster_indices[deaths == 1],
    #                     weights=weights.view(-1)[deaths == 1],
    #                     minlength=T)
    #
    # tied_dead_weights is (n_bootstraps,n_times), e.g.
    # [[2, 1, 1],
    #  [2, 1, 1]]
    assert tied_dead_weights.shape == (B, dataset.n_groups)

    # Create the summation groups --------------------------------------------------------
    if ties == "breslow":
        # The Breslow approximation is fairly straightforward,
        # with summation groups by value of (batch, strata, stop)
        # that correspond to the time "clusters" of people "at risks" at any given time:
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

    # Format the "group" vector for group-wise summations:
    assert group.shape == (I,)
    # group is (n_intervals,),
    # and indicates the summation group that is associated to each interval e.g.
    # [0, 0, 0, 0, 0, 1, 1, 1, 2, 2]

    @typecheck
    def negloglikelihood(
        scores: Float32Tensor["bootstraps intervals"],
    ) -> Float32Tensor["batches"]:
        """The CoxPH neg-log-likelihood that we try to minimize.

        This function takes as input a batch of score values scores[i, j],
        each of whom corresponds to the risk score of the j-th interval
        according to the i-th estimate of the model parameters.

        For linear scores "dot(beta[i], x[j])", this corresponds to
        scores = beta @ x.T, (B,D) @ (D,I) = (B,I)

        """
        if scores.shape[0] != B:
            msg = (
                f"The number of rows {scores.shape[0]} of the `scores` Tensor "
                f"should be equal to the number of bootstrap samples {B}."
            )
            raise ValueError(msg)

        if scores.shape[1] != I:
            msg = (
                f"The number of columns {scores.shape[1]} of the `scores` Tensor "
                f"should be equal to the number of intervals {I} "
                "that are referenced in `dataset.stop`."
            )
            raise ValueError(msg)


        # The linear term in the CoxPH objective - (n_bootstraps,n_batch) ==============
        # This is the term:
        #
        #   Sum_{all dead samples} w[i] * dot(x[i], b)
        # = Sum_{all samples} w[i] * dot(x[i], b) * event[i]
        #
        # that we compute in parallel over:
        # - all n_bootstrap values of the scores,
        # - all n_batch values of the parameter vector.
        #
        # Note that the strata does not matter here, because we sum all contributions
        # identically:
        # Sum_{strata s} Sum_{all samples in strata s} ... = Sum_{all samples} ...
        assert torch.all((dataset.event == 0) | (dataset.event == 1))
        lin = (
            bootstrap.interval_weights.view(B, I)
            * scores.view(B, I)
            * dataset.event.view(1, I)
        )
        lin = group_sum(
            values=lin,
            groups=dataset.batch_intervals,
            output_size=dataset.n_batch,
        )
        assert lin.shape == (B, dataset.n_batch)

        # The log-sum-exp term in the CoxPH log-likelihood - (n_bootstrap, n_batch) ====

        # We add the logarithms of the weights to the scores, so that
        # exp(weighted_scores[b,i]) = w[b,i] * exp(beta[b] . x[i]) = r[b,i]:
        assert bootstrap.interval_log_weights.shape == (B, I)
        weighted_scores = scores + bootstrap.interval_log_weights  # (B,I)
        assert weighted_scores.shape == (B, I)


        # At this stage:
        #
        # - weighted_scores[b,i] = log(r[b,i])
        #   corresponds to the log-risk
        #   for the b-th bootstrap (and therefore, the b-th parameter estimate)
        #   and the i-th interval.
        assert weighted_scores.shape == (B, I)

        # - group[i]
        #   corresponds to the summation group id for the i-th interval.
        #   With "breslow", this is a unique id for (batch, strata, stop).
        #   With "efron", this is a unique id (batch, strata, stop, event).
        assert group.shape == (I,)

        # - dataset.unique_groups[:, t]
        #   is a vector of (batch, strata, stop) values for the t-th group id.
        assert dataset.unique_groups.shape == (3, dataset.n_groups)

        # - (batch > strata > stop > event) is lexicographically sorted:
        assert (
            dataset.is_sorted
        ), "The dataset must be sorted before computing a log-likelihood."


        if ties == "breslow":
            # This is the term:
            #
            # Sum_{strata} ( Sum_{death times t} (              (***)
            #       (Sum_{dead at t} w[i])                      (**)
            #       *
            #       log( Sum_{observed at t} r[i] )             (*)
            # ))
            #
            #
            # that we compute in parallel over:
            # - all n_bootstrap values of the scores,
            # - all n_batch values of the parameter vector.

            # (*) Log-Sum-Exp over the death times,  -------------------------------------
            # in parallel for bootstraps, batches, strata and stop:
            # group corresponds to the values of (batch, strata, stop).
            # groups_scores is (n_bootstraps,n_death_times)
            group_scores = group_logsumexp(
                values=weighted_scores,
                groups=group,
                output_size=n_groups,
            )
            assert weight_factor.shape == (B, n_groups)
            assert group_scores.shape == (B, n_groups)

            # If mode == "unit length", risks sets exactly correspond to our summation
            # groups so we can move on to the next step.
            # However, if mode == "start zero" or "any", risks sets correspond to
            # unions of these summation groups so we need to aggregate the group scores
            # to compute the true "risk set scores".

            if mode == "start zero":
                # At this point, suppose e.g. that data.unique_groups
                # i.e. the values for (batch, strata, stop) is equal to:
                # [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                #  [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 2],
                #  [3, 4, 5, 2, 4, 5, 1, 2, 3, 3, 4, 2]]
                #   a  b  c| d  e  f| g  h  i| k  l| m
                # -> 5 unique values of (batch, strata)
                #
                #
                # Since the "start" of all intervals is equal to 0, the risk sets
                # within each "independent group" are
                #
                # - Group 1: a+b+c, b+c, c
                # - Group 2: d+e+f, e+f, f
                # - Group 3: g+h+i, h+i, i
                # - Group 4: k+l, l
                # - Group 5: m
                #
                # We implement this using a cumulative logsumexp.
                # Note that risks sets shrink over time (at patients die),
                # so we need to compute cumsums in "reverse order".

                # 1) Compute the (log)cumsum(exp)
                # [a+b+c+d+..., b+c+d+..., ..., k+l+m, l+m, m]
                # Since cumsum starts from the first index and we are interested
                # in "backward" sums, we must flip the tensors along the "n_group" dim:
                cumsums = (
                    group_scores.flip(dims=(1,)).logcumsumexp(dim=1).flip(dims=(1,))
                )
                assert cumsums.shape == (B, n_groups)

                # 2) Compute the offsets that correspond to the different
                #    "sums over independent groups":
                #    [(d+e+f) + (g+...), (g+h+i) + ..., (k+l) + m, m]
                #   These are the values of cumsum that correspond to the
                #   "first" (reading from left to right) indices of a new group.
                #   We do not care about the very first value, (a+b+c)+...

                assert dataset.unique_groups.shape == (3, n_groups)

                # batch_strata_group looks like:
                # [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4]
                _, batch_strata_group = torch.unique_consecutive(
                    dataset.unique_groups[0:2],  # (batch, strata)
                    return_inverse=True,
                    dim=-1,
                )
                assert batch_strata_group.shape == (n_groups,)
                # Recall that batch_strata_group is of length n_groups,
                # not n_intervals.

                assert n_groups > 0, "With a non-empty dataset, we always have n_groups >= 1."

                # Identify the indices of the first (batch, strata, stop) group
                # for each value of (batch, strata):
                # ., [F, F, T, F, F, T, F, F, T, F, T]
                first_in_batch_strata_group = (
                    batch_strata_group[1:] != batch_strata_group[:-1]
                )

                # Fetch the values of the cumsum at this stage:
                offsets_per_batch_strata = cumsums[:, 1:][
                    :, first_in_batch_strata_group
                ]

                # At this stage, on every row, offsets_per_group is:
                # [(d+e+f) + (g+...), (g+h+i) + ..., (k+l) + m, m]
                # N.B.: Since we have discarded the cumsum over the full array,
                #       if there is only one value for (batch, strata),
                #       offsets_per_group = tensor([]) !

                # 3) Add an arbitrary "offset" value for the last group.
                #    This is to avoid indexing on empty tensors, but won't be used.
                offsets_per_batch_strata = torch.cat(
                    (
                        offsets_per_batch_strata,
                        torch.zeros_like(group_scores[:, :1]),
                    ),  # (B, 1)
                    dim=1,
                )
                assert offsets_per_batch_strata.shape == (B, batch_strata_group[-1] + 1)

                # 4) Unwrap this offset into a tensor of shape (batch, n_groups).
                #    On every row:
                #   [(d+e+f)+..., idem, idem, (g+h+i)+..., ..., m, m, 0]
                offsets_per_group = offsets_per_batch_strata[:, batch_strata_group]
                assert offsets_per_group.shape == (B, n_groups)

                # 5) We now want to subtract the offsets from the cumsums.
                #    This is not trivial, because we are dealing with logsumexps
                #    instead of sums. We use the following identity:
                #    if a > b,
                #    log(e^a - e^b) = log( e^a  * (1 - e^(b-a)))
                #                   = a + log(1 - e^(b-a))
                def log1mexp(x):
                    """Numerically accurate evaluation of log(1 - exp(x)) for x < 0.

                    See https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf for details.

                    We rely on numerically stable implementations of
                    [x -> log(1+x)] and [x -> exp(x)-1] for x close to 0.

                    If -log(2) < x < 0, we use the following identity:
                    log(1 - exp(x)) = log(-(exp(x) - 1))

                    If x <= -log(2), we use the following identity:
                    log(1 - exp(x)) = log1p(-exp(x))
                    """
                    mask = -np.log(2) < x  # x < 0
                    return torch.where(
                        mask,
                        (-x.expm1()).log(),
                        (-x.exp()).log1p(),
                    )

                # For the last group (the right-most one), there is no offset:
                last_batch_strata = batch_strata_group == batch_strata_group[-1]
                last_batch_strata = last_batch_strata.view(n_groups)

                assert torch.all(
                    cumsums[:, ~last_batch_strata]
                    > offsets_per_group[:, ~last_batch_strata]
                )

                group_scores = torch.where(
                    last_batch_strata,
                    cumsums,
                    cumsums + log1mexp(offsets_per_group - cumsums),
                )
                assert group_scores.shape == (B, n_groups)

            # (**) Product with (Sum_{dead at t} w[i]): ----------------------------------
            lse = weight_factor * group_scores
            assert lse.shape == (B, n_groups)

            # (***) Sum over strata and the death time stop, -----------------------------
            # in parallel for bootstraps and batches:
            lse = group_sum(
                values=lse,
                # "batch" value for each unique (batch, strata, stop) triplet
                groups=dataset.unique_groups[0],
                output_size=dataset.n_batch,
            )

            assert lse.shape == (B, dataset.n_batch)

        # TODO: Update Efron too!
        elif ties == "efron":
            msg = "We are currently re-writing the Efron approximation rule with support for batches and strata."
            raise NotImplementedError(msg)
            # groups_scores is (B,T*2)
            group_scores = group_logsumexp(
                values=weighted_scores,
                groups=groups,
                output_size=T * 2,
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

        # lin and lse are (n_bootstrap, n_batch)
        ret_value = lse - lin  # (n_bootstrap, n_batch) values, computed in parallel
        return ret_value.view(B * dataset.n_batch)

    return negloglikelihood
