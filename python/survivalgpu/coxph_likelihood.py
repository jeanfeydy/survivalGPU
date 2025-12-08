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

# Use PyTorch for fast array manipulations (on the GPU):
import torch

from .bootstrap import Resampling
from .group_reduction import (
    clip_inf,
    first_in_segment,
    group_sum,
    keys_to_segments,
    rank_in_segment,
    segment_cumsum,
)
from .typecheck import Float32Tensor, Int64Tensor, Literal, typecheck


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
    interval_weighted_risks: Float32Tensor["bootstraps intervals"],
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
     - a weighted risk that corresponds to risk = weight * exp( dot(beta, x) ).

    We aggregate these values into a "time-indexed" data table:
    for every bootstrap b, data for interval (start, stop] is aggregated at locations
    [b, stop, 0, event] and [b, start, 1, event],
    where event == 0 if the interval is "censored" and event == 1 if it ends with an event.

    The reduction for counts, weights and weighted risks is a sum.

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
        interval_weighted_risks = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        index_start = torch.tensor([0, 1, 2])
        index_stop = torch.tensor([1, 3, 3])
        event = torch.tensor([0, 1, 1])
        T = 4  # Number of unique time points

        time_counts, time_weights, time_weighted_risks = _compute_time_data(
            interval_counts=interval_counts,
            interval_weights=interval_weights,
            interval_weighted_risks=interval_weighted_risks,
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

        print(time_weighted_risks.view(2, -1))

    .. testoutput::

        tensor([[0., 0., 1., 0., 1., 0., 0., 2., 0., 0., 0., 3., 0., 5., 0., 0.],
                [0., 0., 2., 0., 2., 0., 0., 3., 0., 0., 0., 4., 0., 7., 0., 0.]])

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

    time_weighted_risks = group_sum(
        values=torch.cat((interval_weighted_risks,) * 2, dim=1),
        groups=full_index,
        output_size=T * 4,
    ).view(B, T, 2, 2)

    return time_counts, time_weights, time_weighted_risks




@typecheck
def _intervals_to_time_data(
    *,
    interval_counts: Int64Tensor["bootstraps intervals"],
    interval_weights: Float32Tensor["bootstraps intervals"],
    interval_risks: Float32Tensor["bootstraps intervals"],
    batch: Int64Tensor["intervals"],
    strata: Int64Tensor["intervals"],
    start: Int64Tensor["intervals"],
    stop: Int64Tensor["intervals"],
    event: Int64Tensor["intervals"],
) -> tuple[
    Int64Tensor["bootstraps times 2 2"],  # Counts of intervals at each time
    Float32Tensor["bootstraps times 2 2"],  # Weights of intervals at each time
    Float32Tensor["bootstraps times 2 2"],  # Weighted risks at each time
    Int64Tensor["3 times"],  # Unique (batch, strata, time) values
]:
    """Aggregates interval data into a table indexed by time.

    The format of the output is the same as in _compute_time_data(),
    plus the unique (batch, strata, time) values.

    .. testcode::

        import torch
        from survivalgpu.coxph_likelihood import _intervals_to_time_data

        time_counts, time_weights, time_weighted_risks, unique_batch_strata_time = (
            _intervals_to_time_data(
                interval_counts=torch.tensor([[1, 1, 3], [2, 2, 1]]),
                interval_weights=torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
                interval_risks=torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
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

        print(time_weighted_risks.view(2, -1))

    .. testoutput::

        tensor([[ 0.,  0.,  1., 13.,  1.,  0.,  0.,  0.,  0., 13.,  0.,  0.],
                [ 0.,  0.,  4., 25.,  4.,  0.,  0.,  0.,  0., 25.,  0.,  0.]])

    .. testcode::

        print(unique_batch_strata_time)

    .. testoutput::

        tensor([[0, 0, 0],
                [0, 0, 0],
                [0, 1, 2]])

    """
    B, I = interval_risks.shape

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

    # r[b,i] = w[b,i] * exp( dot(beta[b], x[i]) )
    interval_weighted_risks = interval_weights * interval_risks
    assert interval_weighted_risks.shape == (B, I)

    time_counts, time_weights, time_weighted_risks = _compute_time_data(
        interval_counts=interval_counts,
        interval_weights=interval_weights,
        interval_weighted_risks=interval_weighted_risks,
        index_start=index_start,
        index_stop=index_stop,
        event=event,
        T=T,
    )

    return (
        time_counts,
        time_weights,
        time_weighted_risks,
        unique_batch_strata_time,
    )


@typecheck
def _compute_time_risks(
    *,
    time_weighted_risks: Float32Tensor["bootstraps times 2 2"],
    unique_batch_strata_time: Int64Tensor["3 times"],
) -> Float32Tensor["bootstraps times"]:
    """Computes the "sum" risk over the full risk set of observed patients at each time point.

    .. warning::

        Currently, this is float32-based, which may lead to numerical errors
        when the number of time points T is very large (e.g. T > 10k).

    .. testcode::

        import torch
        from survivalgpu.coxph_likelihood import _compute_time_risks

        # First "bootstrap" corresponds to:
        #  - one interval (0, 1] with a risk of 1 and no event,
        #  - one interval (0, 2] with a risk of 2 and an event.
        # We expect the log-risks to be:
        #  - at time 0: 0 (no risk set),
        #  - at time 1: 1 + 2 = 3
        #  - at time 2: 2 = 2
        #
        # Second "bootstrap" corresponds to:
        #  - one interval (0, 2] with a risk of 1 and no event,
        #  - one interval (1, 2] with a risk of 3 and an event.
        # We expect the log-risks to be:
        #  - at time 0: 0 (no risk set),
        #  - at time 1: 1 = 1
        #  - at time 2: 1 + 3 = 4
        #
        # We also add an empty strata at the end.

        time_risks = _compute_time_risks(
            time_weighted_risks=torch.tensor(
                [
                    [
                        [[0.0, 0.0], [1.0, 2.0]],
                        [[1.0, 0.0], [0.0, 0.0]],
                        [[0.0, 2.0], [0.0, 0.0]],
                        [[0.0, 0.0], [0.0, 0.0]],
                    ],
                    [
                        [[0.0, 0.0], [1.0, 0.0]],
                        [[0.0, 0.0], [0.0, 3.0]],
                        [[1.0, 3.0], [0.0, 0.0]],
                        [[0.0, 0.0], [0.0, 0.0]],
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
        print(time_risks)

    .. testoutput::

        tensor([[0., 3., 2., 0.],
                [0., 1., 4., 0.]])
    """

    B, T, _, _ = time_weighted_risks.shape

    # Reduce over the "no event / event" dimension
    time_risk_updates = clip_inf(time_weighted_risks).sum(dim=-1)
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

    # TODO: implement this with + and - signs in a single cumsum.
    #       This should help with numerical precision.
    # Recall that with our convention, the "stop" index is 0 and the "start" index is 1
    # along the 3rd dimension of our time data tables.
    time_risk_set_stop = segment_cumsum(
        values=time_risk_updates[:, :, 0],
        segments=batch_strata_segments,
    )
    time_risk_set_start = segment_cumsum(
        values=time_risk_updates[:, :, 1],
        segments=batch_strata_segments,
    )

    # At time t, the "weighted risk" over the risk set
    #    Sum_{observed at t} r[i]
    # is equal to the difference:
    #    Sum_{started at time < t} r[i]
    #  - Sum_{stopped at time < t} r[i]
    time_risks = time_risk_set_start - time_risk_set_stop
    assert time_risks.shape == (B, T)

    # We shift these risks to the right by one time step,
    # in order to compensate for the "< t" condition above.
    time_risks = torch.cat(
        (
            torch.zeros_like(time_risks[:, :1]),
            time_risks[:, :-1],
        ),
        dim=1,
    )
    assert time_risks.shape == (B, T)

    # N.B.: This shift fills the first time step of every segment
    #       with a very small value (theoretically equal to 0
    #       since every interval appears once in "start" and once in "stop").
    #       This is not a problem, since the first time step
    #       of every segment can only correspond to a "start" time,
    #       not a "stop" time, and therefore does not contribute
    #       to the log-sum-exp term of the CoxPH objective.

    return time_risks

@typecheck
def _compute_efron_data(
    event_counts: Int64Tensor["bootstraps times"],
) -> tuple[
    Int64Tensor["events"],  # indices in [0, B*T)
    Int64Tensor["events"],  # bootstraps in [0, B)
    Int64Tensor["events"],  # event_counts
    Float32Tensor["events"],  # offsets, i.e. k / {number of deaths at t}
]:
    """Computes the information required to re-index our time tables for Efron summation.

    .. testcode::

        import torch
        from survivalgpu.coxph_likelihood import _compute_efron_data

        efron_indices, efron_bootstraps, efron_event_counts, efron_log_offsets = (
            _compute_efron_data(torch.tensor([[0, 1, 2], [3, 0, 2]]))
        )

        print(efron_indices)

    .. testoutput::

        tensor([1, 2, 2, 3, 3, 3, 5, 5])

    .. testcode::

        print(efron_bootstraps)

    .. testoutput::

        tensor([0, 0, 0, 1, 1, 1, 1, 1])

    .. testcode::

        print(efron_event_counts)

    .. testoutput::

        tensor([1, 2, 2, 3, 3, 3, 2, 2])

    .. testcode::

        print(efron_log_offsets)

    .. testoutput::

        tensor([0.0000, 0.0000, 0.5000, 0.0000, 0.3333, 0.6667, 0.0000, 0.5000])

    """
    efron_indices = torch.repeat_interleave(event_counts.view(-1))
    E = len(efron_indices)
    assert event_counts.sum().item() == E
    assert efron_indices.shape == (E,)
    assert efron_indices.dtype == torch.int64

    B, T = event_counts.shape
    assert B > 0
    assert T > 0
    efron_bootstraps = efron_indices // T
    assert efron_bootstraps.shape == (E,)
    assert efron_bootstraps.dtype == torch.int64
    assert (efron_bootstraps >= 0).all()
    assert (efron_bootstraps < B).all()

    efron_event_counts = torch.index_select(
        event_counts.view(-1),  # Flatten the event_counts tensor
        dim=0,
        index=efron_indices,
    )
    assert efron_event_counts.shape == (E,)
    assert efron_event_counts.dtype == torch.int64
    assert (efron_event_counts > 0).all()

    efron_offsets = rank_in_segment(efron_indices) / efron_event_counts.float()
    assert efron_offsets.shape == (E,)
    assert efron_offsets.dtype == torch.float32

    return efron_indices, efron_bootstraps, efron_event_counts, efron_offsets



@typecheck
def _breslow_efron_logsumexp_term(
    *,
    time_counts: Int64Tensor["bootstraps times 2 2"],
    time_weights: Float32Tensor["bootstraps times 2 2"],
    time_weighted_risks: Float32Tensor["bootstraps times 2 2"],
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
        time_weighted_risks = torch.tensor(
            [
                [[[0.0, 0.0], [2.0, 3.0]], [[2.0, 3.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [2.0, 1.0]], [[2.0, 1.0], [0.0, 0.0]]],
            ]
        )

        # The contribution at time 0 is 0, since there are no deaths at that time.
        # At time 1, with the Breslow approximation for ties, we expect:
        # - for bootstrap 1, 1 * log(2 + 3) = 1.6094
        # - for bootstrap 2, 2 * log(2 + 1) = 2.1972
        print(
            _breslow_efron_logsumexp_term(
                time_counts=time_counts,
                time_weights=time_weights,
                time_weighted_risks=time_weighted_risks,
                unique_batch_strata_time=unique_batch_strata_time,
                ties="breslow",
                n_batches=1,
            )
        )

    .. testoutput::

        tensor([[1.6094],
                [2.1972]])

    .. testcode::

        # With the Efron approximation, we expect:
        # - for bootstrap 1, 1 * log(2 + 3 - (0 / 1) * 3) = 1.6094
        # - for bootstrap 2, 2 / 2 * [
        #       log(2 + 1 - (0 / 2) * 1)
        #     + log(2 + 1 - (1 / 2) * 1)
        #     ] = 2.0149
        print(
            _breslow_efron_logsumexp_term(
                time_counts=time_counts,
                time_weights=time_weights,
                time_weighted_risks=time_weighted_risks,
                unique_batch_strata_time=unique_batch_strata_time,
                ties="efron",
                n_batches=1,
            )
        )

    .. testoutput::

        tensor([[1.6094],
                [2.0149]])

    """

    B, T, _, _ = time_counts.shape

    # Compute "Sum_{observed at t} r[i]"
    time_risks =  _compute_time_risks(
        time_weighted_risks=time_weighted_risks,
        unique_batch_strata_time=unique_batch_strata_time,
    )
    assert time_risks.shape == (B, T)

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
        safe_log = (time_risks + 1e-12).log() # add a safe value to the log to avoid nan issues
        time_contributions = dead_weights * safe_log

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
        # The Efron approximation is more complex - recall that we are computing:
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

        # Extract the "number of deaths at t" from our table:
        # 0 on the 3rd dimension corresponds to "stop" time,
        # 1 on the 4th dimension corresponds to "event".
        event_counts = time_counts[:, :, 0, 1]
        assert event_counts.shape == (B, T)
        assert event_counts.dtype == torch.int64
        assert (event_counts >= 0).all()

        # Extract the "Sum_{dead at t} r[i]" from our table:
        # 0 on the 3rd dimension corresponds to "stop" time,
        # 1 on the 4th dimension corresponds to "event".
        dead_risks = time_weighted_risks[:, :, 0, 1]
        assert dead_risks.shape == (B, T)
        assert dead_risks.dtype == torch.float32

        # We use repeat_interleave to re-index our dataset.
        # E is the total number of deaths on the full table.
        efron_indices, efron_bootstraps, efron_event_counts, efron_offsets = _compute_efron_data(event_counts)
        E = len(efron_indices)
        assert event_counts.sum().item() == E
        # No-death times should not be present in efron_indices:
        assert (efron_event_counts > 0).all()
        assert efron_indices.shape == (E,)
        assert efron_event_counts.shape == (E,)
        assert efron_offsets.shape == (E,)

        # efron_indices is a (E,) Tensor of int64 that records the indices of the
        # "death" times over the flattened time table.
        # Since risk sets have varying sizes, we cannot work with a separate "Bootstrap"
        # dimension, and use flat vectors instead.
        efron_dead_risks = torch.index_select(
            dead_risks.view(-1),  # Flatten the time table
            dim=0,
            index=efron_indices,
        )
        assert efron_dead_risks.shape == (E,)
        assert efron_dead_risks.dtype == torch.float32

        # The Efron offsets correspond to "k / {number of deaths at t}".
        # We use them to compute
        # (k / {number of deaths at t}) * Sum_{dead at t} r[i]
        efron_dead_risks = efron_dead_risks * efron_offsets
        assert efron_dead_risks.shape == (E,)
        assert efron_dead_risks.dtype == torch.float32
        assert not efron_dead_risks.isnan().any()

        # Likewise, we compute the "Sum_{observed at t} r[i]"
        efron_observed_risks = torch.index_select(
            time_risks.view(-1),  # Flatten the time table
            dim=0,
            index=efron_indices,
        )
        assert efron_observed_risks.shape == (E,)
        assert efron_observed_risks.dtype == torch.float32

        # Compute log(
        #             Sum_{observed at t} r[i]
        #             -
        #             (k / {number of deaths at t})
        #             *
        #             Sum_{dead at t} r[i]
        #         )
        efron_log_risks = (efron_observed_risks - efron_dead_risks).log()
        assert efron_log_risks.shape == (E,)
        assert efron_log_risks.dtype == torch.float32
        assert not efron_log_risks.isnan().any()

        # Compute the weight factor
        # (Sum_{dead at t} w[i]) / {number of deaths at t)
        efron_dead_weights = torch.index_select(
            dead_weights.view(-1),  # Flatten the time table
            dim=0,
            index=efron_indices,
        )
        assert efron_dead_weights.shape == (E,)
        assert efron_dead_weights.dtype == torch.float32

        efron_factor = efron_dead_weights / efron_event_counts.float()
        assert efron_factor.shape == (E,)
        assert efron_factor.dtype == torch.float32
        assert not efron_factor.isnan().any()

        # Compute the contributions at each sub-time, i.e.
        # (Sum_{dead at t} w[i]) / {number of deaths at t}
        # * log(
        #             Sum_{observed at t} r[i]
        #             -
        #             (k / {number of deaths at t})
        #             *
        #             Sum_{dead at t} r[i]
        #       )
        efron_contributions = efron_factor * efron_log_risks
        assert efron_contributions.shape == (E,)
        assert efron_contributions.dtype == torch.float32

        # When efron_factor == 0, the contribution is 0, even if efron_log_risks is -inf.
        # If we don't mask things out, we would end up with -inf * 0 == NaN.
        efron_contributions = torch.where(
            efron_factor != 0,
            efron_contributions,
            torch.zeros_like(efron_contributions),
        )
        assert efron_contributions.shape == (E,)

        # Finally, we sum batch-wise over the contributions:
        efron_batch = torch.index_select(
            unique_batch_strata_time[0].repeat(B),
            dim=0,
            index=efron_indices,
        )
        assert efron_batch.shape == (E,)
        assert efron_batch.dtype == torch.int64

        # We must add an offset to prevent mixing bootstraps with each other
        efron_batch = efron_batch + n_batches * efron_bootstraps
        assert efron_batch.shape == (E,)
        assert efron_batch.dtype == torch.int64

        logsumexp_term = group_sum(
            values=efron_contributions.view(1, E),
            groups=efron_batch.view(E),
            output_size=B * n_batches,
        ).view(B, n_batches)

    return logsumexp_term


@typecheck
def _linear_term(
    *,
    scores: Float32Tensor["bootstraps intervals"],
    interval_weights: Float32Tensor["bootstraps intervals"],
    event: Int64Tensor["intervals"],
    batch: Int64Tensor["intervals"],
    n_batches: int,
) -> Float32Tensor["bootstraps batches"]:
    """Computes the term "Sum_{all dead samples} w[i] * dot(x[i], b)"

    .. testcode::

        import torch
        from survivalgpu.coxph_likelihood import _linear_term

        print(
            _linear_term(
                scores=torch.tensor(
                    [
                        [1.0, 2.0, 3.0, 4.0],
                        [2.0, 3.0, 4.0, 5.0],
                    ]
                ),
                interval_weights=torch.tensor(
                    [
                        [1.0, 2.0, 2.0, 1.0],
                        [2.0, 1.0, 3.0, 2.0],
                    ]
                ),
                event=torch.tensor([0, 1, 1, 1]),
                batch=torch.tensor([0, 0, 0, 1]),
                n_batches=2,
            )
        )

    .. testoutput::

        tensor([[10.,  4.],
                [15., 10.]])


    """

    B, I = scores.shape
    weighted_scores = interval_weights * scores * event.float().view(1, I)
    assert weighted_scores.shape == (B, I)
    assert weighted_scores.dtype == torch.float32

    linear_term = group_sum(
        values=weighted_scores,
        groups=batch,
        output_size=n_batches,
    )
    assert linear_term.shape == (B, n_batches)
    assert linear_term.dtype == torch.float32
    return linear_term


@typecheck
def coxph_objective_from_scores(
    *,
    scores: Float32Tensor["bootstraps intervals"],
    dataset,  #: TorchSurvivalDataset, omitted to avoid circular import
    ties: Literal["efron", "breslow"],
    bootstrap: Resampling,
    mode: Literal["unit length", "start zero", "any"] = "any",
) -> Float32Tensor["bootstraps batches"]:
    """Implements the CoxPH loss function.

    This function takes as input a batch of score values scores[i, j],
    each of whom corresponds to the risk score of the j-th interval
    according to the i-th estimate of the model parameters.

    For linear scores "dot(beta[i], x[j])", this corresponds to
    scores = beta @ x.T, (B,D) @ (D,I) = (B,I)

    .. testcode::

        import torch
        from survivalgpu.coxph_likelihood import coxph_objective_from_scores
        from survivalgpu.torch_datasets import TorchSurvivalDataset
        from survivalgpu.bootstrap import Resampling

        # Example dataset with:
        # - 3 patients that die in the first strata of batch 0
        # - 2 patients that die in the second strata of batch 0
        # - 1 patient that dies + 1 that is censored in the first strata of batch 1
        dataset = TorchSurvivalDataset(
            patient=torch.tensor([0, 1, 2, 3, 4, 5, 6]),
            batch=torch.tensor([0, 0, 0, 0, 0, 1, 1]),
            strata=torch.tensor([0, 0, 0, 1, 1, 0, 0]),
            start=torch.tensor([0, 0, 0, 0, 0, 0, 0]),
            stop=torch.tensor([1, 1, 1, 1, 1, 1, 1]),
            event=torch.tensor([1, 1, 1, 1, 1, 0, 1]),
            # Note that the covariates are not used directly in this objective function,
            # but only via the scores that come later in this example.
            covariates=torch.zeros(7, 2, dtype=torch.float32),
        ).sort()

        print(dataset.patient)

    .. testoutput::

        tensor([0, 1, 2, 3, 4, 5, 6])

    .. testcode::

        bootstrap = Resampling(
            indices=torch.tensor(
                [
                    [0, 1, 2, 3, 4, 5, 6],  # All patients
                    [0, 0, 0, 0, 0, 5, 5],  # Just the first patient in each batch
                ]
            ),
            patient=dataset.patient,
        )

        scores = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],  # Scores for bootstrap 1
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],  # Scores for bootstrap 2
            ]
        )

        # With the Breslow approximation, for the first bootstrap, we expect:
        #
        # 1. For the linear term:
        #    a. (s[0] + s[1] + s[2] + s[3] + s[4]) = 10 for batch 0,
        #    b. (s[6]) = 6 for batch 1.
        #
        # 2. For the log-sum-exp term:
        #    a. For the first strata of batch 0:
        #       3 * lse(s[0], s[1], s[2]) = 7.2228
        #    b. For the second strata of batch 0:
        #       2 * lse(s[3], s[4]) = 8.6265
        #    c. For the first strata of batch 1:
        #       1 * lse(s[5], s[6]) = 6.3133
        #
        # So, overall:
        #    a. 7.2228 + 8.6265 - 10 = 5.8493 for batch 0,
        #    b. 6.3133 - 6 = 0.3133 for batch 1.
        #
        # For the second bootstrap, we expect:
        #
        # 1. For the linear term:
        #    a. 5 * s[0] = 5 for batch 0,
        #    b. 0 since no event occurs for batch 1
        #
        # 2. For the log-sum-exp term:
        #    a. For the first strata of batch 0:
        #       5 * log(exp(s[0]) * 5) = 13.0472
        #    b. For the first strata of batch 1:
        #       0 since no event occurs.
        #
        # So, overall:
        #    a. 13.0472 - 5 = 8.0472 for batch 0,
        #    b. 13.3863 - 12 = 1.3863 for batch 1.

        print(
            coxph_objective_from_scores(
                scores=scores,
                dataset=dataset,
                bootstrap=bootstrap,
                ties="breslow",
            )
        )

    .. testoutput::

        tensor([[5.8493, 0.3133],
                [8.0472, 0.0000]])

    .. testcode::

        # With the Efron approximation, linear terms remain the same,
        # but log-sum-exp terms change slightly.
        #
        # For the first bootstrap, we expect:
        #    a. For the first strata of batch 0:
        #       (3 / 3) * (
        #             lse(s[0], s[1], s[2])
        #           + lse(s[0], s[1], s[2]) + log(2 / 3)
        #           + lse(s[0], s[1], s[2]) + log(1 / 3)
        #       ) = 5.7187
        #    b. For the second strata of batch 0:
        #       (2 / 2) * (
        #             lse(s[3], s[4])
        #           + lse(s[3], s[4]) + log(1 / 2)
        #       ) = 7.9334
        #    c. For the first strata of batch 1:
        #       (1 / 1) * lse(s[5], s[6]) = 6.3133
        #
        # So, overall:
        #    a. 5.7187 + 7.9334 - 10 = 3.6521 for batch 0,
        #    b. 6.3133 - 6 = 0.3133 for batch 1.
        #
        # For the second bootstrap, we expect:
        #    a. For the first strata of batch 0:
        #       (5 / 5) * (
        #             lse(s[0], s[0], s[0], s[0], s[0])
        #           + lse(s[0], s[0], s[0], s[0], s[0]) + log(1 / 5)
        #           + lse(s[0], s[0], s[0], s[0], s[0]) + log(2 / 5)
        #           + lse(s[0], s[0], s[0], s[0], s[0]) + log(3 / 5)
        #           + lse(s[0], s[0], s[0], s[0], s[0]) + log(4 / 5)
        #       ) = 9.7875
        #    b. For the first strata of batch 1:
        #       0 since no event occurs.
        # So, overall:
        #    a. 9.7875 - 5 = 4.7875 for batch 0,
        #    b. 0 - 0 = 0 for batch 1.

        print(
            coxph_objective_from_scores(
                scores=scores,
                dataset=dataset,
                bootstrap=bootstrap,
                ties="efron",
            )
        )

    .. testoutput::

        tensor([[3.6521, 0.3133],
                [4.7875, 0.0000]])

    """

    B = len(bootstrap)  # Number of bootstraps to process in parallel
    assert mode in ("unit length", "start zero", "any")

    if not dataset.is_sorted:
        msg = "The dataset must be sorted by (batch, strata, start, stop)."
        raise ValueError(msg)

    # Compute the linear term of the CoxPH objective
    linear_term = _linear_term(
        scores=scores,
        interval_weights=bootstrap.interval_weights,
        event=dataset.event,
        batch=dataset.batch_intervals,
        n_batches=dataset.n_batch,
    )
    assert linear_term.shape == (B, dataset.n_batch)

    # The CoxPH model assumes an exponential relationship between the score and the risk
    interval_risks = scores.exp()

    # Aggregate the risks into time-indexed data tables
    time_counts, time_weights, time_weighted_risks, unique_batch_strata_time = (
        _intervals_to_time_data(
            interval_counts=bootstrap.interval_counts,
            interval_weights=bootstrap.interval_weights,
            interval_risks=interval_risks,
            batch=dataset.batch_intervals,
            strata=dataset.strata_intervals,
            start=dataset.start,
            stop=dataset.stop,
            event=dataset.event,
        )
    )

    # Compute the log-sum-exp term for the Breslow or Efron approximation
    logsumexp_term = _breslow_efron_logsumexp_term(
        time_counts=time_counts,
        time_weights=time_weights,
        time_weighted_risks=time_weighted_risks,
        unique_batch_strata_time=unique_batch_strata_time,
        ties=ties,
        n_batches=dataset.n_batch,
    )
    assert logsumexp_term.shape == (B, dataset.n_batch)

    return logsumexp_term - linear_term



@typecheck
def linear_risk_scores(
    *,
    coef: Float32Tensor["bootstraps batches covariates"],
    dataset, #: TorchSurvivalDataset, omitted to avoid circular import
) -> Float32Tensor["bootstraps intervals"]:
    """Standard function to compute risks in the CoxPH model: dot(beta, x[i])."""
    B, n_batches, D = coef.shape
    I = dataset.n_intervals

    assert dataset.n_covariates == D
    assert n_batches == dataset.n_batch

    if n_batches == 1:
        # Simple case with no batch - don't waste time with indexing operations:
        scattered_coef = coef.view(B, 1, D)

    else:
        # N.B.: Naive implementation with an indexing operation as in
        # scattered_coef = coef[:, dataset.batch_intervals, :]  # (B, I, D)
        # is MASSIVELY inefficient in the backward pass, as discussed in
        # https://github.com/pytorch/pytorch/issues/41162
        # https://github.com/dmlc/dgl/issues/3729
        #
        # Instead, we prefer the following line, with a non-deterministic backward pass:
        scattered_coef = torch.index_select(coef, 1, dataset.batch_intervals)
        assert scattered_coef.shape == (B, I, D)

    X = dataset.covariates  # (I, D)
    assert X.shape == (I, D)

    # [(B, 1, D) or (B, I, D)] * (1, I, D) -> (B, I, D)
    scores = scattered_coef * X.view(1, I, D)
    assert scores.shape == (B, I, D)

    scores = scores.sum(-1)  # (B, I, D) -> (B, I)
    assert scores.shape == (B, I)
    return scores


@typecheck
def coxph_objective(
    *,
    coef: Float32Tensor["bootstraps batches covariates"],
    scales: Float32Tensor["covariates"] | None,
    dataset,  #: TorchSurvivalDataset, omitted to avoid circular import
    ties: Literal["efron", "breslow"],
    bootstrap: Resampling,
    l2_reg: int | float,
    mode: Literal["unit length", "start zero", "any"] = "any",
) -> Float32Tensor["bootstraps batches"]:
    """Implements the CoxPH objective.

    This function is a wrapper around coxph_objective_from_scores() that computes
    the risk scores from the model parameters and the dataset covariates.
    """

    B, n_batches, D = coef.shape
    I = dataset.n_intervals
    assert len(bootstrap) == B
    assert n_batches == dataset.n_batch
    assert dataset.n_covariates == D

    scores = linear_risk_scores(coef=coef, dataset=dataset)
    assert scores.shape == (B, I)

    # Vanilla CoxPH objective
    obj = coxph_objective_from_scores(
        scores=scores,
        dataset=dataset,
        ties=ties,
        bootstrap=bootstrap,
        mode=mode,
    )
    assert obj.shape == (B, n_batches)

    # L2 regularization term
    if scales is None:
        scaled_coef = coef
    else:
        assert scales.shape == (D,)
        scaled_coef = coef * scales

    reg = l2_reg * (scaled_coef**2).sum(dim=-1)
    assert reg.shape == (B, n_batches)

    return obj + reg
