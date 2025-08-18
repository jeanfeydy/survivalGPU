# Use PyTorch for fast array manipulations (on the GPU):
import numpy as np
import torch

from .typecheck import BoolTensor, Float32Tensor, Int64Tensor, Literal, typecheck

# ========================================================================================
#                       "sum" operations over arbitrary groups
# ========================================================================================

@typecheck
def make_2d(
        *,
        groups: Int64Tensor["values"] | Int64Tensor["bootstraps values"],
        values: Int64Tensor["bootstraps values"] | Float32Tensor["bootstraps values"],
    ) -> Int64Tensor["bootstraps values"]:
    """Makes sure that the groups tensor is 2D and has the same shape as the values tensor.

    This works around the fact that torch.scatter_reduce_ fails silently if
    the index tensor does not have the same shape as the values tensor.
    """
    if len(groups.shape) == 1:
        groups = groups.view(1, -1).repeat(values.shape[0], 1)

    assert groups.shape == values.shape
    return groups


class SumTorch(torch.autograd.Function):
    """Custom autograd function to compute the group-wise sum reduction.

    This works around a huge bottleneck in the backward pass because of the use of
    a deterministic algorithm in the indexing_backward_kernel:
    https://github.com/pytorch/pytorch/issues/41162
    https://github.com/dmlc/dgl/issues/3729

    Instead of indexing, we use the (much faster) operation index_select.
    """

    @staticmethod
    @typecheck
    def forward(
            ctx,
            values: Int64Tensor["bootstraps values"] | Float32Tensor["bootstraps values"],
            groups: Int64Tensor["values"] | Int64Tensor["bootstraps values"],
            output_size: int,
        ) -> Int64Tensor["bootstraps {output_size}"] | Float32Tensor["bootstraps {output_size}"]:
        """Forward pass for the group-wise sum reduction."""

        ctx.save_for_backward(groups)
        ctx.output_size = output_size
        return group_reduce(
            values=values,
            groups=groups,
            reduction="sum_forward_pass",
            output_size=output_size,
        )


    @staticmethod
    @typecheck
    def backward(
            ctx,
            grad_output: Int64Tensor["bootstraps output_size"] | Float32Tensor["bootstraps output_size"],
        ):
        """Fast but non-deterministic backward pass for the group-wise sum reduction."""

        (groups,) = ctx.saved_tensors
        assert len(groups.shape) == 1, "Backward pass only supports 1D groups."

        # TODO: what about output_size?
        return torch.index_select(grad_output, 1, groups), None, None


@typecheck
def group_reduce(
        *,
        values: Int64Tensor["bootstraps values"] | Float32Tensor["bootstraps values"],
        groups: Int64Tensor["values"] | Int64Tensor["bootstraps values"],
        reduction: Literal["max", "sum", "sum_forward_pass"],
        output_size: int,
    ) -> Int64Tensor["bootstraps {output_size}"] | Float32Tensor["bootstraps {output_size}"]:
    """Group-wise reduction of the values tensor.

    This is a wrapper around the torch.scatter_reduce_ function.
    Cells of the output tensor that do not correspond to a group
    are set to 0.

    .. testcode::

        import torch
        from survivalgpu.group_reduction import group_reduce

        reduced = group_reduce(
            values=torch.tensor([[1, 2, 3], [2, 3, 4]]),
            groups=torch.tensor([0, 2, 2]),
            reduction="max",
            output_size=4,
        )
        print(reduced)

    .. testoutput::

        tensor([[1, 0, 3, 0],
                [2, 0, 4, 0]])

    """

    if reduction == "sum":
        # We need to use a fast but non-deterministic backward pass.
        reduced = SumTorch.apply(values, groups, output_size)
    else:
        if reduction == "max":
            reduction = "amax"
        elif reduction == "sum_forward_pass":
            reduction = "sum"

        reduced = torch.zeros(
            values.shape[0], output_size, dtype=values.dtype, device=values.device
        ).scatter_reduce_(
            dim=1,
            index=make_2d(groups=groups, values=values),
            src=values,
            reduce=reduction,
            include_self=False,
        )

    assert reduced.shape == (values.shape[0], output_size)
    assert reduced.dtype == values.dtype
    return reduced


def group_expand(*, values, groups, output_size):  # noqa: ARG001
    # return torch.gather(values, 1, groups)
    return torch.index_select(values, 1, groups)

@typecheck
def group_sum(
        *,
        values: Int64Tensor["bootstraps values"] | Float32Tensor["bootstraps values"],
        groups: Int64Tensor["values"] | Int64Tensor["bootstraps values"],
        output_size: int,
    ) -> Int64Tensor["bootstraps {output_size}"] | Float32Tensor["bootstraps {output_size}"]:
    """Group-wise sum reduction.

    This is a wrapper around group_reduce with reduction="sum".

    .. testcode::

        import torch
        from survivalgpu.group_reduction import group_sum

        reduced = group_sum(
            values=torch.tensor([[1, 2, 3], [2, 3, 4]]),
            groups=torch.tensor([0, 2, 2]),
            output_size=4,
        )
        print(reduced)

    .. testoutput::

        tensor([[1, 0, 5, 0],
                [2, 0, 7, 0]])

    """
    B, _ = values.shape
    reduced = group_reduce(
        values=values, groups=groups, reduction="sum", output_size=output_size
    )
    assert reduced.shape == (B, output_size)
    assert reduced.dtype == values.dtype

    return reduced


@typecheck
def group_logsumexp(
        *,
        values: Float32Tensor["bootstraps values"],
        groups: Int64Tensor["values"],
        output_size: int,
    ) -> Float32Tensor["bootstraps {output_size}"]:
    """Group-wise, numerically stable log-sum-exp reduction.

    We apply the log-sum-exp trick (https://en.wikipedia.org/wiki/LogSumExp)
    and rely on scatter/gather operations for fast computations on groups
    that may not have the same sizes.

    Cells of the output tensor that do not correspond to a group
    are set to -inf.

    .. testcode::
        import torch
        from survivalgpu.group_reduction import group_logsumexp

        reduced = group_logsumexp(
            values=torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
            groups=torch.tensor([2, 1, 1]),
            output_size=4,
        )
        print(reduced)

    .. testoutput::

        tensor([[  -inf, 3.3133, 1.0000,   -inf],
                [  -inf, 4.3133, 2.0000,   -inf]])

    """
    B, _ = values.shape

    # First, compute the maximum for each group - group_maxima is (B,T):
    group_maxima = group_reduce(
        values=values,
        groups=groups,
        reduction="max",
        output_size=output_size,
    )
    # Then, expand this information as a (B,N) tensor...
    maxima = group_expand(
        values=group_maxima, groups=groups, output_size=output_size,
    )
    # And normalize the values so that they are all <= 0,
    # with at least one term per group equal to 0:
    values = values - maxima  # (B,N)
    # Apply the exponential...
    exps = values.exp()  # (B,N)
    # And the group-wise sum, without any problem of numeric underflow - group_risks is (B,T):
    group_exps = group_reduce(
        values=exps,
        groups=groups,
        reduction="sum",
        output_size=output_size,
    )
    # Finally, apply the logarithm on the sum...
    # and don't forget to re-add the group maxima!
    reduced = group_exps.log() + group_maxima
    assert reduced.shape == (B, output_size)

    return reduced


# ========================================================================================
#               "cumsum" operations over consecutive groups, i.e. segments
# ========================================================================================


@typecheck
def log1mexp(
        x: Float32Tensor["*values"],
    ) -> Float32Tensor["*values"]:
    """Numerically accurate evaluation of log(1 - exp(x)) for x <= 0.

    See https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf for details.

    We rely on numerically stable implementations of
    [x -> log(1+x)] and [x -> exp(x)-1] for x close to 0.

    If -log(2) < x < 0, we use the following identity:
    log(1 - exp(x)) = log(-(exp(x) - 1))

    If x <= -log(2), we use the following identity:
    log(1 - exp(x)) = log1p(-exp(x))

    .. testcode::

        import torch
        from survivalgpu.group_reduction import log1mexp

        # A naive float32 implementation with -0.0000001
        # would return -15.9424 instead of -16.1181.
        x = torch.tensor([0.0, -0.0, -0.0000001, -0.1, -0.5, -1.0, -2.0])
        print(log1mexp(x))

    .. testoutput::

        tensor([    -inf,     -inf, -16.1181,  -2.3522,  -0.9328,  -0.4587,  -0.1454])

    """
    mask = -np.log(2) < x  # x <= 0
    return torch.where(
        mask,
        (-x.expm1()).log(),
        (-x.exp()).log1p(),
    )



@typecheck
def logdiffexp(
    a: Float32Tensor["*values"],
    b: Float32Tensor["*values"],
) -> Float32Tensor["*values"]:
    """Computes the logarithm of the difference of exponentials, i.e. log(exp(a) - exp(b)).

    This is a numerically stable version of log(exp(a) - exp(b)), which is useful
    when a and b are close to each other.

    .. testcode::

        import torch
        from survivalgpu.group_reduction import logdiffexp

        print(
            logdiffexp(
                torch.tensor([1.0, 2.0]),
                torch.tensor([1.0, 1.5]),
            )
        )

    .. testoutput::

        tensor([  -inf, 1.0672])

    """
    # We use the following identity:
    # if a > b,
    # log(e^a - e^b) = log( e^a  * (1 - e^(b-a)))
    #                = a + log(1 - e^(b-a))
    if False: # torch.any(a < b):
        msg = "a must be greater than or equal to b for logdiffexp."
        raise ValueError(msg)

    # We must take care of the case where a == -inf == b,
    # which would lead to a NaN result.
    return torch.where(
        a <= b,
        torch.tensor(float("-inf"), dtype=a.dtype, device=a.device),
        a + log1mexp(b - a),
    )



@typecheck
def is_segment(
    groups: Int64Tensor["values"],
) -> bool:
    """Checks if the groups tensor is a segment, i.e. if it contains consecutive integers.

    .. testcode::

        import torch
        from survivalgpu.group_reduction import is_segment

        print(is_segment(torch.tensor([0, 0, 1, 1, 2])))

    .. testoutput::

        True

    .. testcode::

        print(is_segment(torch.tensor([0, 1, 2, 3, 5])))

    .. testoutput::

        False

    """
    diff = groups[1:] - groups[:-1]
    return bool(torch.all((diff == 0) | (diff == 1)) and groups[0] == 0)


@typecheck
def first_in_segment(
    segments: Int64Tensor["values"],
) -> BoolTensor["values"]:
    """Returns an indicatrix for the first index in each segment.

    .. testcode::

        import torch
        from survivalgpu.group_reduction import first_in_segment

        print(first_in_segment(torch.tensor([0, 0, 0, 1, 1, 2, 3])))

    .. testoutput::

        tensor([ True, False, False,  True, False,  True,  True])

    """
    assert is_segment(segments), "Segments must be consecutive integers starting from 0."
    diff = segments[1:] != segments[:-1]
    return torch.cat(
        (torch.tensor([True], device=segments.device), diff),
        dim=0,
    )


@typecheck
def last_in_segment(
    segments: Int64Tensor["values"],
) -> BoolTensor["values"]:
    """Returns an indicatrix for the last index in each segment.

    .. testcode::

        import torch
        from survivalgpu.group_reduction import last_in_segment

        print(last_in_segment(torch.tensor([0, 0, 0, 1, 1, 2, 3])))

    .. testoutput::

        tensor([False, False,  True, False,  True,  True,  True])

    """
    assert is_segment(segments), "Segments must be consecutive integers starting from 0."
    diff = segments[1:] != segments[:-1]
    return torch.cat(
        (diff, torch.tensor([True], device=segments.device)),
        dim=0,
    )


@typecheck
def rank_in_segment(
    segments: Int64Tensor["values"],
) -> Int64Tensor["values"]:
    """Returns the rank of each index in its segment.

    .. testcode::

        import torch
        from survivalgpu.group_reduction import rank_in_segment

        print(rank_in_segment(torch.tensor([0, 0, 0, 1, 1, 2, 3])))

    .. testoutput::

        tensor([0, 1, 2, 0, 1, 0, 0])

    """
    # We do *not* check that values are consecutive integers,
    # because we use this function on the "efron_indices".
    if segments.numel() == 0:
        return torch.empty_like(segments)

    # Identify where each new segment starts
    is_start = torch.zeros_like(segments, dtype=torch.bool)
    is_start[0] = True
    is_start[1:] = segments[1:] != segments[:-1]

    # Counter that increases each step
    counter = torch.arange(len(segments), device=segments.device)

    # Subtract the counter value at the start of the current segment
    # First, build segment start indices via cumulative sum of is_start
    seg_ids = torch.cumsum(is_start, dim=0) - 1
    seg_start_idx = counter[is_start][seg_ids]

    return counter - seg_start_idx


@typecheck
def keys_to_segments(
    keys: Int64Tensor["dimensions values"],
) -> Int64Tensor["values"]:
    """Converts a 2D tensor of keys into a vector of consecutive group labels.

    .. testcode::

        import torch
        from survivalgpu.group_reduction import keys_to_segments

        print(
            keys_to_segments(
                torch.tensor(
                    [
                        [0, 0, 0, 0, 1, 1, 1],
                        [0, 0, 1, 2, 0, 0, 0],
                    ]
                )
            )
        )

    .. testoutput::

        tensor([0, 0, 1, 2, 3, 3, 3])

    """
    _, segments = torch.unique_consecutive(
        keys, return_inverse=True, dim=1
    )
    assert is_segment(segments)
    return segments


@typecheck
def segment_logcumsumexp(
    *,
    values: Float32Tensor["bootstraps values"],
    segments: Int64Tensor["values"],
) -> Float32Tensor["bootstraps values"]:
    """Computes the cumulative sum of exponentials over segments, i.e. groups of consecutive indices.

    This is a numerically stable version of the cumulative sum of exponentials.

    .. testcode::

        import torch
        from survivalgpu.group_reduction import segment_logcumsumexp

        print(
            segment_logcumsumexp(
                values=torch.tensor([[-float("inf"), 1.0, 2.0, 3.0, 4.0]]),
                segments=torch.tensor([0, 0, 0, 1, 1]),
            )
        )

    .. testoutput::

        tensor([[  -inf, 1.0000, 2.3133, 3.0000, 4.3133]])

    """
    B, V = values.shape
    S = segments[-1] + 1
    assert is_segment(segments), "Segments must be consecutive integers starting from 0."
    full_logcumsumexp = torch.logcumsumexp(values, dim=1)
    assert full_logcumsumexp.shape == (B, V)

    offset_indices = last_in_segment(segments)
    assert offset_indices.shape == (V,)
    assert offset_indices.sum() == S

    offsets = full_logcumsumexp[:, offset_indices]
    assert offsets.shape == (B, S)

    # At this stage, offsets[s] corresponds to the "cumsum"
    # over segment s.
    # We "shift" these offsets to the right by one position,
    # so that offsets[s] corresponds to the "cumsum" over segment s-1
    # (and offsets[0] is -inf, i.e. a cumsum of exp(-inf) = 0).
    offsets = torch.cat(
        (
            -float("inf") * torch.ones_like(offsets[:, :1]),
            offsets[:, :-1],
        ),
        dim=1,
    )
    assert offsets.shape == (B, S)

    offsets_expanded = torch.index_select(offsets, 1, segments)
    assert offsets_expanded.shape == (B, V)

    result = logdiffexp(full_logcumsumexp, offsets_expanded)
    assert result.shape == (B, V)
    return result
