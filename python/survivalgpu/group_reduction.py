# Use PyTorch for fast array manipulations (on the GPU):
import torch

from .typecheck import (
    BoolTensor,
    FloatTensor,
    Int64Tensor,
    Literal,
    typecheck,
)

LOG0 = float("-inf")  # replace with float(-1e20) for debugging
MINFLOAT = torch.finfo(torch.float).min

def clip_inf(x):
    """Workaround for a bug in PyTorch with logsumexp gradients.

    https://github.com/pytorch/pytorch/issues/31829
    https://github.com/pytorch/pytorch/issues/49724
    """
    return torch.where(
        x == float("-inf"),
        torch.full_like(x, MINFLOAT),
        x,
    )

def clip_zero(x):
    """Workaround for a bug in PyTorch with log gradients."""
    return torch.where(
        x <= 0,
        torch.full_like(x, torch.finfo(torch.float).smallest_normal),
        x,
    )


# ========================================================================================
#                       "sum" operations over arbitrary groups
# ========================================================================================

@typecheck
def make_2d(
        *,
        groups: Int64Tensor["values"] | Int64Tensor["bootstraps values"],
        values: Int64Tensor["bootstraps values"] | FloatTensor["bootstraps values"],
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
            values: Int64Tensor["bootstraps values"] | FloatTensor["bootstraps values"],
            groups: Int64Tensor["values"] | Int64Tensor["bootstraps values"],
            output_size: int,
        ) -> Int64Tensor["bootstraps {output_size}"] | FloatTensor["bootstraps {output_size}"]:
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
            grad_output: Int64Tensor["bootstraps output_size"] | FloatTensor["bootstraps output_size"],
        ):
        """Fast but non-deterministic backward pass for the group-wise sum reduction."""

        (groups,) = ctx.saved_tensors
        assert len(groups.shape) == 1, "Backward pass only supports 1D groups."

        # TODO: what about output_size?
        return torch.index_select(grad_output, 1, groups), None, None


@typecheck
def group_reduce(
        *,
        values: Int64Tensor["bootstraps values"] | FloatTensor["bootstraps values"],
        groups: Int64Tensor["values"] | Int64Tensor["bootstraps values"],
        reduction: Literal["max", "sum", "sum_forward_pass"],
        output_size: int,
    ) -> Int64Tensor["bootstraps {output_size}"] | FloatTensor["bootstraps {output_size}"]:
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
        values: Int64Tensor["bootstraps values"] | FloatTensor["bootstraps values"],
        groups: Int64Tensor["values"] | Int64Tensor["bootstraps values"],
        output_size: int,
    ) -> Int64Tensor["bootstraps {output_size}"] | FloatTensor["bootstraps {output_size}"]:
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



# ========================================================================================
#               "cumsum" operations over consecutive groups, i.e. segments
# ========================================================================================



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
def segment_cumsum(
    *,
    values: FloatTensor["bootstraps values"],
    segments: Int64Tensor["values"],
) -> FloatTensor["bootstraps values"]:
    """Computes the cumulative sum over segments, i.e. groups of consecutive indices.

    .. testcode::

        import torch
        from survivalgpu.group_reduction import segment_cumsum

        print(
            segment_cumsum(
                values=torch.tensor([[0.0, 1.0, 2.0, 1.0, 4.0]]),
                segments=torch.tensor([0, 0, 0, 1, 1]),
            )
        )

    .. testoutput::

        tensor([[0., 1., 3., 1., 5.]])

    """
    B, V = values.shape
    S = segments[-1] + 1
    assert is_segment(segments), "Segments must be consecutive integers starting from 0."

    full_cumsum = torch.cumsum(values, dim=1)
    assert full_cumsum.shape == (B, V)

    offset_indices = last_in_segment(segments)
    assert offset_indices.shape == (V,)
    assert offset_indices.sum() == S

    offsets = full_cumsum[:, offset_indices]
    assert offsets.shape == (B, S)

    # At this stage, offsets[s] corresponds to the "cumsum"
    # over segment s.
    # We "shift" these offsets to the right by one position,
    # so that offsets[s] corresponds to the "cumsum" over segment s-1
    # (and offsets[0] is 0).
    offsets = torch.cat(
        (
            torch.zeros_like(offsets[:, :1]),
            offsets[:, :-1],
        ),
        dim=1,
    )
    assert offsets.shape == (B, S)

    offsets_expanded = torch.index_select(offsets, 1, segments)
    assert offsets_expanded.shape == (B, V)

    result = full_cumsum - offsets_expanded
    assert result.shape == (B, V)
    return result
