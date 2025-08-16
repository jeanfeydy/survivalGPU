# Use PyTorch for fast array manipulations (on the GPU):
import torch

from .typecheck import Float32Tensor, Int64Tensor, Literal, typecheck


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


# Trying to work around a huge bottleneck in the backward pass
# because of the use of a deterministic algorithm in
# the indexing_backward_kernel
# https://github.com/pytorch/pytorch/issues/41162
# https://github.com/dmlc/dgl/issues/3729
#
# Instead of indexing, we should use the (much faster) operation
# index_select.
class SumTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, groups, output_size):
        ctx.save_for_backward(groups)
        ctx.output_size = output_size

        B, _ = values.shape

        reduced = torch.zeros(
            values.shape[0], output_size, dtype=values.dtype, device=values.device
        ).scatter_reduce_(
            dim=1,
            index=make_2d(groups=groups, values=values),
            src=values,
            reduce="sum",
            include_self=False,
        )
        assert reduced.shape == (B, output_size)
        assert reduced.dtype == values.dtype
        return reduced

    @staticmethod
    def backward(ctx, grad_output):
        (groups,) = ctx.saved_tensors
        assert len(groups.shape) == 1, "Backward pass only supports 1D groups."

        # TODO: what about output_size?
        return torch.index_select(grad_output, 1, groups), None, None


@typecheck
def group_reduce(
        *,
        values: Int64Tensor["bootstraps values"] | Float32Tensor["bootstraps values"],
        groups: Int64Tensor["values"] | Int64Tensor["bootstraps values"],
        reduction: Literal["sum", "max"],
        output_size: int,
    ):

    B, _ = values.shape

    # Compatibility switch for PyTorch.scatter_reduce:
    if reduction == "max":
        reduction = "amax"

    if reduction == "sum":
        reduced = SumTorch.apply(values, groups, output_size)
    else:
        reduced = torch.zeros(
            values.shape[0], output_size, dtype=values.dtype, device=values.device
        ).scatter_reduce_(
            dim=1,
            index=make_2d(groups=groups, values=values),
            src=values,
            reduce=reduction,
            include_self=False,
        )

    assert reduced.shape == (B, output_size)
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
    ) -> Int64Tensor["bootstraps output_size"] | Float32Tensor["bootstraps output_size"]:
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
    ) -> Float32Tensor["bootstraps output_size"]:
    """Group-wise, numerically stable log-sum-exp reduction.

    We apply the log-sum-exp trick (https://en.wikipedia.org/wiki/LogSumExp)
    and rely on scatter/gather operations for fast computations on groups
    that may not have the same sizes.

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
