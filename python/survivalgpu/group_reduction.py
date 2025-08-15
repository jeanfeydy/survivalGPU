# Use PyTorch for fast array manipulations (on the GPU):
import torch


def make_2d(g):
    if len(g.shape) == 1:
        return g.view(1, -1)
    elif len(g.shape) == 2:
        return g
    else:
        msg = "Invalid shape for groups"
        raise ValueError(msg)



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
    def forward(ctx, values, groups, dim_size):
        ctx.save_for_backward(groups)
        ctx.dim_size = dim_size
        z = torch.zeros(
            values.shape[0], dim_size, dtype=values.dtype, device=values.device
        )
        i = make_2d(groups)
        return z.scatter_reduce_(
            dim=1,
            index=i,
            src=values,
            reduce="sum",
            include_self=False,
        )

    @staticmethod
    def backward(ctx, grad_output):
        (groups,) = ctx.saved_tensors
        # TODO: what about dim_size?
        return torch.index_select(grad_output, 1, groups), None, None


def group_reduce(*, values, groups, reduction, output_size):
    # Compatibility switch for PyTorch.scatter_reduce:
    if reduction == "max":
        reduction = "amax"

    assert len(values.shape) == 2
    if reduction == "sum":
        return SumTorch.apply(values, groups, output_size)
    else:
        return torch.zeros(
            values.shape[0], output_size, dtype=values.dtype, device=values.device
        ).scatter_reduce_(
            dim=1,
            index=make_2d(groups),
            src=values,
            reduce=reduction,
            include_self=False,
        )


def group_expand(*, values, groups, output_size):  # noqa: ARG001
    # return torch.gather(values, 1, groups)
    return torch.index_select(values, 1, groups)


def group_logsumexp(*, values, groups, output_size):
    """Group-wise, numerically stable log-sum-exp reduction.

    We apply the log-sum-exp trick (https://en.wikipedia.org/wiki/LogSumExp)
    and rely on scatter/gather operations for fast computations on groups
    that may not have the same sizes.
    """
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
    return group_exps.log() + group_maxima
