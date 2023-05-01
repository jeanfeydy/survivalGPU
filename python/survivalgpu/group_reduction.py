# Use PyTorch for fast array manipulations (on the GPU):
import torch

# Use torch_scatter (https://github.com/rusty1s/pytorch_scatter)
# from the PyTorch Geometric project for fast heterogeneous summations:
import torch_scatter


class SlicedSummation(torch.autograd.Function):
    """Adds a list of vectors as "suffixes".

    The PyTorch autograd engine does not support in-place operation,
    so we have to use a custom operator to implement in a differentiable way
    the update:

        output[slice_start[i]:] += slices[i]

    that is required for the efficient implementation of the Efron approximation.
    """

    @staticmethod
    def forward(ctx, slice_indices, *slices):
        ctx.save_for_backward(slice_indices)
        full_sum = slices[0].clone()
        for slice_start, current_slice in zip(slice_indices, slices[1:]):
            full_sum[slice_start:] += current_slice
        return full_sum

    @staticmethod
    def backward(ctx, grad_output):
        (slice_indices,) = ctx.saved_tensors
        # No gradient for slice_indices, but backprop the gradient on all the slices:
        return (
            None,
            grad_output,
            *tuple(grad_output[slice_start:] for slice_start in slice_indices),
        )


# Handcrafted fix for a bug in torch_scatter,
# https://github.com/rusty1s/pytorch_scatter/issues/299
class SumCSR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, groups):
        ctx.save_for_backward(groups)
        return torch_scatter.segment_csr(values, groups, reduce="sum")

    @staticmethod
    def backward(ctx, grad_output):
        (groups,) = ctx.saved_tensors
        return GatherCSR.apply(grad_output, groups), None


class GatherCSR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, groups):
        ctx.save_for_backward(groups)
        return torch_scatter.gather_csr(values, groups)

    @staticmethod
    def backward(ctx, grad_output):
        (groups,) = ctx.saved_tensors
        return SumCSR.apply(grad_output, groups), None


class SumCOO(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, groups, dim_size):
        ctx.save_for_backward(groups)
        ctx.dim_size = dim_size
        return torch_scatter.segment_coo(
            values, groups, dim_size=dim_size, reduce="sum"
        )

    @staticmethod
    def backward(ctx, grad_output):
        (groups,) = ctx.saved_tensors
        return GatherCOO.apply(grad_output, groups, ctx.dim_size), None, None


class GatherCOO(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, groups, dim_size):
        ctx.save_for_backward(groups)
        ctx.dim_size = dim_size
        return torch_scatter.gather_coo(values, groups)

    @staticmethod
    def backward(ctx, grad_output):
        (groups,) = ctx.saved_tensors
        return SumCOO.apply(grad_output, groups, ctx.dim_size), None, None


def group_reduce(*, values, groups, reduction, output_size, backend):
    if backend == "torch":
        # Compatibility switch for PyTorch.scatter_reduce:
        if reduction == "max":
            reduction = "amax"
        return torch.scatter_reduce(
            values, 1, groups, reduction, output_size=output_size
        )
    elif backend == "pyg":
        return torch_scatter.scatter(
            values, groups, dim=1, dim_size=output_size, reduce=reduction
        )

    elif False:  # backend == "coo":
        return torch_scatter.segment_coo(
            values, groups, dim_size=output_size, reduce=reduction
        )
    elif backend == "coo":  # backend == "my_coo":
        if reduction == "sum":
            return SumCOO.apply(values, groups, output_size)
        else:
            return torch_scatter.segment_coo(
                values, groups, dim_size=output_size, reduce=reduction
            )

    elif False:  # backend == "csr":
        return torch_scatter.segment_csr(values, groups, reduce=reduction)
    elif backend == "csr":  # backend == "my_csr":
        if reduction == "sum":
            return SumCSR.apply(values, groups)
        else:
            return torch_scatter.segment_csr(values, groups, reduce=reduction)
    else:
        raise ValueError(
            f"Invalid value for the scatter backend ({backend}), "
            "should be one of 'torch', 'pyg', 'coo' or 'csr'."
        )


def group_expand(*, values, groups, output_size, backend):
    if backend in ["torch", "pyg"]:
        return torch.gather(values, 1, groups)
    elif backend == "coo":
        return GatherCOO.apply(values, groups, output_size)
    elif backend == "csr":
        return GatherCSR.apply(values, groups)
    else:
        raise ValueError(
            f"Invalid value for the scatter backend ({backend}), "
            "should be one of 'torch', 'pyg', 'coo' or 'csr'."
        )


def group_logsumexp(*, values, groups, output_size, backend):
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
        backend=backend,
    )
    # Then, expand this information as a (B,N) tensor...
    maxima = group_expand(
        values=group_maxima, groups=groups, output_size=output_size, backend=backend
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
        backend=backend,
    )
    # Finally, apply the logarithm on the sum...
    # and don't forget to re-add the group maxima!
    group_values = group_exps.log() + group_maxima

    return group_values