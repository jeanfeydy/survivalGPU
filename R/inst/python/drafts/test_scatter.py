# Test file for a current issue in torch_scatter:
# https://github.com/rusty1s/pytorch_scatter/issues/299

import torch
import torch_scatter


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


# Values:
val = torch.FloatTensor([[0, 1, 2]])
# Groups:
gr_coo = torch.LongTensor([[0, 0, 1]])
gr_csr = torch.LongTensor([[0, 2, 3]])

val.requires_grad = True
B, D = val.shape


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

    elif backend == "coo":
        return torch_scatter.segment_coo(
            values, groups, dim_size=output_size, reduce=reduction
        )
    elif backend == "my_coo":
        if reduction == "sum":
            return SumCOO.apply(values, groups, output_size)
        else:
            return torch_scatter.segment_coo(
                values, groups, dim_size=output_size, reduce=reduction
            )

    elif backend == "csr":
        return torch_scatter.segment_csr(values, groups, reduce=reduction)
    elif backend == "my_csr":
        if reduction == "sum":
            return SumCSR.apply(values, groups)
        else:
            return torch_scatter.segment_csr(values, groups, reduce=reduction)
    else:
        raise ValueError(
            f"Invalid value for the scatter backend ({backend}), "
            "should be one of 'torch', 'pyg', 'coo' or 'csr'."
        )


for backend in ["torch", "pyg", "coo", "my_coo", "csr", "my_csr"]:

    red = group_reduce(
        values=val,
        groups=gr_csr if "csr" in backend else gr_coo,
        reduction="sum",
        output_size=2,
        backend=backend,
    )

    # Compute an arbitrary scalar value out of our reduction:
    v = (red**2).sum(-1) + 0.0 * (val**2).sum(-1)
    # Gradient:
    g = torch.autograd.grad(v.sum(), [val], create_graph=True)[0]
    # Hessian:
    h = torch.zeros(B, D, D).type_as(val)
    for d in range(D):
        h[:, d, :] = torch.autograd.grad(g[:, d].sum(), [val], retain_graph=True)[0]

    print(backend, ":")
    print("Value:", v.detach().numpy())
    print("Grad :", g.detach().numpy())
    print("Hessian:")
    print(h.detach().numpy())
    print("--------------")
