# ======================================================================================
# ========================= Automatic differentiation ==================================
# ======================================================================================
#
# This repository implements a Newton optimizer that relies on the derivatives
# of order 1 (gradient vector) and 2 (Hessian matrix) of a scalar, convex objective
# to minimize.
# We rely extensively on the PyTorch automatic differentiation engine to compute these
# quantities. The function "derivatives_012(...)" below implements all the necessary
# "magic", so that our code can focus entirely on the definition of the objective "f".

import torch


def derivatives_012(f):
    """Appends a gradient and Hessian to a batch-wise loss function.

    Args:
        f (function): takes as input beta, a (B,D) array of (batched) input
            parameters [beta[0], ..., beta[B-1]] and returns a (B,) vector of
            scalar values [f(beta[0]), ..., f(beta[B-1])].

    Returns:
        f_grad_hessian (function): takes as input beta, a (B,D) array of
            (batched) input parameters and returns:
            - a (B,) vector of scalar values [f(beta[i])].
            - a (B,D) array that represents the gradient of loss with
              respect to the B independent values of beta,
              i.e. [grad_f(beta[0]), ..., grad_f(beta[B-1])].
            - a (B,D,D) tensor that represents the B Hessians
              (symmetric (D,D) matrices) of f with respect to the independent
              values beta[0], ..., beta[B-1].
    """

    def f_grad_hessian(beta):
        B, D = beta.shape
        f_beta = f(beta)
        assert f_beta.shape == (B,)

        (grad_beta,) = torch.autograd.grad(f_beta.sum(), [beta], create_graph=True)
        assert grad_beta.shape == (B, D)

        Hessian_beta = torch.zeros(B, D, D).type_as(beta)
        for d in range(D):
            Hessian_beta[:, d, :] = torch.autograd.grad(
                grad_beta[:, d].sum(), [beta], retain_graph=True
            )[0]
        # torch.autograd.functional.hessian(loss, beta)

        return f_beta, grad_beta, Hessian_beta

    return f_grad_hessian
