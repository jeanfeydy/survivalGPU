# ======================================================================================
# ============================== Newton optimizer ======================================
# ======================================================================================
#

# Use PyTorch for fast array manipulations (on the GPU):
import torch

# Wrapper to compute the gradient and Hessian of our objective:
from .autodiff import derivatives_012

from .utils import numpy, int64

"""
# N.B.: using cholesky_inverse guarantees the symmetry of the result:
if False:
    self.imat_ = torch.cholesky_inverse(torch.linalg.cholesky(self.hessian_))
else:
    self.imat_ = torch.inverse(self.hessian_)
    self.imat_ = (self.imat_ + self.imat_.transpose(-1, -2)) / 2

self.std_ = self.imat_.diagonal(dim1=-2, dim2=-1).sqrt()
"""


def newton(*, loss, start, maxiter, eps=1e-9, verbosity=0):
    """Estimates optimal parameters by minimizing a convex objective function.

    Args:
        loss (function): the convex objective function.
            Takes as input beta, a (B,D) array of (batched) input
            parameters [beta[0], ..., beta[B-1]] and returns a (B,) vector of
            scalar values [f(beta[0]), ..., f(beta[B-1])].
        start ((B,D) tensor): initial starting values.
    """

    # Automatic differentiation wrapper to get the derivatives of order 1 and 2:
    loss_grad_hessian = derivatives_012(loss)

    B, D = start.shape

    # Current "candidates" at a given iteration:
    candidates = start.clone()  # (B,D)
    # best_params are the best observed candidates so far
    # (associated to the smallest values of the objective):
    best_params = candidates.clone()  # (B,D)
    # We're going to use PyTorch autodiff engine to compute derivatives
    # of order 1 and 2, so we need to ask PyTorch to keep in mind
    # that "candidates" is a differentiable variable:
    candidates.requires_grad = True

    # Current estimates for the best values - we keep B values in parallel:
    best_values = torch.ones(B, device=candidates.device) * float("inf")  # (B,)
    # Step size "dampener" - once again, B values in parallel:
    rejections = torch.zeros(B, device=candidates.device)  # (B,)
    # Break - (B,) vector of bool:
    break_loop = best_values == rejections  # = (False, False, ..., False)

    # Monitoring information:
    # Actual number of iterations used:
    iters = torch.zeros(B, dtype=int64, device=candidates.device)
    # Are we running into infinite or NaN values?
    notfinites = torch.zeros(B, dtype=int64, device=candidates.device)

    if maxiter < 1:
        raise ValueError(
            f"The Newton solver expects at least 1 iteration but received {maxiter}."
        )

    for it in range(maxiter):
        # Compute the value of the convex objective, its gradient and its Hessian:
        # (We perform this step in parallel over the B bootstrap samples.)
        values, grads, hessians = loss_grad_hessian(candidates)
        # values is (B,)
        # grads is (B,D)
        # hessians is (B,D,D)
        # Note that since our loss function is convex,
        # the B hessian matrices of shape (D,D) should be positive definite.

        # Newton step = (H \ grad). (B,D,D) @ (B,D) = (B,D)
        # N.B.: Currently, we encounter a strange CUDA bug with linsolve.
        #       A simple workaround is to come back to the CPU, just for this operation.
        # TODO: remove this "duct tape" fix as soon as possible.
        steps = torch.linalg.solve(hessians.cpu(), grads.cpu()).to(grads.device)

        # The R survival package returns the score test statistic at iteration 0,
        # so we do the same:
        if it == 0:
            score_test = (steps * grads).sum(-1)  # (B,), should be >= 0

        # Did the values of the loss function decrease?
        accept = values < best_values  # (B,) boolean vector

        if verbosity > 0:
            print(f"Iteration {it+1:3d} -- {best_values[0].detach().item():.6e}")
        if verbosity > 1:
            print(f"Best parameters: {numpy(best_params)}")
            print(f"Best value:      {numpy(best_values)}")
            print(f"Candidate: {numpy(candidates)}")
            print(f"Value:     {numpy(values)}")
            print(f"Update this iter? {numpy(accept)}")
            print("")
        if verbosity > 2:
            print("Gradient:", numpy(grads))
            print("Hessian:", numpy(hessians))
            print("Step:", numpy(steps))
            print("")

        # Update the "best values" and "best params" seen so far:
        best_values[accept] = values[accept]
        best_params[accept] = candidates[accept]

        # If we have accepted a new best value, we obtain our next candidate
        # by making a Hessian-adjusted gradient descent step of size 1:
        candidates.data[accept] = best_params[accept] - steps[accept]
        # And we reset the rejections counter:
        rejections[accept] = 0

        # Otherwise, our next candidate will be closer to the "best_params".
        # To do this, we compute a barycentric interpolation of the
        # current candidate and the best_params, with a weight
        # that becomes increasingly large if the steps in this direction
        # get rejected several times:
        rejections[~accept] += 1
        # And try a new candidate:
        closer_steps = (candidates + rejections.view(B, 1) * best_params) / (
            1 + rejections.view(B, 1)
        )
        candidates.data[~accept] = closer_steps[~accept]

    # Recompute the local descriptors at the optimum:
    values, grads, hessians = loss_grad_hessian(best_params)

    # N.B.: detach() removes the autograd history of the variables.
    # It is critical to prevent memory leaks, and allow us to scale up
    # to large datasets and bootstrap copies.
    return {
        "fun": values.detach(),
        "x": best_params.detach(),
        "grad": grads.detach(),
        "hessian": hessians.detach(),
        "score test init": score_test.detach(),
    }
