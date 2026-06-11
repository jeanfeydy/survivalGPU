# ======================================================================================
# ============================== Newton optimizer ======================================
# ======================================================================================
#

# Use PyTorch for fast array manipulations (on the GPU):
import torch

# Wrapper to compute the gradient and Hessian of our objective:
from .autodiff import derivatives_012
from .typecheck import FloatTensor, typecheck
from .utils import numpy


class NewtonResult:
    @typecheck
    def __init__(
        self,
        *,
        fun: FloatTensor["batch"],
        fun_init: FloatTensor["batch"],
        x: FloatTensor["batch dim"],
        jac: FloatTensor["batch dim"],
        hess: FloatTensor["batch dim dim"],
        score_test_init: FloatTensor["batch"],
        iterations,
    ):
        """Stores the result of a Newton optimization, with the inverse Hessian and standard errors.

        Args:
            fun ((B,) tensor): value of the objective function at the optimum,
                for each of the B batches.
            fun_init ((B,) tensor): value of the objective function at the
                starting point.
            x ((B,D) tensor): optimal parameters.
            jac ((B,D) tensor): gradient of the objective function at the optimum.
            hess ((B,D,D) tensor): Hessian of the objective function at the optimum.
            score_test_init ((B,) tensor): score test statistic, computed at the
                starting point.
            iterations (int): number of iterations that were performed.

        In addition to storing these arguments as attributes, this computes:
            imat ((B,D,D) tensor): inverse of the Hessian, symmetrized -
                an estimate of the variance-covariance matrix of the parameters.
            std ((B,D) tensor): standard errors of the parameters, i.e. the
                square root of the diagonal of imat.
        """
        self.fun = fun
        self.fun_init = fun_init
        self.x = x
        self.jac = jac
        self.hess = hess
        self.score_test_init = score_test_init
        self.iterations = iterations

        if False:
            # N.B.: using cholesky_inverse guarantees the symmetry of the result:
            self.imat = torch.cholesky_inverse(torch.linalg.cholesky(self.hessian))
        else:
            self.imat = torch.inverse(self.hess.cpu()).to(self.hess.device)
            self.imat = (self.imat + self.imat.transpose(-1, -2)) / 2

        self.std = self.imat.diagonal(dim1=-2, dim2=-1).sqrt()


def newton(*, loss, start, maxiter, eps=1e-9, verbosity=0):  # noqa: ARG001
    """Estimates optimal parameters by minimizing a convex objective function.

    Args:
        loss (function): the convex objective function.
            Takes as input beta, a (B,D) array of (batched) input
            parameters [beta[0], ..., beta[B-1]] and returns a (B,) vector of
            scalar values [f(beta[0]), ..., f(beta[B-1])].
        start ((B,D) tensor): initial starting values.
        maxiter (int): maximum number of Newton iterations. If 0, no Newton
            step is performed: only the score test statistic at `start` is computed.
        eps (float, optional): unused. Defaults to 1e-9.
        verbosity (int, optional): level of detail of the logs printed at each
            iteration. Defaults to 0 (no logs).

    Returns:
        NewtonResult: the optimization result, with fun, fun_init, x, jac, hess,
            score_test_init, iterations, imat and std attributes.
    """

    # Automatic differentiation wrapper to get the derivatives of order 1 and 2:
    loss_grad_hessian = derivatives_012(loss)

    B, D = start.shape

    dtype = start.dtype

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
    best_values = best_values.to(dtype)
    # Step size "dampener" - once again, B values in parallel:
    rejections = torch.zeros(B, device=candidates.device)  # (B,)
    # Break - (B,) vector of bool:
    # break_loop = best_values == rejections  # = (False, False, ..., False)

    # Monitoring information:
    # Actual number of iterations used:
    # iters = torch.zeros(B, dtype=int64, device=candidates.device)
    # Are we running into infinite or NaN values?
    # notfinites = torch.zeros(B, dtype=int64, device=candidates.device)

    if maxiter < 0:
        msg = f"The Newton solver expects at least 0 iteration but received {maxiter}."
        raise ValueError(msg)

    for it in range(maxiter + 1):
        # Compute the value of the convex objective, its gradient and its Hessian:
        # (We perform this step in parallel over the B bootstrap samples.)

        # Replace with (float32 mode only)
        if dtype == torch.float32:
            values, grads, hessians = loss_grad_hessian(candidates.to(torch.float64))
            values = values.to(torch.float32)
            # grads and hessians stay float64 → existing cast at lines 117-118 becomes a no-op
        else:
            values, grads, hessians = loss_grad_hessian(candidates)

        # values is (B,)
        # grads is (B,D)
        # hessians is (B,D,D)
        # Note that since our loss function is convex,
        # the B hessian matrices of shape (D,D) should be positive definite.

        # Newton step = (H \ grad). (B,D,D) @ (B,D) = (B,D)
        # N.B.: Currently, we encounter a strange CUDA bug with linsolve.
        #       A simple workaround is to come back to the CPU, just for this operation.
        # TODO: remove this "duct tape" fix.
        #
        # N.B. torch.linalg.solve(hessians.cpu(), grads.cpu()) is a very sensitive
        #      operation and doing in float32 leads to error that can lead to a final
        #      error that are close to 1%. This step is done on the CPU so float64
        #      are always available. This sensitive step is thus done in float64
        #      it is important to put it on the cpu before putting it in f64



        grads_cpu = grads.cpu()
        hessians_cpu = hessians.cpu()


        steps = torch.linalg.solve(hessians_cpu, grads_cpu)

        # we then send the steps in f32 the nto the device

        if dtype == torch.float32:
            steps = steps.to(torch.float32).to(grads.device)
        else:
            steps = steps.to(grads.device)




        # The R survival package returns the score test statistic at iteration 0,
        # so we do the same:
        if it == 0:
            score_test = (steps * grads).sum(-1)  # (B,), should be >= 0

        if maxiter == 0:
            # We only want to compute the score test statistic at iteration 0:
            break

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
            print("Dtype of steps:", steps.dtype)

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
    if maxiter > 0:
        values, grads, hessians = loss_grad_hessian(best_params)

    # N.B.: detach() removes the autograd history of the variables.
    # It is critical to prevent memory leaks, and allow us to scale up
    # to large datasets and bootstrap copies.
    return NewtonResult(
        fun=values.detach(),
        fun_init=loss(start).detach(),
        x=best_params.detach(),
        jac=grads.detach(),
        hess=hessians.detach(),
        score_test_init=score_test.detach(),
        iterations=it,
    )
