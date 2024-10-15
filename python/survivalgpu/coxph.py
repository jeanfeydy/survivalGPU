# Use NumPy for basic array manipulation:
import numpy as np

# Use PyTorch for fast array manipulations (on the GPU):
import torch

# The convex CoxPH objective:
from .coxph_likelihood import coxph_objective_torch, group_reduce
from .coxph_likelihood_keops import coxph_objective_keops

# We currently support 5 different backends,
# and will eventually settle on the fastest one(s):
coxph_objectives = {
    "torch": coxph_objective_torch,
    "pyg": coxph_objective_torch,
    "coo": coxph_objective_torch,
    "csr": coxph_objective_torch,
    "keops": coxph_objective_keops,
}

# Convex optimizer for the CoxPH objective:
from .optimizers import newton

from .utils import numpy
from .utils import use_cuda, device, float32, int32, int64


# Main function:
def coxph_torch(
    *,
    x,
    times,
    deaths,
    bootstrap=1,
    batchsize=0,
    ties="efron",
    backend="torch",
    maxiter=20,
    init=None,
    eps=1e-9,
    alpha=0,
    verbosity=0,
    doscale=False,
):
    """Implements the Cox Proportional Hazards model.

    Args:
        x ((N,D) or (C,N,D) float32 tensor): the input features.
            If C > 1, we solve C problems in parallel with different feature vectors.
        times ((N,) int32 tensor): observation times.
        deaths ((N,) int32 tensor): 1 if the subjects dies at time t, 0 if it survives.
        bootstrap (int, optional): Number of repeats for the bootstrap cross-validation.
            Defaults to 1.
        batchsize (int, optional): Number of bootstrap copies that should be handled at a time.
            Defaults to 0, which means that we handle all copies at once.
            If you run into out of memory errors, please consider using batchsize=100, 10 or 1.
        ties (str, optional): method to handle ties - either "efron" or "breslow".
            Defaults to "efron".
        backend (str, optional): method to compute the log-sum-exp reduction.
            Use either "torch" for a torch.scatter-based implementation,
            or "keops" for a LazyTensor-based implementation.
            Defaults to "torch".

    Raises:
        ValueError: If the batchsize is non-zero and does not divide the requested
            number of bootstraps.

    Returns:
        dict of torch Tensors: with keys
            "coef": (B,D) or (B,C,D) tensor of optimal weights for each bootstrap.
            "loglik init": (B,) or (B,C) tensor of values of the log-likelihood at iteration 0.
            "sctest init": (B,) or (B,C) tensor of values of the score test at iteration 0.
            "loglik": (B,) or (B,C) tensor of values of the log-likelihood at the optimum.
            "u": (B,D) or (B,C,D) tensor of gradients of the log-likelihood at the optimum (should be close to zero).
            "hessian": (B,D,D) or (B,C,D,D) tensor that represents, for each bootstrap, the Hessian of the neg-log-likelihood at the optimum - this should be a symmetric, positive (D,D) matrix.
            "imat": (B,D,D) or (B,C,D,D) tensor that represents, for each bootstrap, the inverse of the Hessian above. This corresponds to an estimated variance matrix for the optimal coefficients.
    """

    # Step 1: Pre-processing ===================================================

    # 1.a: Basic re-ordering and counting --------------------------------------
    # Select the CPU or GPU device:
    device = x.device

    # Extract the main integer dimensions:
    if len(x.shape) == 2:
        single_C = True
        x = x[None, :, :]
    else:
        single_C = False

    C, N, D = x.shape  # Number of samples, number of input features

    # Re-order the input variables to make sure that the times are increasing:
    order = (2 * times + deaths).argsort()  # Lexicographic order on (time, death)
    times = times[order]  #   (N,), e.g. [2, 2, 2, 2, 2, 5, 5, 6, 6, 6]
    deaths = deaths[order]  # (N,), e.g. [0, 0, 0, 1, 1, 0, 1, 0, 0, 1]
    x = x[:, order]  # (C,N,D), e.g. [[2, 4], ..., [1, 0]]

    means = x.mean(dim=1)  # (C,D)
    if doscale:
        # For the sake of numerical stability, we may normalize the covariates
        # This should have zero impact on the CoxPH objective:
        x = x - means.view(C, 1, D)
        # Use the L1 norms for scale as in the R survival package:
        scales = x.abs().sum(dim=1)  # (C,D)
        scales[scales == 0] = 1  # Rare case of a constant covariate
        scales = 1 / scales
        x = x * scales.view(C, 1, D)

    # Count the number of death times:
    unique_times, cluster_indices = torch.unique_consecutive(times, return_inverse=True)
    # unique_times is (T,), e.g.    [2, 5, 6]
    # cluster_indices is (N,), e.g. [0, 0, 0, 0, 0, 1, 1, 2, 2, 2]
    T = len(unique_times)  # in our example, T = 3

    # For each event time, count the number of (possibly tied) deaths:
    tied_deaths = torch.bincount(cluster_indices[deaths == 1], minlength=T)
    # tied_deaths is (T,), e.g. [2, 1, 1]

    # Filter out the times where no one dies - this may be a common occurence
    # in some datasets where the raw "start-stop columns" sample every single
    # month or year:
    # N.B.: This optimization should have zero impact on the result.
    #       We leave it enabled by default - but feel free to comment out
    #       this block to perform some tests.
    if True:
        mask = tied_deaths[cluster_indices] > 0  # (N,) vector of booleans

        # Just keep the lines that correspond to times where someone dies:
        times = times[mask]
        deaths = deaths[mask]
        x = x[:, mask]

        # Don't forget to update all the variables that may have been impacted:
        C, N, D = x.shape

        unique_times, cluster_indices = torch.unique_consecutive(
            times, return_inverse=True
        )
        T = len(unique_times)
        tied_deaths = torch.bincount(cluster_indices[deaths == 1], minlength=T)
        assert (tied_deaths > 0).all()

        # Please note that the means and scales are not affected by this filtering
        # pass, to stick to the conventions of the R survival package.

    # 1.b: Bootstrap-related weighting -----------------------------------------
    if batchsize == 0:
        # No batchsize -> we handle the bootstrap*C problems together:
        batchsize = bootstrap

    if bootstrap % batchsize != 0:
        raise ValueError(
            f"Please use a number of bootstraps (found {bootstrap}) "
            f"which is a multiple of the batchsize (found {batchsize})."
        )

    B = batchsize  # Number of bootstraps that we handle at a time
    results = []  # We store one output per batch
    
    # Vector of inital values of the Newton iteration. Zero for all
    # variables by default.
    if init is None:
        init = torch.zeros(B * C, D, dtype=float32, device=device)
    else:
        init_tensor = torch.tensor(init, dtype=float32, device=device)
        init = init_tensor.repeat(B * C, 1)

    for batch_it in range(bootstrap // batchsize):
        # We simulate bootstrapping using an integer array
        # of "weights" numbers of shape (B, N) where B is the number of bootstraps.
        # The original sample corresponds to weights = [1, ..., 1],
        # while other values for the vector always sum up to
        # the number of patients.
        bootstrap_indices = torch.randint(N, (B, N), dtype=int64, device=device)
        # Our first line corresponds to the original sample,
        # i.e. there is no re-sampling if bootstrap == 1:
        if batch_it == 0:
            bootstrap_indices[0, :] = torch.arange(N, dtype=int64, device=device)
        # bootstrap_indices is (B,N), e.g.:
        # [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        #  [3, 3, 1, 2, 8, 6, 0, 0, 8, 9]]

        # Compute the numbers of occurrences of each sample index in the
        # rows of bootstrap_indices:
        sample_weights = torch.ones(B, N, dtype=float32, device=device)
        weights = group_reduce(
            values=sample_weights,
            groups=bootstrap_indices,
            reduction="sum",
            output_size=N,
            backend="torch",
        )
        weights = weights.to(device=device, dtype=float32)
        # Equivalent to:
        # weights = torch.stack([torch.bincount(b_ind, minlength=N) for b_ind in bootstrap_indices])
        #
        # Weights is (B,N) with lines that sum up to N, e.g.
        # [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #  [2, 1, 1, 2, 0, 0, 1, 0, 2, 1]]

        # Pre-compute the logarithms of the weights:
        # TODO: We are currently adding a small value to prevent NaN.
        #       This is not very clean...
        log_weights = (weights + 1e-8).log()  # (B,N), e.g.
        # [[ 0, 0, 0,  0,   0,   0, 0,    0,  0, 0],
        #  [.7, 0, 0, .7,-inf,-inf, 0, -inf, .7, 0]]

        # 1.c: Aggregate the "total weights for dead samples" at each time point ---
        # These are required as multiplicative factors by the Efron and Breslow rules

        # Compute the total weight of dead samples for every event time:
        dead_weights = weights[:, deaths == 1]
        # dead_weights is (B, Ndeads), e.g.
        # [[1, 1, 1, 1],
        #  [2, 0, 1, 1]]
        dead_cluster_indices = cluster_indices[deaths == 1].repeat(B, 1)
        # dead_cluster_indices is (B, Ndeads), e.g.
        # [[0, 0, 1, 2],
        #  [0, 0, 1, 2]]
        tied_dead_weights = group_reduce(
            values=dead_weights,
            groups=dead_cluster_indices.long(),
            reduction="sum",
            output_size=T,
            backend="torch",
        )
        # Equivalent to:
        # tied_dead_weights = torch.bincount(cluster_indices[deaths == 1],
        #                     weights=weights.view(-1)[deaths == 1],
        #                     minlength=T)
        #
        # tied_dead_weights is (B,T), e.g.
        # [[2, 1, 1],
        #  [2, 1, 1]]

        if verbosity > 0:
            print("Pre-processing:")
            print(f"Working with {B:,} bootstrap, {C:,} channels, {T:,} death times,")
            print(f"{N:,} rows (= observations) and {D:,} columns (= features).")
            print("")

        if verbosity > 1:
            print("Cluster indices:")
            print(numpy(cluster_indices))
            # print("Number of tied deaths:")
            # print(numpy(tied_deaths))
            print("Tied dead weights:")
            print(numpy(tied_dead_weights))
            print("")

        def linear_risk_scores(params):
            """Standard function to compute risks in the CoxPH model: dot(beta, x[i]).

            Args:
                params ((B*C,D) float32 tensor): batch of B model parameters "beta"
                    for C different problems, identified with vectors of size D.

            Returns:
                (B*C,N) float32 tensor: predicted scores for the B*C model parameters
                    and N subjects (= feature vectors).
            """

            # If C was equal to 1, the lines below would be equivalent to:
            # return params @ x.view(N,D).T  # (B,D) @ (D,N) = (B,N)
            p = params.view(B, C, D)  # (B*C,D) -> (B,C,D)
            s = torch.einsum("bcd,cnd->bcn", p, x)  # (B,C,D) @ (C,N,D) = (B,C,N)
            return s.view(B * C, N)

        # Make C copies of all the relevant arrays, i.e.
        # (B,N) -> (B*C,N) with blocks of C constant lines:
        def C_copies(arr):
            b, n = arr.shape
            assert B == b
            return arr.view(b, 1, n).tile((1, C, 1)).view(b * C, n)

        # Step 2: select the proper backend and perform the last pre-computations ==
        #backend = "torch"
        print(backend)
        objective = coxph_objectives[backend](
            N=N,
            B=B * C,
            T=T,
            risk_scores=linear_risk_scores,
            deaths=deaths,
            weights=C_copies(weights),
            log_weights=C_copies(log_weights),
            ties=ties,
            tied_deaths=tied_deaths,
            tied_dead_weights=C_copies(tied_dead_weights),
            cluster_indices=cluster_indices,
            backend=backend,
        )


        def loss(b):
            obj = objective(b)
            reg = alpha * (b**2).sum(dim=1)
            return obj + reg

        # Step 3: Newton iterations ================================================
        # We estimate the optimal vector of parameters "beta"
        # by minimizing a convex objective function.
        
        res = newton(
            loss=loss,
            start=init,
            maxiter=maxiter,
            eps=eps,
            verbosity=verbosity,
        )

        # Post-processing: re-cast our B*C problems as
        # a sequence of B problems (bootstraps on one set of features)
        # or a (B, C) array of problems (bootstraps on C sets of features).
        def C_unwrap(arr):
            if single_C:
                return arr
            else:
                sh = arr.shape
                return arr.view(sh[0] // C, C, *sh[1:])

        # We rename the output to fit with the conventions of R survival:

        # (B*C,) -> (B,) or (B,C):
        res["sctest init"] = C_unwrap(res.pop("score test init"))
        # loglik[0] in the R survival package:
        res["loglik init"] = C_unwrap(-loss(init))
        res["loglik"] = C_unwrap(-res.pop("fun"))  # loglik[1] in the R survival package

        # (B*C,D) -> (B,D) or (B,C,D):
        res["coef"] = C_unwrap(res.pop("x"))
        res["u"] = C_unwrap(-res.pop("grad"))

        # N.B.: using cholesky_inverse guarantees the symmetry of the result:
        if False:
            res["imat"] = torch.cholesky_inverse(torch.linalg.cholesky(res["hessian"]))
        else:
            res["imat"] = torch.inverse(res["hessian"])
            res["imat"] = (res["imat"] + res["imat"].transpose(-1, -2)) / 2

        # (B*C,D,D) -> (B,D,D) or (B,C,D,D)
        res["hessian"] = C_unwrap(res["hessian"])
        res["imat"] = C_unwrap(res["imat"])

        # Step 4: Finish ===========================================================

        # If the covariates have been normalized for the sake of stability,
        # we shouldn't forget to "de-normalize" the results:

        if doscale:
            scales = scales.view(D) if single_C else scales.view(C, D)

            # (B,D) *= (D,) or (B,C,D) *= (C,D):
            res["coef"] = res["coef"] * scales
            # (B,D) /= (D,) or (B,C,D) /= (C,D):
            res["u"] = res["u"] / scales

            if single_C:  # (B,D,D) *= (D,D)
                res["imat"] = res["imat"] * (scales.view(1, D) * scales.view(D, 1))
                res["hessian"] = res["hessian"] / (
                    scales.view(1, D) * scales.view(D, 1)
                )
            else:  # (B,C,D,D) *= (C,D,D)
                res["imat"] = res["imat"] * (
                    scales.view(C, 1, D) * scales.view(C, D, 1)
                )
                res["hessian"] = res["hessian"] / (
                    scales.view(C, 1, D) * scales.view(C, D, 1)
                )

        # We also return the means of our covariates:
        res["means"] = means.view(D) if single_C else means.view(C, D)

        results.append(res)

    # Step 5: turn the list of dicts into a dicts of concatenated results ==========
    output = {
        key: torch.cat(tuple(res[key] for res in results)) for key in results[0].keys()
    }

    return output


# Python >= 3.7:
from contextlib import nullcontext

from time import sleep

# NumPy wrapper:
def coxph_numpy(
    *,
    x,
    times,
    deaths,
    **kwargs,
):
    x = torch.tensor(x, dtype=float32, device=device)
    times = torch.tensor(times, dtype=int32, device=device)
    deaths = torch.tensor(deaths, dtype=int32, device=device)

    result = coxph_torch(x=x, times=times, deaths=deaths, **kwargs)
    result = {k: numpy(v) for k, v in result.items()}
    return result


def coxph_R(
    data,
    stop,
    death,
    covars,
    bootstrap=1,
    batchsize=0,
    ties="efron",
    maxiter=20,
    init=None,
    doscale=False,
    profile=None,
):

    if profile is not None:
        print("Profile trace:", profile)
        print("use_cuda:", use_cuda)

    with torch.autograd.profiler.profile(
        use_cuda=use_cuda
    ) if profile is not None else nullcontext() as prof:

        times = np.array(data[stop])
        deaths = np.array(data[death])
        N = len(times)

        cov = [data[covar] for covar in covars]
        x = np.array(cov).T.reshape([N, len(cov)])

        res = coxph_numpy(
            x=x,
            times=times,
            deaths=deaths,
            ties=ties,
            backend="torch",
            bootstrap=int(bootstrap),
            batchsize=int(batchsize),
            maxiter=int(maxiter) if profile is None else 1,
            init=init,
            verbosity=0,
            alpha=0.0,
            doscale=doscale,
        )


    if profile is not None:
        prof.export_chrome_trace(profile)

    return res
