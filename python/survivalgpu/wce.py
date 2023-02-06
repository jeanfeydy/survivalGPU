# Use NumPy for basic array manipulation:
import numpy as np

# Use PyTorch for fast array manipulations (on the GPU):
import torch

from .coxph import coxph_torch

from .utils import numpy, timer
from .utils import use_cuda, device, float32, int32, int64
from .wce_features import wce_features_batch, bspline_atoms


def constrain(*, features, constrained, order):
    """Enforces a boundary condition on the B-Spline by discarding some basis functions.

    Args:
        features ((N,D) tensor): Time-dependent WCE features.
            Each line corresponds to a sampling time.
            Each column corresponds
        constrained (string or None): Boundary constraint.
            If constrained == "R" or "Right", we remove features that correspond
            to basis functions that have a non-zero value or derivative
            on the "right" of the domain, i.e. around the "exposure+cutoff" time.
            This is useful to model a risk function that vanishes "at infinity".
            If constrained == "L" or "Left", we remove features that correspond
            to basis functions that have a non-zero value or derivative
            on the "left" of the domain, i.e. around the exposure time.
            This is useful to model a risk function that has no "immediate" impact.
            For other values of constrained, this function does not do anything.
        order (int): The order of the B-Spline functions (3 for cubic, etc.).

    Returns:
        truncated features ((N,D) or (N,D-(order-1)) tensor: Relevant WCE features.
    """
    if constrained in ["R", "Right"]:
        return features[:, : -(order - 1)]
    elif constrained in ["L", "Left"]:
        return features[:, (order - 1) :]
    else:
        return features


def wce_torch(
    *,
    ids,
    covariates,
    doses,
    events,
    times,
    cutoff,
    constrained=None,
    nknots=1,
    order=3,
    bootstrap=1,
    batchsize=0,
    verbosity=1,
):

    # Just in case the user provided float numbers (super easy with R...):
    nknots = int(nknots)
    order = int(order)
    bootstrap = int(bootstrap)
    batchsize = int(batchsize)

    # Step 1: compute the time-dependent features (= exposures) ================
    if verbosity > 0:
        tstart = timer()
        print("Step 1 : Computing the WCE features... ", end="", flush=True)

    wce_features, knots = wce_features_batch(
        ids=ids,
        times=times,
        doses=doses,
        nknots=nknots,
        cutoff=cutoff,
        order=order,
    )

    # If constrained == "Right", we remove the B-Spline atoms that
    # correspond to the end of the observation window.
    # If constrainted == "Left", we remove the start of the observation window.
    wce_features = constrain(
        features=wce_features, constrained=constrained, order=order
    )

    # Step 2: perform a CoxPH regression with the new covariates ===============
    if verbosity > 0:
        print(f"Done in {timer() - tstart:.3f}s.\n")
        print("Step 2 : CoxPH regression... ")
        tstart = timer()

    if covariates is None:
        # No external covariates, just drug doses:
        ncovariates = 0
        covariates = wce_features
    else:
        # We observe other covariates such as the sex, etc.
        ncovariates = covariates.shape[-1]
        covariates = torch.cat((covariates, wce_features), dim=-1)

    result = coxph_torch(
        x=covariates,
        times=times,
        deaths=events,
        bootstrap=bootstrap,
        batchsize=batchsize,
        verbosity=verbosity,
    )

    # Step 3: save the results in the expected format ==========================
    if verbosity > 0:
        print(f"Done in {timer() - tstart:.3f}s.\n")
        print("Step 3 : Post-processing... ", end="", flush=True)
        tstart = timer()

    # Save the knots values:
    result["knotsmat"] = knots

    # Estimate the standard deviations of the coefficients for the covariates:
    variances = torch.diagonal(result["imat"], dim1=1, dim2=2)
    stds = torch.sqrt(variances)
    result["std"] = stds[:, :ncovariates]
    result["SED"] = stds[:, ncovariates:]

    # Sample the estimated risk function (one per bootstrap) on the
    # interval [1, 2, ..., cutoff]:
    weights = result["coef"][:, ncovariates:]  # (B,D) tensor of B-Spline coefficients

    # Sample the B-Spline basis functions:
    atoms, _ = bspline_atoms(cutoff=cutoff, order=order, knots=knots)
    atoms = constrain(features=atoms, constrained=constrained, order=order)

    # Take the linear combinations: (1,cutoff,D) @ (B,1,D) -> (B,cutoff)
    result["WCEmat"] = (
        atoms.view(1, len(atoms), -1) * weights.view(bootstrap, 1, -1)
    ).sum(-1)
    result["est"] = weights

    # Save the estimated variance-covariance matrix for the parameters of the model:
    result["vcovmat"] = result["imat"]

    if verbosity > 0:
        print(f"Done in {timer() - tstart:.3f}s.\n")

    return result


# Python >= 3.7:
from contextlib import nullcontext


def wce_numpy(
    *,
    ids,
    covariates,
    doses,
    events,
    times,
    profile=None,
    **kwargs,
):
    with torch.autograd.profiler.profile(
        use_cuda=use_cuda
    ) if profile is not None else nullcontext() as prof:

        ids = torch.tensor(ids, dtype=int32, device=device)
        if covariates is not None:
            covariates = torch.tensor(covariates, dtype=float32, device=device)
        doses = torch.tensor(doses, dtype=float32, device=device)
        times = torch.tensor(times, dtype=int32, device=device)
        events = torch.tensor(events, dtype=int32, device=device)

        result = wce_torch(
            ids=ids,
            covariates=covariates,
            doses=doses,
            events=events,
            times=times,
            **kwargs,
        )

    if profile is not None:
        prof.export_chrome_trace(profile)

    result = {k: numpy(v) for k, v in result.items()}
    return result


def wce_R(
    *,
    data,
    ids,
    covars,
    stop,
    doses,
    events,
    **kwargs,
):

    ids = np.array(data[ids])
    doses = np.array(data[doses])
    times = np.array(data[stop])
    events = np.array(data[events])
    N = len(times)

    if covars is not None and len(covars) > 0:
        cov = [data[covar] for covar in covars]
        covariates = np.array(cov).reshape([len(cov), N]).T.reshape([N, len(cov)])
    else:
        covariates = None

    res = wce_numpy(
        ids=ids,
        covariates=covariates,
        doses=doses,
        events=events,
        times=times,
        **kwargs,
    )

    return res
