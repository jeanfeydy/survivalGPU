# Use NumPy for basic array manipulation:
import numpy as np

# We use matplotlib to display the results:
from matplotlib import pyplot as plt

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



class WCESurvivalAnalysis:
    def __init__(self, *, cutoff, nknots=1, order=3, constrained="Right"):
        
        # Let the model remember the parameters of the analysis:
        self.cutoff = cutoff
        self.knots = knots
        self.atoms = atoms
        self.atom_areas = areas

    def fit(self, *, doses, times, events, covariates=None):
        

        # Rough Gaussian-like esimation of the confidence intervals for the total risk area: -----
        # We use a simple model:
        # the ideal coefs are the minimizers of the neglog-likelihood function of the CoxPH model,
        # with gradient = 0 at the optimum and a Hessian that is a positive-definite matrix
        # of shape (Features, Features) for each drug.
        # For each drug, we may reasonably expect the "coefs" vector to follow a Gaussian
        # distribution with:
        # - mean = estimated vector "coefs[drug]"
        # - covariance = Imat[drug] = inverse(Hessian[drug]).
        #
        # In this context, the total risk area = \sum_{b-spline atom i} areas[i] * coef[i]
        # is a 1D-Gaussian vector with:
        # - mean[drug] = \sum_{b-spline atom i} areas[i] * estimated_coefs[drug,i]
        # - variance[drug] = \sum_{i, j} areas[i] * areas[j] *

        risk_variances = torch.einsum("ijk,j,k->i", Imat, areas, areas)
        risk_stds = risk_variances.sqrt()
        print("Estimated log-HR:", form(risk_means), "+-", form(risk_stds))

        assert risk_variances.shape == (Drugs,)
        assert risk_stds.shape == (Drugs,)

        # ci_95 = 1.96 / np.sqrt(coefs.shape[-1])

        # (Drugs, Features, Features) @ (Features,)
        ci_95 = Imat @ areas
        ci_95 = 1.96 * ci_95 / risk_stds.view(Drugs, 1)
        assert ci_95.shape == (Drugs, Features)

        if False:
            print("Area deltas for the 95% CI:")
            print(form(ci_95 @ areas))
            print("Expected values:")
            print(form(1.96 * risk_stds))


    def display_atoms(self, ax=None):

        ax = plt.gca() if ax is None else ax
        ax.title("B-Spline atoms")
        for i, f in enumerate(atoms.t()):
            ax.plot(x, numpy(f), label=f"{i}")
        ax.legend()

    def display_risk_functions(self, ax=None):

        ax = plt.gca() if ax is None else ax
        ax.title("Estimated risk functions, with 95% CI for the total risk area")
        for i, (coef, ci) in enumerate(zip(coefs, ci_95)):
            ax.plot(numpy(atoms @ coef), label=f"{i}")
            ax.fill_between(
                x, numpy(atoms @ (coef - ci)), numpy(atoms @ (coef + ci)), alpha=0.2
            )
        ax.legend()

    def display_risk_distribution(self, *, drug, ax=None):

        ax = plt.gca() if ax is None else ax
        ax.title(f"Distribution of the total risk for drug {drug}")
        
        t = np.linspace(bootstrap_risk.min().item(), bootstrap_risk.max().item(), 100)
        plt.plot(
            t,
            np.exp(-0.5 * (t - risk_mean_est) ** 2 / risk_std_est**2)
            / np.sqrt(2 * np.pi * risk_std_est**2),
            label="Estimation",
        )
        plt.hist(
            numpy(bootstrap_risk),
            density=True,
            histtype="step",
            bins=50,
            log=True,
            label="Bootstrap",
        )
        plt.legend()