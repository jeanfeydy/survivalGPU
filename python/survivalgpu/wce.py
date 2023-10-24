# Use NumPy for basic array manipulation:
import numpy as np

# We use matplotlib to display the results:
from matplotlib import pyplot as plt

# Use PyTorch for fast array manipulations (on the GPU):
import torch

from .coxph import CoxPHSurvivalAnalysis

from .utils import numpy, timer
from .utils import use_cuda, device, float32, int32, int64
from .wce_features import wce_features_batch, bspline_atoms

from .typecheck import typecheck, Optional, Literal
from .typecheck import Int, Real, Bool
from .typecheck import Int64Array, Float64Array
from .typecheck import Float32Tensor
from .typecheck import TorchDevice


# Our main, object-oriented API ==========================================================


class WCESurvivalAnalysis:
    @typecheck
    def __init__(
        self,
        *,
        cutoff: Int,
        n_knots: Int = 1,
        order: Int = 3,
        constrained: Optional[Literal["right", "left"]] = None,
        survival_model=CoxPHSurvivalAnalysis(),
    ):
        """Weighted Cumulative Exposure Model that combines B-spline time-varying features with a CoxPH analysis.

        This model is fully described in...


        The total number of degrees of freedom for the risk function (i.e. WCE covariates)
        is equal to:
            n_knots + order + 1 if constrained is None,
            n_knots + 2         if constrained is "left" or "right".

        Args:
            cutoff (int): size of the time window for the risk function.
            n_knots (int): number of knots for the B-splines.
            order (int): order of the B-splines used to model the risk function.
                order = 0 corresponds to a piecewise constant risk function,
                order = 1 corresponds to a piecewise linear risk function,
                order = 3 corresponds to a piecewise cubic risk function.
            constrained (str, optional): whether the B-splines should be constrained.
                Defaults to None (i.e. no constrain).
                Other options are:
                - "Left" or "L": the drug has no immediate effect on the risk.
                  We remove features that correspond to basis functions that have
                  a non-zero value or derivative on the "left" of the domain,
                  i.e. around the exposure time.
                  This is useful to model a risk function that has no "immediate" impact.

                - "Right" or "R": the drug has no effect on the risk around the cutoff time.
                  We remove features that correspond to basis functions that have
                  a non-zero value or derivative on the "right" of the domain,
                  i.e. around the "exposure+cutoff" time.
                  This is useful to model a risk function that vanishes "at infinity".
            survival_model (Estimator, optional): estimator that will be used to
                perform a risk analysis from the WCE covariates.
                For now, we only support the CoxPHSurvivalAnalysis model.
        """
        # Let the model remember the parameters of the analysis.
        # Note that all type and value checks are performed in the attribute setters:
        self.order = order
        self.cutoff = cutoff
        self.n_knots = n_knots
        self.constrained = constrained
        self.survival_model = survival_model

    def set_non_negative_int(self, value, name):
        if int(value) != value:
            raise TypeError(
                f"{name} should be an integer. "
                f"Received {value} of type {type(value)}."
            )
        elif int(value) < 0:
            raise ValueError(f"{name} should be >= 0. " f"Received {value}.")
        else:
            setattr(self, "_" + name, int(value))

    # The order should be an integer >= 0 --------------------------------
    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, new_o):
        self.set_non_negative_int(new_o, "order")

    # The number of extra knots should be an integer >= 0 --------------------------------
    @property
    def n_knots(self):
        return self._n_knots

    @n_knots.setter
    def n_knots(self, new_n):
        self.set_non_negative_int(new_n, "n_knots")

    # The cutoff value should be an integer >= 0 -----------------------------------------
    @property
    def cutoff(self):
        return self._cutoff

    @cutoff.setter
    def cutoff(self, new_cutoff):
        self.set_non_negative_int(new_cutoff, "cutoff")

    # "Constrained" only accepts three values: None, "Left" and "Right" ------------------
    @property
    def constrained(self):
        return self._constrained

    @constrained.setter
    def constrained(self, new_c):
        supported_values = [None, "left", "right"]
        if new_c not in supported_values:
            raise ValueError(
                f"constrained should be one of {supported_values}. "
                f"Received {new_c}."
            )
        else:
            self._constrained = new_c

    # The number of WCE features depends on n_knots, the order and constrained -----------
    @property
    def n_atoms(self):
        if self.constrained is None:
            return self.n_knots + self.order + 1
        else:
            return self.n_knots + 2

    # Functions related to the B-Spline atoms --------------------------------------------
    def _constrain(self, features):
        """Enforces a boundary condition on the B-Spline by discarding some basis functions.

        Args:
            features ((N,D) tensor): Time-dependent WCE features.
                Each line corresponds to a sampling time.
                Each column corresponds to a WCE basis function (= "atom").

        Returns:
            truncated features ((N,D) or (N,D-(order-1)) tensor: Relevant WCE features.
        """
        assert len(features.shape) == 2
        assert features.shape[1] == self.n_knots + self.order + 1

        if self.constrained == "right":
            return features[:, : -(self.order - 1)]

        elif self.constrained == "left":
            return features[:, (self.order - 1) :]

        elif self.constrained is None:
            return features
        else:
            raise ValueError(
                "constrained should be None, 'left' or 'right'. "
                f"Received {self.constrained}."
            )

    @property
    def atoms(self):
        """Samples the B-spline basis functions on the interval [0, cutoff-1]."""
        atoms, _ = bspline_atoms(
            cutoff=self.cutoff, order=self.order, knots=self.n_knots
        )
        atoms = self._constrain(atoms)
        assert atoms.shape == (self.cutoff, self.n_atoms)
        return atoms

    @property
    def atom_areas(self):
        """Compute the "total risk area under the curve" that is associated to the B-spline basis functions."""
        areas = self.atoms.sum(0)  # {Cutoff, Features) -> (Features,)
        assert areas.shape == (self.n_atoms,)
        return areas

    # Computation of the WCE features ====================================================

    def _wce_features(self, *, ids, doses, times):
        """Computes the WCE B-Spline covariates on a batch of patients and drugs."""
        wce_features, knots = wce_features_batch(
            ids=ids,
            times=times,
            doses=doses,
            nknots=self.n_knots,
            cutoff=self.cutoff,
            order=self.order,
        )
        wce_features = self._constrain(wce_features)
        assert wce_features.shape == (len(times), self.n_atoms)
        return wce_features, knots

    def fit(
        self,
        *,
        doses: Float64Array["intervals"],
        stop: Int64Array["intervals"],
        start: Int64Array["intervals"],
        event: Int64Array["intervals"],
        patient: Int64Array["intervals"],
        covariates: Optional[Float64Array["intervals covariates"]] = None,
        strata: Optional[Int64Array["patients"]] = None,
        batch: Optional[Int64Array["patients"]] = None,
        init: Optional[Float64Array["fullcovariates"]] = None,
        n_bootstraps: Optional[Int] = None,
        batch_size: Optional[Int] = None,
        device: Optional[TorchDevice] = None,
    ):
        if not torch.all(stop == start + 1):
            raise NotImplementedError(
                "Currently, we only support unit length intervals."
            )

        # Step 1: compute the time-dependent features (= exposures)
        exposures, knots = self._wce_features(ids=patient, doses=doses, times=stop)
        assert exposures.shape == (len(stop), self.n_atoms)

        # Step 2: perform a CoxPH regression with the new covariates
        if covariates is None:
            # No external covariates, just drug doses:
            self.n_covariates = 0
            covariates = exposures
        else:
            # We observe other covariates such as the sex, etc.
            self.n_covariates = covariates.shape[-1]
            covariates = torch.cat((covariates, exposures), dim=-1)

        self.survival_model.fit(
            covariates=covariates,
            start=start,
            stop=stop,
            event=event,
            patient=patient,
            strata=strata,
            batch=batch,
            init=init,
            n_bootstraps=n_bootstraps,
            batch_size=batch_size,
            device=device,
        )

        # Step 3: Save the results in the expected format
        # Save the knots values:
        self.knots_ = knots

        # Optimal coefficients: ------------------------------------------------
        n_batch = len(self.survival_model.coef_)
        assert self.survival_model.coef_.shape == (
            n_batch,
            self.n_covariates + self.n_atoms,
        )

        # Coefficients for the covariates:
        self.coef_ = self.survival_model.coef_[:, : self.n_covariates]
        assert self.coef_.shape == (n_batch, self.n_covariates)

        # Coefficients for the WCE B-Spline features:
        self.WCE_coef_ = self.survival_model.coef_[:, self.n_covariates :]
        assert self.WCE_coef_.shape == (n_batch, self.n_atoms)
        # Estimated risk function:
        # (n_batch, n_atoms) @ (n_atoms, cutoff) -> (n_batch, cutoff)
        self.risk_function_ = self.WCE_coef_ @ self.atoms.T
        assert self.risk_function_.shape == (n_batch, self.cutoff)

        # Standard deviations for the coefficients:
        self.std_ = self.survival_model.std_[:, : self.n_covariates]
        assert self.std_.shape == (n_batch, self.n_covariates)

        # Standard deviations for the WCE B-Spline weights:
        self.SED_ = self.survival_model.std_[:, self.n_covariates :]
        assert self.SED_.shape == (n_batch, self.n_atoms)

        # Batch coefficients: --------------------------------------------------
        if n_bootstraps is not None:
            assert self.survival_model.bootstrap_coef_.shape == (
                n_bootstraps,
                n_batch,
                self.n_covariates + self.n_atoms,
            )

            # Bootstrap coefficients for the covariates:
            self.bootstrap_coef_ = self.survival_model.bootstrap_coef_[
                :, :, : self.n_covariates
            ]
            assert self.bootstrap_coef_.shape == (
                n_bootstraps,
                n_batch,
                self.n_covariates,
            )

            # Bootstrap weights for the WCE B-Spline features:
            self.bootstrap_WCE_coef_ = self.survival_model.bootstrap_coef_[
                :, :, self.n_covariates :
            ]
            assert self.bootstrap_WCE_coef_.shape == (
                n_bootstraps,
                n_batch,
                self.n_atoms,
            )

            # Estimated risk function:
            # (n_bootstraps, n_batch, n_atoms) @ (n_atoms, cutoff) -> (n_bootstraps, n_batch, cutoff)
            self.bootstrap_risk_functions_ = self.bootstrap_WCE_coef_ @ self.atoms.T

        # Usual CoxPH results: -------------------------------------------------
        self.means_ = self.survival_model.means_
        self.score_ = self.survival_model.score_
        self.sctest_init_ = self.survival_model.sctest_init_
        self.loglik_init_ = self.survival_model.loglik_init_
        self.loglik_ = self.survival_model.loglik_
        self.hessian_ = self.survival_model.hessian_
        self.imat_ = self.survival_model.imat_
        self.iter_ = self.survival_model.iter_


def wce_numpy(
    *,
    ids,
    covariates,
    doses,
    events,
    times,
    cutoff: Int,
    n_knots: Int = 1,
    order: Int = 3,
    constrained: Optional[Literal["right", "left"]] = None,
    strata: Optional[Int64Array["patients"]] = None,
    batch: Optional[Int64Array["patients"]] = None,
    init: Optional[Float64Array["fullcovariates"]] = None,
    n_bootstraps: Optional[Int] = None,
    batch_size: Optional[Int] = None,
    device: Optional[TorchDevice] = None,
    **kwargs,
):
    surv_model = CoxPHSurvivalAnalysis(**kwargs)
    model = WCESurvivalAnalysis(
        cutoff=cutoff,
        n_knots=n_knots,
        order=order,
        constrained=constrained,
        survival_model=surv_model,
    )

    model.fit(
        doses=doses,
        stop=times,
        start=times - 1,
        event=events,
        patient=ids,
        covariates=covariates,
        strata=strata,
        batch=batch,
        init=init,
        n_bootstraps=n_bootstraps,
        batch_size=batch_size,
        device=device,
    )

    output = dict(
        knots=model.knots_,
        coef=model.coef_,
        WCE_coef=model.WCE_coef_,
        risk_function=model.risk_function_,
        std=model.std_,
        SED=model.SED_,
        means=model.means_,
        score=model.score_,
        sctest_init=model.sctest_init_,
        loglik_init=model.loglik_init_,
        loglik=model.loglik_,
        hessian=model.hessian_,
        imat=model.imat_,
        iter=model.iter_,
    )

    if n_bootstraps is not None:
        output.update(
            bootstrap_coef=model.bootstrap_coef_,
            bootstrap_WCE_coef=model.bootstrap_WCE_coef_,
            bootstrap_risk_functions=model.bootstrap_risk_functions_,
        )

    return output


# Python >= 3.7:
from contextlib import nullcontext


def wce_R(
    *,
    data,
    ids,
    covars,
    stop,
    doses,
    events,
    # WCE parameters:
    cutoff,
    n_knots=1,
    order=3,
    constrained=None,
    bootstrap=1,
    # Cox parameters:
    profile=None,
    batchsize=0,
    ties="efron",
    maxiter=20,
    init=None,
    doscale=False,
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

    if profile is not None:
        print("Profile trace:", profile)
        print("use_cuda:", use_cuda)
        myprof = torch.autograd.profiler.profile(use_cuda=use_cuda)
    else:
        myprof = nullcontext()

    if strata is not None:
        strata = np.array(strata, dtype=np.int64)

    with myprof as prof:
        res = wce_numpy(
            ids=ids,
            covariates=covariates,
            doses=doses,
            events=events,
            times=times,
            cutoff=int(cutoff),
            n_knots=int(n_knots),
            order=int(order),
            constrained=constrained,
            strata=strata,
            batch=None,
            init=init,
            n_bootstraps=int(bootstrap),
            batch_size=int(batchsize) if batchsize > 0 else None,
            device=device,
            maxiter=int(maxiter),
            ties=ties,
            doscale=doscale,
        )

    if profile is not None:
        prof.export_chrome_trace(profile)

    res.update(
        WCEmat=res["bootstrap_risk_functions"],
        est=res["bootstrap_WCE_coef"],
        vcovmat=res["imat"],
    )

    return res


if False:
    # Obsolete code below? ===============================================================

    class Deprecated:
        @property
        def n_drugs(self):
            """The number of drugs that have been fitted."""
            return self._n_drugs

        @property
        def Imat(self):
            """The inverse of the Hessian matrix for each drug."""
            return self._Imat

        # Rough Gaussian-like estimation of the confidence intervals for the total risk area: -----

        def _wce_features_multidrugs(self, *, doses, times):
            """Computes the WCE B-Spline covariates on a batch of patients and drugs.

            TODO: use this code somewhere?

            Args:
                doses ((Drugs, Patients, Times) float32 Tensor): a table of drug consumption
                    doses, for an arbitrary number of drugs, patients and sampling times.
                times ((Times,) int32 tensor): the time values that correspond to the doses.

            Returns:
                (Drugs, Patients, Times, n_atoms) float32 Tensor: the WCE B-spline covariates
                    that encode, for each patient, at each time, the cumulated exposure
                    to each drug through a vector of n_atoms correlations with the basis
                    B-spline functions.
            """
            Drugs, Patients, Times = doses.shape
            device = doses.device
            N = Drugs * Patients * Times

            # We specify a batch computation of (Drugs * Patients) blocks of
            # "Times" values.
            # Assuming that Patients = 3 and Drugs = 2,
            # ids [0, 1, 2, 3, 4, 5]
            ids = torch.arange(Drugs * Patients, dtype=torch.int32, device=device)
            # Then, repeat the ids to create a nice "batch vector"
            # of size (Drugs * Patients * Times). Assuming that Times = 2,
            # ids = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
            ids = ids.view(Drugs * Patients, 1).tile([1, Times]).view(N)

            exposures, _ = wce_features_batch(
                ids=ids,
                times=times.tile([Drugs * Patients]).view(N),  # [0, 1, 0, 1, ...]
                doses=doses.view(N),
                nknots=self.n_knots,
                cutoff=self.cutoff,
                order=self.order,
            )
            assert exposures.shape == (
                Drugs * Patients * Times,
                self.n_knots + self.order + 1,
            )
            # Remove some of the covariates if required:
            exposures = self._constrain(exposures)
            exposures = exposures.view(Drugs, Patients, Times, self.n_atoms)
            return exposures

        @property
        def drug_total_risks(self):
            r"""Returns the mean and std of the total risk area for each drug.

            The total risk area is the area under the curve of the risk function.
            It corresponds to the logarithm of the Hazard Ratio that is associated
            to a dose of 1 unit of the drug.
            It is computed as a weighted sum of the risk areas of each b-spline atom.

            We use a simple heuristic to estimate the std of the total risk area for each drug.
            Recall that the coefficients of the fitted model are the minimizers of
            the neglog-likelihood function of the CoxPH model, with gradient = 0
            at the optimum and a Hessian that is a positive-definite matrix
            of shape (n_atoms, n_atoms) for each drug.

            For each drug, we may then reasonably expect the "coefs" vector to follow
            a Gaussian distribution with:
            - mean = estimated vector "coefs[drug]" of shape (n_atoms,)
            - covariance = Imat[drug] = inverse(Hessian[drug]) of shape (n_atoms, n_atoms).

            In this context,
            total risk area = \sum_{b-spline atom i} atom_areas[i] * coef[i]
            is a 1D-Gaussian vector with:
            - mean[drug] = \sum_{b-spline atom i} areas[i] * estimated_coefs[drug,i]
            - variance[drug] = \sum_{i, j} areas[i] * areas[j] * Imat[drug, i, j]
            """
            coefs = self.coefs  # (n_drugs, n_atoms)
            areas = self.atom_areas  # (n_atoms,)
            risk_means = torch.einsum("di,i->d", coefs, areas)  # (n_drugs,)

            Imat = self.Imat  # (n_drugs, n_atoms, n_atoms)
            risk_variances = torch.einsum(
                "dij,i,j->d", Imat, areas, areas
            )  # (n_drugs,)
            risk_stds = risk_variances.sqrt()

            assert risk_means.shape == (self.n_drugs,)
            assert risk_stds.shape == (self.n_drugs,)
            return risk_means, risk_stds

        @property
        def drug_coeff_ci_95(self):
            # ci_95 = 1.96 / np.sqrt(coefs.shape[-1])

            # (Drugs, Features, Features) @ (Features,)
            ci_95 = Imat @ areas
            ci_95 = 1.96 * ci_95 / risk_stds.view(Drugs, 1)
            assert ci_95.shape == (Drugs, Features)
            assert ci_95 @ areas == 1.96 * risk_stds

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

            t = np.linspace(
                bootstrap_risk.min().item(), bootstrap_risk.max().item(), 100
            )
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
        weights = result["coef"][
            :, ncovariates:
        ]  # (B,D) tensor of B-Spline coefficients

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
