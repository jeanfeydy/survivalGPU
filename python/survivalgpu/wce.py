# Use NumPy for basic array manipulation:
# Python >= 3.7:
from contextlib import nullcontext

import numpy as np

# Use PyTorch for fast array manipulations (on the GPU):
import torch

from .coxph import CoxPHSurvivalAnalysis
from .typecheck import (
    Float64Array,
    Int,
    Int64Array,
    Literal,
    TorchDevice,
    typecheck,
)
from .utils import default_device, float32, float64, use_cuda
from .wce_features import bspline_atoms, wce_features_batch


class WCESurvivalAnalysis:
    @typecheck
    def __init__(
        self,
        *,
        cutoff: Int,
        nknots: Int = 1,
        order: Int = 3,
        constrained: Literal["right", "left"] | None = None,
        knot_placement: Literal["quantile", "uniform"] = "quantile",
        criterion: Literal["aic", "bic"] = "bic",
        survival_model=None,
        dtype = np.float64,
        device = None,
        nbootstraps: Int | None = None,
        batchsize: Int | None = None,

    ):
        """Weighted Cumulative Exposure Model that combines B-spline time-varying features with a CoxPH analysis.

        The total number of degrees of freedom for the risk function (i.e. WCE covariates)
        is equal to:
            nknots + order + 1 if constrained is None,
            nknots + 2         if constrained is "left" or "right".

        Parameters
    ----------
        cutoff
            Size of the time window for the risk function.
        nknots
            Number of knots for the B-splines.
        order
            Order of the B-splines used to model the risk function.
            `order == 0` corresponds to a piecewise constant risk function,
            `order == 1` corresponds to a piecewise linear risk function,
            `order == 3` corresponds to a piecewise cubic risk function.
        criterion
            Which information criterion to report in `self.info_criterion_`: "aic"
            (penalty = 2 per degree of freedom) or "bic" (penalty =
            log(n_events) per degree of freedom). Defaults to "bic".
            Matches the `my_bic_c()` formula from the reference `WCE` R
            package.
        constrained
            Whether the B-splines should be constrained.
            Defaults to None (i.e. no constraint).

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
        knot_placement
            How to position the B-Spline knots. Defaults to "quantile".

            - "quantile": matches the WCE R package convention. The `nknots`
                inner knots sit at regular quantiles of the time window, but the
                boundary "padding" knots are always spaced by 1, regardless of
                the interior spacing. For a large cutoff relative to nknots,
                this makes the boundary basis functions narrower than the
                interior ones.

            - "uniform": every knot -- inner and boundary padding alike -- is
                spaced by the same amount, cutoff / (nknots + 1), so all basis
                functions have the same width. This is the design required by
                P-splines (Eilers & Marx, 1996): their difference penalty on
                adjacent coefficients assumes equally-spaced knots, and the
                uneven "quantile" spacing distorts that penalty near the
                boundaries.
        survival_model
            Estimator that will be used to
            perform a risk analysis from the WCE covariates.
            For now, we only support the CoxPHSurvivalAnalysis model.
        dtype
            Either np.float32 or np.float64. Defaults to np.float64.
        device
            Device (e.g. "cpu" or "cuda") on which to run the computations.
            Defaults to the best available device.
        nbootstraps
            Number of bootstrap resamples to fit, in addition to the main model.
            Defaults to None (i.e. no bootstrapping).
        batchsize
            Number of bootstrap resamples to process at once on the GPU.
            Defaults to None, i.e. all the bootstraps are processed at once.
        """
        # Let the model remember the parameters of the analysis.
        # Note that all type and value checks are performed in the attribute setters:
        self.order = order
        self.cutoff = cutoff
        self.nknots = nknots
        self.constrained = constrained
        self.knot_placement = knot_placement
        self.criterion = criterion


        if survival_model is None:
            survival_model = CoxPHSurvivalAnalysis()

        if isinstance(survival_model, CoxPHSurvivalAnalysis):
            survival_model = CoxPHSurvivalAnalysis(
                maxiter=20, device=device, nbootstraps=nbootstraps, batchsize=batchsize, dtype = dtype
            )


        self.survival_model = survival_model



        if dtype == np.float32:
            self.dtype = float32
        elif dtype == np.float64:
            self.dtype = float64
        else:
            msg = f"dtype should be np.float32 or np.float64. Received {dtype}."
            raise ValueError(msg)

        self.device = device if device is not None else default_device

        if nbootstraps == 0:
            nbootstraps = None

        self.nbootstraps = nbootstraps
        self.batchsize = batchsize





    def set_non_negative_int(self, value, name):
        if int(value) != value:
            msg = (
                f"{name} should be an integer. "
                f"Received {value} of type {type(value)}."
            )
            raise TypeError(msg)

        if int(value) < 0:
            msg = f"{name} should be >= 0. " f"Received {value}."
            raise ValueError(msg)

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
    def nknots(self):
        return self._nknots

    @nknots.setter
    def nknots(self, new_n):
        self.set_non_negative_int(new_n, "nknots")

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
            msg = (
                f"constrained should be one of {supported_values}. "
                f"Received {new_c}."
            )
            raise ValueError(msg)
        self._constrained = new_c

    # "knot_placement" only accepts "quantile" and "uniform" -----------------------------
    @property
    def knot_placement(self):
        return self._knot_placement

    @knot_placement.setter
    def knot_placement(self, new_k):
        supported_values = ["quantile", "uniform"]
        if new_k not in supported_values:
            msg = (
                f"knot_placement should be one of {supported_values}. "
                f"Received {new_k}."
            )
            raise ValueError(msg)
        self._knot_placement = new_k

    # The number of WCE features depends on nknots, the order and constrained -----------
    @property
    def n_atoms(self):
        if self.constrained is None:
            return self.nknots + self.order + 1
        else:
            # TODO: fix when self.order != 3
            return self.nknots + 2

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
        assert features.shape[1] == self.nknots + self.order + 1

        # TODO: fix when self.order != 3

        if self.constrained == "right":
            return features[:, : -(self.order - 1)]

        elif self.constrained == "left":
            return features[:, (self.order - 1) :]

        elif self.constrained is None:
            return features
        else:
            msg = (
                "constrained should be None, 'left' or 'right'. "
                f"Received {self.constrained}."
            )
            raise ValueError(msg)

    @property
    def atoms(self):
        """Samples the B-spline basis functions on the interval [0, cutoff-1]."""
        atoms, _ = bspline_atoms(
            cutoff=self.cutoff, order=self.order, nknots=self.nknots,
            knot_placement=self.knot_placement, dtype=self.dtype,
            device=self.device,
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

    def plot_basis(self, ax=None, show=True):
        """Plots the B-spline basis functions ("atoms") used to model the risk function.

        This is useful to visually check that a given basis configuration
        (order, nknots, constrained, ...) behaves as expected.

        Args:
            ax (matplotlib.axes.Axes, optional): axes to draw on.
                Defaults to None, i.e. a new figure and axes are created.
            show (bool, optional): whether to call plt.show(). Defaults to True.

        Returns:
            matplotlib.axes.Axes: the axes that were used for the plot.
        """
        import matplotlib.pyplot as plt

        atoms = self.atoms.detach().cpu().numpy()

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        for i, atom in enumerate(atoms.T):
            ax.plot(atom, label=f"Atom {i}")

        ax.plot(atoms.sum(1), "--", color="black", label="Sum")
        ax.set_xlabel("Time")
        ax.set_ylabel("Basis value")
        ax.set_title(
            f"B-spline basis (order={self.order}, nknots={self.nknots}, "
            f"constrained={self.constrained}, knot_placement={self.knot_placement})"
        )
        ax.legend()

        if show:
            plt.show()

        return ax

    def plot_fitted_basis(self, index=0, ax=None, show=True):
        """Plots the fitted B-spline atoms, scaled by their estimated coefficients.

        Requires the model to have been `.fit()` first: each basis function is
        multiplied by its corresponding entry in `WCE_coef_`, and their sum
        (the dashed curve) is exactly `risk_function_`. Useful to see how much
        each atom actually contributes to the fitted risk function.

        Args:
            index (int, optional): which row of `WCE_coef_` to plot -- the
                main fit is at index 0, bootstrap resamples (if any) come
                after. Defaults to 0.
            ax (matplotlib.axes.Axes, optional): axes to draw on.
                Defaults to None, i.e. a new figure and axes are created.
            show (bool, optional): whether to call plt.show(). Defaults to True.

        Returns:
            matplotlib.axes.Axes: the axes that were used for the plot.
        """
        import matplotlib.pyplot as plt

        atoms = self.atoms.detach().cpu().numpy()  # (cutoff, n_atoms)
        coef = self.WCE_coef_[index]  # (n_atoms,)
        scaled_atoms = atoms * coef[None, :]

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        for i, atom in enumerate(scaled_atoms.T):
            ax.plot(atom, label=f"Atom {i} * coef")

        ax.plot(
            self.risk_function_[index].detach().cpu().numpy(),
            "--", color="black", linewidth=2, label="Risk function (sum)",
        )
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_xlabel("Time")
        ax.set_ylabel("Log hazard contribution")
        ax.set_title(
            f"Fitted B-spline basis (order={self.order}, nknots={self.nknots}, "
            f"constrained={self.constrained}, knot_placement={self.knot_placement})"
        )
        ax.legend()

        if show:
            plt.show()

        return ax

    # Computation of the WCE features ====================================================

    @typecheck
    def _wce_features(
        self,
        *,
        patient: Int64Array["intervals"],
        dose: Float64Array["intervals"],
        time: Int64Array["intervals"],
    ):
        """Computes the WCE B-Spline covariates on a batch of patients and drugs."""

        patient = torch.from_numpy(patient).to(self.device)
        dose = torch.from_numpy(dose).to(self.device)
        time = torch.from_numpy(time).to(self.device)

        wce_features, knots = wce_features_batch(
            ids=patient,
            times=time,
            doses=dose,
            nknots=self.nknots,
            cutoff=self.cutoff,
            order=self.order,
            knot_placement=self.knot_placement,
            dtype=self.dtype,
            device=self.device,
        )

        wce_features = wce_features.cpu().numpy()
        knots = knots.cpu().numpy()

        wce_features = self._constrain(wce_features)
        assert wce_features.shape == (len(time), self.n_atoms)
        return wce_features, knots

    @typecheck
    def fit(
        self,
        *,
        dose: Float64Array["intervals"],
        stop: Int64Array["intervals"],
        start: Int64Array["intervals"],
        event: Int64Array["intervals"],
        patient: Int64Array["intervals"],
        covariates: Float64Array["intervals covariates"] | None = None,
        strata: Int64Array["intervals"] | None = None,
        batch: Int64Array["intervals"] | None = None,
        init: Float64Array["fullcovariates"] | None = None,
    ):
        """Fits the WCE model to the data and stores the results as attributes.

        Args:
            dose ((I,) float64 array): drug dose received during each interval.
            stop ((I,) int64 array): end time of each interval.
            start ((I,) int64 array): start time of each interval.
                Currently, only unit-length intervals (stop == start + 1) are supported.
            event ((I,) int64 array): 1 if the interval ends with an event (death),
                0 if it is censored.
            patient ((I,) int64 array): patient id for each interval.
            covariates ((I,C) float64 array, optional): additional (non-WCE)
                covariates. Defaults to None, i.e. only the WCE features are used.
            strata ((I,) int64 array, optional): stratum id for each interval.
                Defaults to a single stratum for all patients.
            batch ((I,) int64 array, optional): batch id for each interval.
                Independent WCE models are fitted for each batch.
            init ((C + n_atoms,) float64 array, optional): initial values for
                the coefficients. Defaults to zeros.

        Results are stored as attributes: knots_, coef_, WCE_coef_,
        risk_function_, std_, SED_, means_, score_, loglik_, loglik_init_,
        sctest_init_, hessian_, imat_, iter_, n_events_, info_criterion_, and
        (if nbootstraps is set) bootstrap_coef_, bootstrap_WCE_coef_,
        bootstrap_risk_functions_.
        """

        if not np.all(stop == start + 1):
            msg = "Currently, we only support unit length intervals."

            raise NotImplementedError(msg)

        # Step 1: compute the time-dependent features (= exposures)
        exposures, knots = self._wce_features(patient=patient, dose=dose, time=stop)
        assert exposures.shape == (len(stop), self.n_atoms)
        exposures = np.array(exposures, dtype=np.float64)

        # Step 2: perform a CoxPH regression with the new covariates
        if covariates is None:
            # No external covariates, just drug doses:
            self.n_covariates = 0
            covariates = exposures
        else:
            # We observe other covariates such as the sex, etc.
            self.n_covariates = covariates.shape[-1]
            covariates = np.concatenate((covariates, exposures), axis=-1)

        # print("\n\nNote: the WCE features are computed on the stop times of the intervals.")

        self.survival_model.fit(
            covariates=covariates,
            start=start,
            stop=stop,
            event=event,
            # patient=patient,
            strata=strata,
            batch=batch,
            init=init,
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

        self.risk_function_ = torch.from_numpy(self.WCE_coef_).to(self.device) @ self.atoms.to(self.dtype).T
        assert self.risk_function_.shape == (n_batch, self.cutoff)

        # Standard deviations for the coefficients:
        self.std_ = self.survival_model.std_[:, : self.n_covariates]
        assert self.std_.shape == (n_batch, self.n_covariates)

        # Standard deviations for the WCE B-Spline weights:
        self.SED_ = self.survival_model.std_[:, self.n_covariates :]
        assert self.SED_.shape == (n_batch, self.n_atoms)

        # Batch coefficients: --------------------------------------------------
        if self.nbootstraps is not None:
            assert self.survival_model.bootstrap_coef_.shape == (
                self.nbootstraps,
                n_batch,
                self.n_covariates + self.n_atoms,
            )

            # Bootstrap coefficients for the covariates:
            self.bootstrap_coef_ = self.survival_model.bootstrap_coef_[
                :, :, : self.n_covariates
            ]
            assert self.bootstrap_coef_.shape == (
                self.nbootstraps,
                n_batch,
                self.n_covariates,
            )

            # Bootstrap weights for the WCE B-Spline features:
            self.bootstrap_WCE_coef_ = self.survival_model.bootstrap_coef_[
                :, :, self.n_covariates :
            ]
            assert self.bootstrap_WCE_coef_.shape == (
                self.nbootstraps,
                n_batch,
                self.n_atoms,
            )


            # Estimated risk function:
            # (nbootstraps, n_batch, n_atoms) @ (n_atoms, cutoff) -> (nbootstraps, n_batch, cutoff)
            self.bootstrap_risk_functions_ = torch.tensor(self.bootstrap_WCE_coef_, dtype=self.dtype).to(self.device) @ self.atoms.to(self.dtype).T
        # Usual CoxPH results: -------------------------------------------------
        self.means_ = self.survival_model.means_
        self.score_ = self.survival_model.score_
        self.sctest_init_ = self.survival_model.sctest_init_
        self.loglik_init_ = self.survival_model.loglik_init_
        self.loglik_ = self.survival_model.loglik_
        self.hessian_ = self.survival_model.hessian_
        self.imat_ = self.survival_model.imat_
        self.iter_ = self.survival_model.iter_
        self.n_events_ = int(np.sum(event))
        # Compute the information criterion for the WCE model, matching the
        # `my_bic_c()` formula from the reference `WCE` R package: AIC uses a
        # penalty of 2 per degree of freedom, BIC uses log(n_events).
        penalty_per_df = 2.0 if self.criterion == "aic" else np.log(self.n_events_)
        self.info_criterion_ = -2 * np.asarray(self.loglik_) + (self.n_atoms + self.n_covariates) * penalty_per_df


    def HR(self,
           vecnum:Int64Array["intervals"],
           vecdenom:Int64Array["intervals"],
           level = 0.95):

        """Computes the Hazard Ratio between two vectors of doses.
        Args:
            vecnum ((Intervals,) int64 array): a vector of doses for the numerator.
            vecdenom ((Intervals,) int64 array): a vector of doses for the denominator.
        Returns:
            (Intervals,) float64 array: the Hazard Ratio between the two vectors of doses.
            """

        cutoff = self.cutoff


        if (len(vecnum) != cutoff) or (len(vecdenom) != cutoff):
            msg = f"vecnum and vecdenom should have length {cutoff}."
            raise ValueError(msg)

        if hasattr(self, "bootstrap_risk_functions_"):
            hr_boot = np.exp(self.bootstrap_risk_functions_.squeeze(1).cpu().numpy() @ (vecnum - vecdenom)).tolist()
            lower = np.quantile(hr_boot, (1-level)/2).tolist()
            upper = np.quantile(hr_boot, 1-(1-level)/2).tolist()
            hr = np.exp(self.risk_function_.reshape(-1).cpu().numpy() @ (vecnum - vecdenom)).tolist()
            return {"HR" : hr, "CI_lower": lower, "CI_upper": upper}

        else:
            hr = np.exp(self.risk_function_.reshape(-1).cpu().numpy() @ (vecnum - vecdenom)).tolist()
            return {"HR" : hr}



def wce_numpy(
    *,
    ids,
    covariates,
    doses,
    events,
    start,
    stop,
    cutoff: Int,
    nknots: Int = 1,
    order: Int = 3,
    constrained: Literal["right", "left"] | None = None,
    knot_placement: Literal["quantile", "uniform"] = "quantile",
    criterion: Literal["aic", "bic"] = "bic",
    strata: Int64Array["intervals"] | None = None,
    batch: Int64Array["intervals"] | None = None,
    init: Float64Array["fullcovariates"] | None = None,
    nbootstraps: Int | None = None,
    batchsize: Int | None = None,
    device: TorchDevice | None = None,
    dtype = np.float64,
    **kwargs,
):
    """Functional interface to WCESurvivalAnalysis: fits a WCE model and returns the results as a dict.

    Builds a CoxPHSurvivalAnalysis from `kwargs`, wraps it in a WCESurvivalAnalysis
    with the given WCE parameters, fits it on the data, and collects the results.

    Args:
        ids ((I,) array): patient id for each interval.
        covariates ((I,C) array or None): additional (non-WCE) covariates.
        doses ((I,) array): drug dose received during each interval.
        events ((I,) array): 1 if the interval ends with an event (death),
            0 if it is censored.
        start ((I,) array): start time of each interval.
        stop ((I,) array): end time of each interval.
        cutoff (int): size of the time window for the risk function.
        nknots (int, optional): number of knots for the B-splines. Defaults to 1.
        order (int, optional): order of the B-splines. Defaults to 3.
        constrained ("left", "right" or None, optional): boundary constraint
            on the B-splines. Defaults to None.
        knot_placement ("quantile" or "uniform", optional): how to position
            the B-spline knots. "uniform" spaces every knot (including the
            boundary padding) equally, which is required for P-splines.
            Defaults to "quantile".
        criterion ("aic" or "bic", optional): information criterion reported
            as "info_criterion" in the output. Defaults to "bic".
        strata ((I,) int64 array, optional): stratum id for each interval.
        batch ((I,) int64 array, optional): batch id for each interval.
        init ((fullcovariates,) float64 array, optional): initial values for
            the coefficients.
        nbootstraps (int, optional): number of bootstrap resamples. 0 is
            treated as None (no bootstrapping).
        batchsize (int, optional): number of bootstrap resamples processed
            at once on the GPU.
        device (optional): device on which to run the computations.
        dtype (optional): np.float32 or np.float64. Defaults to np.float64.
        **kwargs: extra keyword arguments forwarded to CoxPHSurvivalAnalysis.

    Returns:
        dict: with keys "knotsmat", "coef", "std", "WCE_coef", "SED",
            "risk_function", "info_criterion", "means", "score", "sctest_init",
            "loglik_init", "loglik", "hessian", "imat", "iter", and (if
            nbootstraps is set) "bootstrap_coef", "bootstrap_WCE_coef",
            "bootstrap_risk_functions".
    """
    if nbootstraps == 0:
        nbootstraps = None

    surv_model = CoxPHSurvivalAnalysis(**kwargs)
    model = WCESurvivalAnalysis(
        cutoff=cutoff,
        nknots=nknots,
        order=order,
        constrained=constrained,
        knot_placement=knot_placement,
        criterion=criterion,
        survival_model=surv_model,
        nbootstraps=nbootstraps,
        batchsize=batchsize,
        device=device,
        dtype=dtype
    )

    model.fit(
        dose=doses,
        stop=stop,
        start=start,
        event=events,
        patient=ids,
        covariates=covariates,
        strata=strata,
        batch=batch,
        init=init,
    )

    # # Estimate the standard deviations of the coefficients for the covariates:
    # variances = torch.diagonal(result["imat"], dim1=1, dim2=2)
    # stds = torch.sqrt(variances)
    # result["std"] = stds[:, :ncovariates]
    # result["SED"] = stds[:, ncovariates:]



    output = dict(
        knotsmat=model.knots_,
        coef=model.coef_,
        std=model.std_,
        WCE_coef=model.WCE_coef_,
        SED=model.SED_,
        risk_function=model.risk_function_.cpu().numpy(),
        info_criterion=model.info_criterion_,
        means=model.means_,
        score=model.score_,
        sctest_init=model.sctest_init_,
        loglik_init=model.loglik_init_,
        loglik=model.loglik_,
        hessian=model.hessian_,
        imat=model.imat_,
        iter=model.iter_,
    )


    if nbootstraps is not None:
        output.update(
            bootstrap_coef=model.bootstrap_coef_,
            bootstrap_WCE_coef=model.bootstrap_WCE_coef_,
            bootstrap_risk_functions=model.bootstrap_risk_functions_.squeeze(axis=1).cpu().numpy(),
        )


    return output



def wce_R(
    *,
    data,
    ids,
    covars,
    start,
    stop,
    doses,
    events,
    # WCE parameters:
    cutoff,
    nknots=1,
    order=3,
    constrained=None,
    knot_placement="quantile",
    aic=False,
    bootstrap=None,
    # Cox parameters:
    profile=None,
    batchsize=0,
    ties="efron",
    maxiter=20,
    init=None,
    doscale=False,
    strata = None,
    device = None,
    double_precision = True,
):
    """R-facing wrapper around wce_numpy: fits a WCE model from a long-format data.frame.

    Args:
        data (pandas.DataFrame): the long-format dataset.
        ids (str): name of the column with patient ids.
        covars (list[str] or None): names of additional (non-WCE) covariate columns.
        start (str): name of the column with interval start times.
        stop (str): name of the column with interval end times.
        doses (str): name of the column with drug doses.
        events (str): name of the column with the event indicator.
        cutoff (int): size of the time window for the risk function.
        nknots (int, optional): number of knots for the B-splines. Defaults to 1.
        order (int, optional): order of the B-splines. Defaults to 3.
        constrained (str or None, optional): one of "None"/None, "left"/"Left"/"l"/"L"
            or "right"/"Right"/"r"/"R". Defaults to None.
        knot_placement ("quantile" or "uniform", optional): how to position
            the B-spline knots. "uniform" spaces every knot (including the
            boundary padding) equally, which is required for P-splines.
            Defaults to "quantile".
        aic (bool, optional): if True, "info_criterion" in the output is the
            AIC (penalty = 2 per degree of freedom); otherwise it is the BIC
            (penalty = log(n_events) per degree of freedom). Matches the
            `aic` argument of the reference `WCE` R package. Defaults to False.
        bootstrap (int): number of bootstrap resamples (0 disables bootstrapping).
            Must be set to an integer, since it is converted with int(bootstrap).
        profile (str, optional): if set, path where a Chrome trace of the
            computation is exported. Defaults to None.
        batchsize (int, optional): number of bootstrap resamples processed at
            once on the GPU; 0 means "process all at once". Defaults to 0.
        ties ("breslow" or "efron", optional): tie-handling method. Defaults to "efron".
        maxiter (int, optional): maximum number of Newton iterations. Defaults to 20.
        init (array, optional): initial values for the coefficients.
        doscale (bool, optional): whether to rescale the covariates. Defaults to False.
        strata (array, optional): stratum id for each interval.
        device (str or None, optional): "None", "cpu" or "cuda". Defaults to None
            (auto-select).
        double_precision (bool, optional): if True, use float64; otherwise
            float32. Defaults to True.

    Returns:
        dict: with keys "knotsmat", "coef", "std", "WCE_coef", "SED",
            "risk_function", "info_criterion", "means", "score", "sctest_init",
            "loglik_init", "loglik", "hessian", "imat", "iter", and (if
            bootstrap > 0) "bootstrap_coef", "bootstrap_WCE_coef",
            "bootstrap_risk_functions".
    """

    if constrained == "None":
        constrained = None
    elif constrained in ["L", "l", "Left", "left"]:
        constrained = "left"
    elif constrained in ["R", "r", "Right", "right"]:
        constrained = "right"
    else:
        msg = f"constrained should be 'None', 'left', 'Left', 'l', 'L', 'right', 'Right', or 'R'. Received {constrained}."
        raise ValueError(msg)


    if device == "None":
        device = None

    if device not in [None, "cpu", "cuda"]:
        msg = f"device should be None, 'cpu' or 'cuda'. Received {device}."
        raise ValueError(msg)

    if device == "cuda" and not use_cuda:
        msg = "CUDA device requested but no GPU available."
        raise ValueError(msg)



    # if device is not None:
    #     if device == "cpu":
    #         device = torch.device("cpu")
    #     elif device == "cuda":
    #         device = torch.device("cuda")
    #     else:
    #         msg = f"device should be 'cpu' or 'cuda'. Received {device}."
    #         raise ValueError(msg)
    #     torch.cuda.set_device(device)

    if device == torch.device("cuda") and not use_cuda:
        msg = "CUDA device requested but no GPU available."
        raise ValueError(msg)

    dtype = np.float64 if double_precision else np.float32



    ids = np.array(data[ids], dtype = np.int64)
    doses = np.array(data[doses], dtype = np.float64)
    start = np.array(data[start], dtype = np.int64)
    stop = np.array(data[stop], dtype = np.int64)
    events = np.array(data[events], dtype = np.int64)
    N = len(stop)





    #TODO type of covars, test must be OK for None, in first part will ignore it
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
            start=start,
            stop=stop,
            cutoff=int(cutoff),
            nknots=int(nknots),
            order=int(order),
            constrained=constrained,
            knot_placement=knot_placement,
            criterion="aic" if aic else "bic",
            strata=strata,
            batch=None,
            init=init,
            nbootstraps=int(bootstrap),
            batchsize=int(batchsize) if batchsize > 0 else None,
            device=device,
            maxiter=int(maxiter),
            ties=ties,
            doscale=doscale,
            dtype=dtype,
        )

    if profile is not None:
        prof.export_chrome_trace(profile)

    return res
