"""Implements our main API for the Cox Proportional Hazards models.

We provide:

- CoxPHSurvivalAnalysis, a scikit-learn-like estimator.
- coxph_R, a functional wrapper around CoxPHSurvivalAnalysis that is used
  by our R survivalGPU package via reticulate.

"""


# Use NumPy for basic array manipulation:
import numpy as np

# Use PyTorch for fast array manipulations (on the GPU):
import torch

# We use functools.partial
import functools

# The convex CoxPH objective:
from .coxph_likelihood import coxph_objective
from .bootstrap import Resampling


# Convex optimizer for the CoxPH objective:
from .optimizers import newton

from .utils import numpy
from .utils import use_cuda, float32, int32, int64
from .utils import device as default_device

from .typecheck import typecheck, Optional, Literal
from .typecheck import Int, Real, Bool
from .typecheck import Int64Array, Float64Array
from .typecheck import Float32Tensor
from .typecheck import TorchDevice

from .datasets import SurvivalDataset
from .torch_datasets import TorchSurvivalDataset


def numpy(x):
    return x.detach().cpu().numpy()


class CoxPHSurvivalAnalysis:
    """Cox Proportional Hazards model.

    The API of this object is loosely based on scikit-survival.

    Args:
        alpha (float): L2 regularization parameter.
        ties (str): Ties handling method. One of "efron", "breslow".
        backend (str): Backend to use. One of "numpy", "csr", "torch".
        bootstrap (int): Number of bootstrap samples to use.
        batchsize (int): Number of bootstrap samples to process in parallel.
        maxiter (int): Maximum number of Newton iterations.
        eps (float): Convergence tolerance.
        doscale (bool): Whether to scale the covariates for stability.
        verbosity (int): Verbosity level.
        mode (str or None): One of "unit length", "start zero", "any".
            Assumptions made on the (start, stop] intervals.
            Defaults to None, which selects automatically the fastest backend.
    """

    @typecheck
    def __init__(
        self,
        alpha: Real = 0.0,
        ties: Literal["efron", "breslow"] = "efron",
        backend: Literal["torch", "pyg", "coo", "csr"] = "csr",
        maxiter: Int = 20,
        eps: Real = 1e-5,
        doscale: Bool = False,
        verbosity: Int = 0,
        mode: Optional[Literal["unit length", "start zero", "any"]] = None,
    ):
        self.alpha = alpha
        self.ties = ties
        self.backend = backend
        self.maxiter = maxiter
        self.eps = eps
        self.doscale = doscale
        self.verbosity = verbosity
        self.mode = mode

    @typecheck
    def fit(
        self,
        covariates: Float64Array["intervals covariates"],
        stop: Int64Array["intervals"],
        *,
        start: Optional[Int64Array["intervals"]] = None,
        event: Optional[Int64Array["intervals"]] = None,
        patient: Optional[Int64Array["intervals"]] = None,
        strata: Optional[Int64Array["patients"]] = None,
        batch: Optional[Int64Array["patients"]] = None,
        init: Optional[Float64Array["covariates"]] = None,
        n_bootstraps: Optional[Int] = None,
        batch_size: Optional[Int] = None,
        device: Optional[TorchDevice] = None,
    ):
        """Fit the model.

        Args:
            X (array-like): Covariates.
            y (array-like): Survival times and event indicators.
            sample_weight (array-like): Sample weights.
        """
        # Pre-process the input data: ----------------------------------------------------
        # Create a dataset object: this enforces checks on the input data
        dataset = SurvivalDataset(
            covariates=covariates,
            stop=stop,
            start=start,
            event=event,
            patient=patient,
            strata=strata,
            batch=batch,
        )
        # Re-encode the data arrays as PyTorch tensors on the correct device,
        # with the correct dtype (float64 -> float32)
        if device is None:
            device = default_device
        dataset = dataset.to_torch(device)

        # Re-order the input arrays by lexicographical order on (batch, strata, stop, event):
        dataset.sort()

        # Scale the covariates for the sake of numerical stability:
        means, scales = dataset.scale(rescale=self.doscale)

        # Count the number of death times:
        dataset.count_deaths()

        # Filter out the times that have no impact on the CoxPH model
        # (e.g. censoring that occurs before the first death):
        dataset.prune(mode=self.mode)

        n_batch, n_covariates = dataset.n_batch, dataset.n_covariates

        # Choose the fastest implementation of the CoxPH objective, ----------------------
        # i.e. the partial neg-log-likelihood of the CoxPH model.

        if self.mode is None:
            # Case 1: all the intervals are )t-1, t] (i.e. no-interval mode),
            # we can group times using equality conditions on the stop times.
            # This is typically the case when using time-dependent covariates as in the WCE model.
            if torch.all(dataset.stop == dataset.start + 1):
                mode = "unit length"

            # Case 2: all the intervals are )0, t]:
            # this opens the door to a more efficient implementation using a cumulative hazard.
            elif torch.all(dataset.start == 0):
                mode = "start zero"

            # Case 3: general case )start, stop], we use two cumulative hazards:
            else:
                raise NotImplementedError(
                    "Currently, general intervals are not supported."
                )
                mode = "any"

        else:
            mode = self.mode

        objective = functools.partial(
            coxph_objective,
            dataset=dataset,
            ties=self.ties,
            backend=self.backend,
            mode=mode,
        )

        # Define the loss function:
        def loss(*, bootstrap):
            """Our loss function, including the L2 regularization term."""
            obj = objective(bootstrap=bootstrap)

            def aux(coef):
                scores = self._linear_risk_scores(
                    coef=coef, dataset=dataset, bootstrap=bootstrap
                )
                if scales is None:
                    scaled_coef = coef
                else:
                    assert scales.shape == (coef.shape[-1],)
                    scaled_coef = coef * scales
                reg = self.alpha * (scaled_coef**2).sum(dim=1)
                return obj(scores) + reg

            return aux

        # Run the Newton optimizer: ------------------------------------------------------

        # Vector of inital values of the Newton iteration.
        # Zero for all variables by default.
        if init is None:
            init_tensor = torch.zeros(
                (n_batch, n_covariates), dtype=float32, device=device
            )

        else:
            init_tensor = torch.tensor(init, dtype=float32, device=device)
            assert init_tensor.shape == (n_covariates,)
            init_tensor = init_tensor.repeat(n_batch, 1)

        res = newton(
            loss=loss(bootstrap=dataset.original_sample()),
            start=init_tensor,
            maxiter=self.maxiter,
            eps=self.eps,
            verbosity=self.verbosity,
        )

        # Extract the results: -----------------------------------------------------------

        # Coef is the truly important result:
        self.coef_ = res.x

        # We also include information to ensure compatibility with the R survival package:
        self.means_ = means
        # Gradient of the log-likehood at the optimum:
        self.score_ = -res.jac
        # Score test statistics (= dot(step, gradient)) at iteration 0:
        self.sctest_init_ = res.score_test_init
        # Log-likelihood at iteration 0:
        self.loglik_init_ = -res.fun_init
        # Log-likelihood at the optimum:
        self.loglik_ = -res.fun
        # Hessian of the log-likelihood at the optimum:
        self.hessian_ = res.hess
        # Inverse of the Hessian at the optimum:
        self.imat_ = res.imat
        # Estimate for the standard errors of the coefficients:
        self.std_ = res.std
        # Number of Newton iteration
        self.iter_ = res.iterations

        # If required, compute a distribution of the coefficients using bootstrap: -------
        if n_bootstraps is not None:
            bootstrap_coef = []
            for bootstrap in dataset.bootstraps(
                n_bootstraps=n_bootstraps, batch_size=batch_size
            ):
                # Vector of inital values of the Newton iteration.
                # Zero for all variables by default.
                if init is None:
                    init_tensor = torch.zeros(
                        (len(bootstrap) * n_batch, n_covariates),
                        dtype=float32,
                        device=device,
                    )
                else:
                    init_tensor = torch.tensor(init, dtype=float32, device=device)
                    assert init_tensor.shape == (n_covariates,)
                    init_tensor = init_tensor.repeat(len(bootstrap) * n_batch, 1)

                res = newton(
                    loss=loss(bootstrap=bootstrap),
                    start=init_tensor,
                    maxiter=self.maxiter,
                    eps=self.eps,
                    verbosity=self.verbosity,
                )
                bootstrap_coef.append(res.x)

            self.bootstrap_coef_ = torch.stack(bootstrap_coef).view(
                n_bootstraps, n_batch, n_covariates
            )

        # If the covariates have been normalized for the sake of stability,
        # we shouldn't forget to "de-normalize" the results:
        self._rescale(scales=scales, n_covariates=n_covariates)

        # And convert all the attributes to NumPy float64 arrays:
        self.to_numpy()

        # Finally, check the shapes of the results: --------------------------------------
        loglik_shape = (n_batch,)
        coef_shape = (n_batch, n_covariates)
        hessian_shape = (n_batch, n_covariates, n_covariates)

        assert self.means_.shape == (n_covariates,)  # TODO batch this...: coef_shape
        assert self.coef_.shape == coef_shape
        assert self.std_.shape == coef_shape
        assert self.score_.shape == coef_shape

        assert self.sctest_init_.shape == loglik_shape
        assert self.loglik_init_.shape == loglik_shape
        assert self.loglik_.shape == loglik_shape

        assert self.hessian_.shape == hessian_shape
        assert self.imat_.shape == hessian_shape

        if n_bootstraps is not None:
            assert self.bootstrap_coef_.shape == (n_bootstraps, n_batch, n_covariates)

    @typecheck
    def _linear_risk_scores(
        self,
        *,
        coef: Float32Tensor["batches covariates"],
        dataset: TorchSurvivalDataset,
        bootstrap: Resampling,
    ) -> Float32Tensor["bootstraps intervals"]:
        """Standard function to compute risks in the CoxPH model: dot(beta, x[i]).

        TODO: clean docstring below
        - If `batch_size == dataset.n_batch`, the vector of coefficients `coef[i]` will be
        associated to the subset of patients such that `dataset.batch == i`.
        - If `batch_size == len(bootstrap) * dataset.n_batch`, the vector of coefficients
        `coef[i]` will be associated to the subset of patients such that
        `dataset.batch == i % dataset.n_batch`.
        In other words, the `(batch_size, covariates)` Tensor of coefficients `coef`
        is interpreted as a `(n_bootstraps, dataset.n_batch, covariates)` Tensor.
        """

        assert coef.shape[0] == len(bootstrap) * dataset.n_batch

        # coef is (n_bootstraps, n_batch, covariates):
        coef = coef.view(len(bootstrap), dataset.n_batch, -1)

        # scattered_coef is (n_bootstraps, n_intervals, covariates):

        if True and dataset.n_batch == 1:
            # Simple case with no batch - don't waste time with indexing operations:
            scattered_coef = coef.view(len(bootstrap), 1, -1)

        else:
            # N.B.: Naive implementation with an indexing operation as in
            #
            if False:
                scattered_coef = coef[:, dataset.batch_intervals, :]  # (B, I, D)
            #
            # is MASSIVELY inefficient in the backward pass, as discussed in
            # https://github.com/pytorch/pytorch/issues/41162
            # https://github.com/dmlc/dgl/issues/3729
            #
            # Instead, we prefer the following line, with a non-deterministic backward pass:
            else:
                scattered_coef = torch.index_select(coef, 1, dataset.batch_intervals)

            assert scattered_coef.shape == (
                len(bootstrap),
                dataset.n_intervals,
                dataset.n_covariates,
            )

        X = dataset.covariates  # (I, D)
        # (B, I, D) * (1, I, D) -> (B, I, D)
        scores = scattered_coef * X.view(1, dataset.n_intervals, dataset.n_covariates)
        scores = scores.sum(-1)  # (B, I, D) -> (B, I)
        assert scores.shape == (len(bootstrap), dataset.n_intervals)
        return scores

    @typecheck
    def _rescale(
        self,
        *,
        scales: Optional[Float32Tensor["covariates"]],
        n_covariates: int,
    ):
        """Restores proper scaling for the parameters of the CoxPH model.

        For the sake of numerical stability (and backward compatibility with the R
        survival package), if doscale is True, we normalize the covariates with:

        cov[:,i] *= scales[i]

        where

        scales[i] = 1 / mean(abs(cov[:,i])).

        This induces a scaling of the model parameters that we "undo" with the code below.
        """

        # If the covariates have been normalized for the sake of stability,
        # we shouldn't forget to "de-normalize" the results:

        if scales is not None:
            assert scales.shape == (n_covariates,)

            # (B,D) * (D,) = (B,D)
            self.coef_ = self.coef_ * scales
            self.std_ = self.std_ * scales
            self.score_ = self.score_ / scales

            # (B,D,D) * (D,D) = (B,D,D)
            scales_2 = scales.view(n_covariates, 1) * scales.view(1, n_covariates)
            self.hessian_ = self.hessian_ / scales_2
            self.imat_ = self.imat_ * scales_2

    @typecheck
    def to_numpy(self):
        self.means_ = numpy(self.means_)
        self.coef_ = numpy(self.coef_)
        self.std_ = numpy(self.std_)
        self.score_ = numpy(self.score_)

        self.sctest_init_ = numpy(self.sctest_init_)
        self.loglik_init_ = numpy(self.loglik_init_)
        self.loglik_ = numpy(self.loglik_)

        self.hessian_ = numpy(self.hessian_)
        self.imat_ = numpy(self.imat_)

        if hasattr(self, "bootstrap_coef_"):
            self.bootstrap_coef_ = numpy(self.bootstrap_coef_)


# Functional Numpy API, called by our R wrapper:
def coxph_numpy(
    *,
    x,
    times,
    deaths,
    bootstrap=1,
    batchsize=None,
    ties="efron",
    survtype,
    strata=None,
    backend="csr",
    maxiter=20,
    init=None,
    eps=1e-9,
    alpha=0,
    verbosity=0,
    doscale=False,
):
    """Implements the Cox Proportional Hazards model.

    Args:
        x ((N,D) float32 tensor): the input features.
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
    model = CoxPHSurvivalAnalysis(
        alpha=alpha,
        ties=ties,
        backend=backend,
        maxiter=maxiter,
        eps=eps,
        doscale=doscale,
        verbosity=verbosity,
    )

    # Configure 'start' according to survtype ('counting' or 'right')
    if survtype == "counting":
        start = times - 1
    else:
        start = None

    model.fit(
        covariates=x,
        stop=times,
        start=start,
        event=deaths,
        strata=strata,
        n_bootstraps=bootstrap,
        batch_size=batchsize,
        init=init,
    )

    # Step 5: turn the list of dicts into a dicts of concatenated results ==========
    output = {
        "means": model.means_,
        "coef": model.coef_,
        "std": model.std_,
        "loglik init": model.loglik_init_,
        "loglik": model.loglik_,
        "sctest_init": model.sctest_init_,
        "u": model.score_,
        "imat": model.imat_,
        "hessian": model.hessian_,
        "iter": model.iter_,
    }

    if hasattr(model, "bootstrap_coef_"):
        output["bootstrap coef"] = model.bootstrap_coef_

    return output


# Python >= 3.7:
from contextlib import nullcontext


def coxph_R(
    data,
    stop,
    death,
    covars,
    survtype,
    bootstrap=1,
    batchsize=0,
    ties="efron",
    strata=None,
    maxiter=20,
    init=None,
    doscale=False,
    profile=None,
):
    if ties == "efron":
        # Raise Warning and change to "breslow"
        import warnings

        warnings.warn(
            "Efron ties are not yet supported in our new implementation. "
            "Switching to the 'breslow' approximation."
        )
        ties = "breslow"

    if profile is not None:
        print("Profile trace:", profile)
        print("use_cuda:", use_cuda)
        myprof = torch.autograd.profiler.profile(use_cuda=use_cuda)
    else:
        myprof = nullcontext()

    if strata is not None:
        strata = np.array(strata, dtype=np.int64)

    with myprof as prof:
        times = np.array(data[stop], dtype=np.int64)
        deaths = np.array(data[death], dtype=np.int64)
        N = len(times)

        assert times.dtype == np.int64
        assert deaths.dtype == np.int64

        cov = [data[covar] for covar in covars]
        x = np.array(cov).T.reshape([N, len(cov)])

        res = coxph_numpy(
            x=x,
            times=times,
            deaths=deaths,
            ties=ties,
            survtype=survtype,
            strata=strata,
            backend="csr",
            bootstrap=int(bootstrap),
            batchsize=int(batchsize) if batchsize > 0 else None,
            maxiter=int(maxiter) if profile is None else 1,
            init=init,
            verbosity=0,
            alpha=0.0,
            doscale=doscale,
        )

    if profile is not None:
        prof.export_chrome_trace(profile)

    return res
