import torch

from . import (
    _env_setup,  # -- must run first: sets CPATH for PyKeOps
    optimizers,
    utils,
)
from .coxph import CoxPHSurvivalAnalysis, coxph_numpy, coxph_R
from .simulations import (
    ConstantCovariate,
    TimeDependentCovariate,
    WCECovariate,
    simulate_dataset,
    simulate_dataset_batch,
    simulate_for_experiment,
)
from .utils import float32, int32, int64, use_cuda

# WCE relies on PyKeOps, which is an optional dependency (not available on
# Windows). Importing survivalgpu (and using the CoxPH model) must succeed
# even if pykeops is missing; only actually calling into WCE should fail,
# with a clear, actionable error message.
try:
    from .wce import WCESurvivalAnalysis, wce_numpy, wce_R
except ImportError as _wce_import_error:

    def _wce_unavailable(*_args, **_kwargs):
        msg = (
            "The WCE model requires the optional 'pykeops' dependency, "
            "which is not installed (PyKeOps is not available on Windows). "
            "Install it with `pip install survivalgpu[wce]` on Linux/macOS "
            "to use WCESurvivalAnalysis. The CoxPH model does not require "
            "pykeops and remains usable."
        )
        raise ImportError(msg) from _wce_import_error

    class WCESurvivalAnalysis:
        def __init__(self, *_args, **_kwargs):
            _wce_unavailable()

    def wce_numpy(*_args, **_kwargs):
        _wce_unavailable()

    def wce_R(*_args, **_kwargs):
        _wce_unavailable()

# On Ampere+ GPUs, the default behaviour of PyTorch is to sacrifice
# precision for speed using tensor cores (with typical errors ~0.1%).
# This doesn't make sense for CoxPH computations, so we come
# back to a standard implementation of the matrix multiplication:
torch.backends.cuda.matmul.allow_tf32 = False

__all__ = [
        "coxph_numpy",
        "coxph_R",
        "wce_numpy",
        "wce_R",
        "CoxPHSurvivalAnalysis",
        "WCESurvivalAnalysis",
        "simulate_for_experiment",
        "simulate_dataset",
        "WCECovariate",
        "ConstantCovariate",
        "TimeDependentCovariate",
        "permalgo",
        "simulate_dataset_batch"
    ]

__version__ = "0.1.0"

# Ties in the CoxPH model:
SUPPORTED_TIES = ["breslow", "efron"]

# Warm up the GPU:
if use_cuda:
    _ = torch.zeros(1, device="cuda" if use_cuda else "cpu")
