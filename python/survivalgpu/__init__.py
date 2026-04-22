import torch

from . import optimizers, utils
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
from .wce import WCESurvivalAnalysis, wce_numpy, wce_R

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
