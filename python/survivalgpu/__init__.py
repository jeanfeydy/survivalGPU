import torch

# On Ampere+ GPUs, the default behaviour of PyTorch is to sacrifice
# precision for speed using tensor cores (with typical errors ~0.1%).
# This doesn't make sense for CoxPH computations, so we come
# back to a standard implementation of the matrix multiplication:
torch.backends.cuda.matmul.allow_tf32 = False

from .coxph import coxph_torch, coxph_numpy, coxph_R
from .wce import wce_torch, wce_numpy, wce_R
from . import utils, optimizers
from .utils import use_cuda, device, float32, int32, int64

__all__ = sorted(
    [
        "coxph_torch",
        "coxph_numpy",
        "coxph_R",
        "wce_torch",
        "wce_numpy",
        "wce_R",
    ]
)
