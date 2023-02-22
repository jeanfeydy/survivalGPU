from beartype import beartype as typechecker
from jaxtyping import Float, jaxtyped
from beartype.vale import Is
from typing import Annotated  # <--------------- if Python ≥ 3.9.0

# from typing_extensions import Annotated   # <-- if Python < 3.9.0

import torch
import numpy as np


def typecheck(func):
    return jaxtyped(typechecker(func))


# PEP-compliant type hint matching only a floating-point PyTorch tensor.
TorchTensorFloat = Annotated[tensor, Is[lambda tens: tens.type() is torch_float]]

# PEP-compliant type hint matching only an integral PyTorch tensor.
TorchTensorInt = Annotated[tensor, Is[lambda tens: tens.type() is torch_int]]
