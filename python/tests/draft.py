# Import both the annotation and the `jaxtyped` decorator from `jaxtyping`
from jaxtyping import Float, jaxtyped
import numpy as np
import torch
from typing import Union
from numpy.typing import ArrayLike

Array = torch.Tensor  # Union[np.ndarray, torch.Tensor]

# Use your favourite typechecker: usually one of the two lines below.
from typeguard import typechecked as typechecker
from beartype import beartype as typechecker


def typecheck(func):
    return jaxtyped(typechecker(func))


# Write your function. @jaxtyped must be applied above @typechecker!
@typecheck
def batch_outer_product(
    x: Float[Array, "b c1"], y: Float[Array, "b c2"]
) -> Float[Array, "b c1 c2"]:
    return x[:, :, None] * y[0, None, :]


# Call your function with the correct types.
batch_outer_product(torch.ones((2, 3)), torch.ones((2, 4)))
