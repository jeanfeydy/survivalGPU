from beartype import beartype as typechecker
from jaxtyping import Int64, Float64, jaxtyped
from typing import Optional


import torch
import numpy as np


def typecheck(func):
    return jaxtyped(typechecker(func))


Array = np.ndarray


class Int64Array:
    def __class_getitem__(cls, shape):
        return Int64[Array, shape]


class Float64Array:
    def __class_getitem__(cls, shape):
        return Float64[Array, shape]
