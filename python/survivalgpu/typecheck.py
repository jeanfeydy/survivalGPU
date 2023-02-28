from beartype import beartype as typechecker
import jaxtyping as jxt
from typing import Optional, Callable, Union


import torch
import numpy as np


def typecheck(func):
    return jxt.jaxtyped(typechecker(func))


Array = np.ndarray
Tensor = torch.Tensor

Bool = Union[bool, np.bool_]
Int = Union[int, np.integer]
Float = Union[float, np.floating]
Real = Union[Int, Float]
Int64 = np.int64
Float64 = np.float64


class UInt8Array:
    """Numpy Array of uint8."""

    def __class_getitem__(cls, shape):
        return jxt.UInt8[Array, shape]


class Int64Array:
    """Numpy Array of int64 ("long")."""

    def __class_getitem__(cls, shape):
        return jxt.Int64[Array, shape]


class Float64Array:
    """Numpy Array of float64 ("double")."""

    def __class_getitem__(cls, shape):
        return jxt.Float64[Array, shape]


class Int64Tensor:
    """Torch tensor of int64 ("long")."""

    def __class_getitem__(cls, shape):
        return jxt.Int64[Tensor, shape]


class Float32Tensor:
    """Torch tensor of float32 ("float")."""

    def __class_getitem__(cls, shape):
        return jxt.Float32[Tensor, shape]
