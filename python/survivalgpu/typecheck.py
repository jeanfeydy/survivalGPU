from beartype import beartype as typechecker
import jaxtyping as jxt
from typing import Optional, Callable, Union


import torch
import numpy as np


def typecheck(func):
    return jxt.jaxtyped(typechecker(func))


Array = np.ndarray
Int = Union[int, np.integer]
Int64 = np.int64
Float64 = np.float64


class UInt8Array:
    def __class_getitem__(cls, shape):
        return jxt.UInt8[Array, shape]


class Int64Array:
    def __class_getitem__(cls, shape):
        return jxt.Int64[Array, shape]


class Float64Array:
    def __class_getitem__(cls, shape):
        return jxt.Float64[Array, shape]
