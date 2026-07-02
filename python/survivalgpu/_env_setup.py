# python/survivalgpu/_env_setup.py
"""
Ensure PyKeOps can find pybind11's and Python's own C headers when
JIT-compiling kernels.

PyKeOps shells out to a subprocess to locate both pybind11's include path
and Python's own development headers (Python.h) when compiling formulas.
On some environments (notably CI with multiple Python installs present)
this resolves to the wrong interpreter and the compile step fails with
`pybind11/pybind11.h` or `Python.h: No such file or directory`.
Setting CPATH explicitly sidesteps that unreliable detection.
See: https://github.com/getkeops/keops/issues/219
"""

import os
import sysconfig

import pybind11

_include_dirs = [
    pybind11.get_include(),
    sysconfig.get_paths()["include"],
]

_existing = os.environ.get("CPATH", "")
os.environ["CPATH"] = os.pathsep.join(
    _include_dirs + ([_existing] if _existing else [])
)
