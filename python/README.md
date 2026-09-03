# survivalGPU


**GPU-accelerated survival analysis** — Cox Proportional Hazards (CoxPH) and
Weighted Cumulative Exposure (WCE) models, built on [PyTorch](https://pytorch.org),
with the WCE model additionally relying on [KeOps](https://www.kernel-operations.io).
survivalGPU scales classical survival models to large datasets and to heavy
bootstrap resampling by running the core computations on the GPU.


## Features

- **Cox Proportional Hazards** models (`CoxPHSurvivalAnalysis`, `coxph_numpy`),
  with Breslow and Efron handling of ties. Only requires PyTorch — works on
  Windows.
- **Weighted Cumulative Exposure (WCE)** models (`WCESurvivalAnalysis`,
  `wce_numpy`) for time-varying exposure effects. Requires the optional
  `pykeops` dependency, which is **not available on Windows** (see below).
- **GPU acceleration** of the likelihood and its gradients via PyTorch (+ KeOps
  for WCE), with a CPU fallback.
- **Bootstrap** resampling and **dataset simulation** utilities for reproducible
  experiments.

## Installation

```bash
pip install survivalgpu[wce]     # CoxPH + WCE (recommended; requires pykeops, not on Windows)
pip install survivalgpu          # CoxPH only, no pykeops (e.g. on Windows)
```

### Requirements

- **Python >= 3.10**
- **A C++ compiler (WCE only):**  [`pykeops`](https://www.kernel-operations.io)
  just-in-time compiles C++/CUDA kernels at runtime, so a working C++ toolchain
  must be present. For GPU acceleration you also need the **CUDA toolkit**
  (`nvcc`) installed, not just a CUDA-capable GPU. The code runs on CPU without
  a GPU — the GPU is simply where the speedups come from. The CoxPH model
  doesn't need any of this.

### macOS (Apple Silicon)

`pykeops` needs [OpenMP](https://www.openmp.org), which isn't bundled with Apple's
compiler toolchain on Apple Silicon (M1/M2/M3/M4). Without it, `pykeops` disables
OpenMP and falls back to a much less-tested code path — we've seen this cause
crashes. Install it via Homebrew before setting up your environment:

```bash
brew install libomp
```

### Windows

`survivalgpu` (CoxPH) installs and runs natively on Windows. **pykeops**, needed
only for the WCE model, compiles C++/CUDA kernels at runtime and is **not
supported natively on Windows** — calling `WCESurvivalAnalysis`/`wce_numpy`
without it raises a clear `ImportError`. If you need WCE on Windows, you have
three working options:

- **WSL2:** Install a Linux distribution through the
  [Windows Subsystem for Linux](https://learn.microsoft.com/windows/wsl/install),
  then run `pip install survivalgpu[wce]` inside it exactly as you would on Linux.
- **Docker:** KeOps publishes a reference container with a full CUDA + PyTorch +
  KeOps stack — and it is already configured with survivalGPU on its `PYTHONPATH`,
  so it is ready to run this package. See the
  [KeOps Dockerfile](https://github.com/getkeops/keops/blob/main/Dockerfile)
  and the
  [KeOps installation guide](https://www.kernel-operations.io/keops/python/installation.html).

## Quick start

survivalGPU exposes both a scikit-learn-style class interface and a NumPy
functional interface:

```python
import numpy as np
from survivalgpu import CoxPHSurvivalAnalysis

# Three (start, stop] intervals, one covariate:
stop = np.array([1, 1, 2], dtype=np.int64)
event = np.array([0, 1, 1], dtype=np.int64)
covariates = np.array([[1.0], [0.0], [4.0]])

model = CoxPHSurvivalAnalysis(ties="efron")
model.fit(covariates, stop, event=event)

print(model.coef_)
```

Full, runnable examples live in the
[repository](https://github.com/jeanfeydy/survivalGPU), and the
[test suite](https://github.com/jeanfeydy/survivalGPU/tree/main/python/tests)
doubles as a usage reference until the documentation is complete.

## Development

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/jeanfeydy/survivalGPU.git
cd survivalGPU

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
```

The test suite (`pip` >= 25.1 required) can then be run in two modes:

- **Full suite (CoxPH + WCE)** — requires `pykeops` (not available on Windows):

  ```bash
  pip install -e ".[wce]" --group test
  pytest
  ```

- **CoxPH-only** — no `pykeops` required, works on Windows:

  ```bash
  pip install -e . --group test
  pytest
  ```

  In this mode, tests marked `needs_keops` (the WCE-specific tests) are
  automatically skipped instead of failing, and a warning banner is
  printed at the end of the run listing what was skipped and how to get
  full coverage (`pip install survivalgpu[wce]`).
  [`python/tests/test_optional_keops.py`](https://github.com/jeanfeydy/survivalGPU/blob/main/python/tests/test_optional_keops.py)
  is a dedicated regression test that always runs, in both modes, and
  asserts on this exact contract: `CoxPHSurvivalAnalysis` keeps working,
  and `WCESurvivalAnalysis`/`wce_numpy` raise a clear `ImportError` when
  pykeops is unavailable.

Both modes are exercised in CI (see
[python-tests.yml](https://github.com/jeanfeydy/survivalGPU/blob/main/.github/workflows/python-tests.yml)),
the no-`pykeops` mode across Linux, macOS and Windows.

## Citation

If you use survivalGPU in your research, please cite it.

```bibtex
@software{survivalgpu,
  author = {Jean Feydy, Antoine Poirot-Bourdain, Alexis van Straaten},
  title  = {{survivalGPU}: GPU-accelerated survival analysis},
  url    = {https://github.com/jeanfeydy/survivalGPU},
  year   = {2026},
}
```

## License

Distributed under the terms of the **LGPL-2.1-or-later** license. See
[LICENSE](https://github.com/jeanfeydy/survivalGPU/blob/main/LICENSE).
