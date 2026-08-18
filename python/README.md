# survivalGPU


**GPU-accelerated survival analysis** — Cox Proportional Hazards (CoxPH) and
Weighted Cumulative Exposure (WCE) models, built on [PyTorch](https://pytorch.org)
and [KeOps](https://www.kernel-operations.io). survivalGPU scales classical
survival models to large datasets and to heavy bootstrap resampling by running the
core computations on the GPU.


## Features

- **Cox Proportional Hazards** models (`CoxPHSurvivalAnalysis`, `coxph_numpy`),
  with Breslow and Efron handling of ties.
- **Weighted Cumulative Exposure (WCE)** models (`WCESurvivalAnalysis`,
  `wce_numpy`) for time-varying exposure effects.
- **GPU acceleration** of the likelihood and its gradients via PyTorch + KeOps,
  with a CPU fallback.
- **Bootstrap** resampling and **dataset simulation** utilities for reproducible
  experiments.

## Installation

```bash
pip install survivalgpu
```

### Requirements

- **Python >= 3.10**
- **A C++ compiler:**  [`pykeops`](https://www.kernel-operations.io) just-in-time
  compiles C++/CUDA kernels at runtime, so a working C++ toolchain must be present.
  For GPU acceleration you also need the **CUDA toolkit** (`nvcc`) installed, not
  just a CUDA-capable GPU. The code runs on CPU without a GPU — the GPU is simply
  where the speedups come from.

### macOS (Apple Silicon)

`pykeops` needs [OpenMP](https://www.openmp.org), which isn't bundled with Apple's
compiler toolchain on Apple Silicon (M1/M2/M3/M4). Without it, `pykeops` disables
OpenMP and falls back to a much less-tested code path — we've seen this cause
crashes. Install it via Homebrew before setting up your environment:

```bash
brew install libomp
```

### Windows

**pykeops** compiles C++/CUDA kernels at runtime and is **not supported natively on
Windows**. Windows users have three working options:

- **WSL2:** Install a Linux distribution through the
  [Windows Subsystem for Linux](https://learn.microsoft.com/windows/wsl/install),
  then run `pip install survivalgpu` inside it exactly as you would on Linux.
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

Clone the repository and install in editable mode with the test dependency group
(requires `pip` >= 25.1):

```bash
git clone https://github.com/jeanfeydy/survivalGPU.git
cd survivalGPU

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
pip install -e . --group test
```

Run the test suite with:

```bash
pytest
```

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
