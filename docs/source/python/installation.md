# Installation

```bash
pip install survivalgpu[wce]     # CoxPH + WCE (recommended; requires pykeops, not on Windows)
pip install survivalgpu          # CoxPH only, no pykeops (e.g. on Windows)
```

## Requirements

- **Python >= 3.10**
- **A C++ compiler (WCE only):** [`pykeops`](https://www.kernel-operations.io)
  just-in-time compiles C++/CUDA kernels at runtime, so a working C++ toolchain
  must be present. For GPU acceleration you also need the **CUDA toolkit**
  (`nvcc`) installed, not just a CUDA-capable GPU. The code runs on CPU without
  a GPU — the GPU is simply where the speedups come from. The CoxPH model
  doesn't need any of this.

## macOS (Apple Silicon)

`pykeops` needs [OpenMP](https://www.openmp.org), which isn't bundled with
Apple's compiler toolchain on Apple Silicon (M1/M2/M3/M4). Without it,
`pykeops` disables OpenMP and falls back to a much less-tested code path —
this has been observed to cause crashes. Install it via Homebrew before
setting up your environment:

```bash
brew install libomp
```

## Windows

`survivalgpu` (CoxPH) installs and runs natively on Windows. **pykeops**,
needed only for the WCE model, compiles C++/CUDA kernels at runtime and is
**not supported natively on Windows** — calling
`WCESurvivalAnalysis`/`wce_numpy` without it raises a clear `ImportError`. If
you need WCE on Windows, you have two working options:

- **WSL2:** Install a Linux distribution through the
  [Windows Subsystem for Linux](https://learn.microsoft.com/windows/wsl/install),
  then run `pip install survivalgpu[wce]` inside it exactly as you would on
  Linux.
- **Docker:** KeOps publishes a reference container with a full CUDA +
  PyTorch + KeOps stack. See the
  [KeOps Dockerfile](https://github.com/getkeops/keops/blob/main/Dockerfile)
  and the
  [KeOps installation guide](https://www.kernel-operations.io/keops/python/installation.html).
