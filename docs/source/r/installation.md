# Installation

survivalGPU is not on CRAN yet: you install it from GitHub, together with
the Python libraries it uses behind the scenes (via the
[reticulate](https://rstudio.github.io/reticulate/) R package).

## 1. Before you start

You need **R >= 4.1**, **Python >= 3.10**, and a C/C++ compiler to build
the package:

| System | Install first |
|---|---|
| **Linux** | A C/C++ compiler, usually already there. If not, on Debian/Ubuntu: `sudo apt install build-essential` |
| **macOS** | Apple's compiler: run `xcode-select --install` in a terminal.<br>For the WCE model, also run `brew install libomp` (needs [Homebrew](https://brew.sh)). |
| **Windows** | [Rtools](https://cran.r-project.org/bin/windows/Rtools/), matching your R version. |

If you don't have Python >= 3.10, reticulate can install it for you:
`reticulate::install_python("3.11")`.

## 2. Install the R package

```r
install.packages("remotes")
remotes::install_github("jeanfeydy/survivalGPU", subdir = "R")
```

## 3. Install the Python libraries

Create a Python environment called `survivalGPU`, and install the libraries
in it. You only do this once; PyTorch is a large download, so it takes a few
minutes.

```r
library(reticulate)

virtualenv_create("survivalGPU", version = ">=3.10")
virtualenv_install("survivalGPU", c("numpy", "torch", "pandas", "scipy",
                                    "matplotlib", "beartype", "jaxtyping"))

# Only for the WCE model, wceGPU() -- not available on Windows:
virtualenv_install("survivalGPU", "pykeops")
```

## 4. Use it

At the start of each R session, select the Python environment **before**
loading survivalGPU:

```r
library(reticulate)
use_virtualenv("survivalGPU")

library(survivalGPU)
use_cuda()   # TRUE if a GPU is detected; FALSE means survivalGPU runs on CPU
```

Then use `coxphGPU()` and `wceGPU()`: see the {doc}`user guide <user_guide/index>`
for examples.

## Good to know

- **Windows:** the Cox model (`coxphGPU()`) works, but the WCE model
  (`wceGPU()`) doesn't: it needs `pykeops`, which doesn't support Windows.
  Skip the `pykeops` line above. To use WCE on Windows, install R inside
  [WSL2](https://learn.microsoft.com/windows/wsl/install) and follow the
  Linux instructions.
- **macOS:** there's no NVIDIA GPU on Macs, so survivalGPU runs on CPU.
- **GPU on Windows:** the PyTorch installed by default on Windows only uses
  the CPU. To use an NVIDIA GPU, reinstall PyTorch from its GPU package
  index, with the CUDA version that matches your setup (see
  [pytorch.org](https://pytorch.org/get-started/locally/)), for example:

  ```r
  reticulate::virtualenv_install("survivalGPU", "torch",
    pip_options = "--index-url https://download.pytorch.org/whl/cu124")
  ```

- **GPU and the WCE model:** to run `wceGPU()` on an NVIDIA GPU, KeOps also
  needs the **CUDA toolkit** (`nvcc`), not only the GPU driver.
- **`virtualenv_create()` fails** with "Suitable Python installation for
  creating a venv not found"? The Python it picked can't create virtual
  environments (e.g. on Ubuntu, without the `python3.X-venv` package). Give
  it another Python explicitly:

  ```r
  virtualenv_create("survivalGPU", python = "/usr/bin/python3.10")
  # or: virtualenv_create("survivalGPU", python = install_python("3.11"))
  ```

- **Your own Python environment:** you can also install these libraries in
  an environment of your own, and select it with
  `reticulate::use_virtualenv()`, `reticulate::use_condaenv()` or the
  `RETICULATE_PYTHON` environment variable before loading survivalGPU. See
  the [reticulate documentation](https://rstudio.github.io/reticulate/articles/versions.html).
