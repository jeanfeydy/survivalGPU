# survivalGPU

<!-- badges: start -->
[![R-CMD-check](https://github.com/jeanfeydy/survivalGPU/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/jeanfeydy/survivalGPU/actions/workflows/R-CMD-check.yaml)
[![Build Status](https://github.com/jeanfeydy/survivalGPU/actions/workflows/python-package.yml/badge.svg?branch=refactor_objects&event=push)](https://github.com/jeanfeydy/survivalGPU/actions)
[![Codecov Status](https://codecov.io/gh/jeanfeydy/survivalGPU/branch/refactor_objects/graph/badge.svg)](https://codecov.io/gh/jeanfeydy/survivalGPU)
<!-- badges: end -->

Fast implementation of survival analysis models (CoxPH, WCE...) with GPU support, for R and Python.
Please note that this package is still little more than a proof of concept: we are working to publish a first stable version by the summer of 2023. We have opened the code to get a first feed back from the community, but stress that our solver has not yet been tested thoroughly. The user interface of the Python and R packages are also likely to change over the next few months.

If you find this work useful, please cite:


*Accélération des calculs à l'aide de cartes graphiques pour la détection de signaux de pharmacovigilance sur le Système national des données de santé : le package survivalGPU*, A. Van Straaten, P. Sabatier, J. Feydy, A-S. Jannot, Revue d'Épidémiologie et de Santé Publique, Volume 71, Supplement 1, 2023, 101467, ISSN 0398-7620, https://doi.org/10.1016/j.respe.2023.101467.



## Python: setup

The Python package requires Python >= 3.10. From this directory, create and activate a
virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

Then install the package. For everyday use:

```bash
pip install -e .
```

To also pull in the tools needed to run the test suite and linters (`pytest`,
`hypothesis`, `rpy2`, `black`, `flake8`, `pre-commit`...), use the `test` extra
instead:

```bash
pip install -e . --group test
```

`pykeops`, one of the core dependencies, just-in-time compiles CUDA/C++ kernels,
so a working C++ compiler is required; a CUDA-capable GPU is optional (the
package will fall back to CPU otherwise).

## Python: quickstart

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

See `python/survivalgpu/datasets.py` (e.g. `load_drugs`) for utilities to
generate larger synthetic datasets, and `python/tests/` for further usage
examples.

## Python: run tests

Once the package is installed with the `test` extra (see above), you can run
the pre-commit hooks with:
```bash
pre-commit install
pre-commit run --all-files
```

And run the tests with:
```bash
pytest
```

Note: `python/tests/test_wce_drugdata.py` cross-checks results against the R
`WCE` package via `rpy2`, so it requires `WCE` to be installed in your R
library (`install.packages("WCE")` from an R session; it is also listed as a
`Suggests` dependency in `R/DESCRIPTION`). If it isn't installed, `pytest`
will fail at collection for the whole suite; skip that file instead:
```bash
pytest --ignore=python/tests/test_wce_drugdata.py
```

## R: setup

The R package is a thin wrapper around the Python implementation, calling into
it via `reticulate`. Set up the Python package first (see "Python: setup"
above), then install the R-side dependencies:

```r
install.packages(c("devtools", "pkgload", "testthat", "roxygen2", "reticulate"))
```

`devtools` depends on `fs`, which links against the system `libuv` library.
If `install.packages("fs")` fails with "libuv was not found" (no `libuv1-dev`
on the machine and no permission to install it), build `fs`'s bundled copy of
libuv instead:

```bash
USE_BUNDLED_LIBUV=1 Rscript -e 'install.packages("fs")'
```

By default, `reticulate` picks its own Python interpreter, which may not have
`survivalgpu` installed. Point it at the virtual environment created in the
Python setup step by setting `RETICULATE_PYTHON` before launching R:

```bash
export RETICULATE_PYTHON=/path/to/survivalGPU/.venv/bin/python
```

## R: quickstart

From the `R/` folder, launch an R interactive session and run:

```r
devtools::load_all()
library(survival)

# Three (start, stop] intervals, one covariate:
my_data <- data.frame(
  start = c(0, 0, 0),
  stop  = c(1, 1, 2),
  event = c(0, 1, 1),
  x     = c(1.0, 0.0, 4.0)
)

fit <- coxphGPU(Surv(start, stop, event) ~ x, data = my_data, ties = "efron")
coef(fit)
```

## R: run tests

The test suite cross-checks results against the `WCE` R package, listed as a
`Suggests` dependency in `R/DESCRIPTION`:

```r
install.packages("WCE")
```

Then, from the `R/` folder:

```r
library(devtools)
load_all()
test()

# To render the documentation as a static website:
pkgdown::build_site()
```
