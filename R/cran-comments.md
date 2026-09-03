# CRAN comments

## Test environments

* Local: Ubuntu 22.04, R 4.6.1, `R CMD check --as-cran` run against a
  clean checkout with `RETICULATE_PYTHON` pointed at a bare Python
  installation with none of `torch`/`pykeops` present — simulating a CRAN
  check machine, since none of them have these either. Result: 0 errors,
  0 warnings, 2 NOTEs (both explained below).
* win-builder (R-devel, Windows): 1 NOTE (explained below); install,
  dependencies, tests, and vignette rebuilding all clean.
* GitHub Actions CI (r-lib/actions), on every push: Ubuntu R-devel,
  R-release, R-oldrel-1 (with a full `torch`/`pykeops` install), a
  dedicated job with no Python/torch/pykeops installed at all, and macOS
  R-release.

## This is a new release

This is the first submission of survivalGPU to CRAN.

## `SystemRequirements: python, pytorch, pykeops`

The package wraps a bundled Python backend (via `reticulate`) for its two
GPU-capable models, `coxphGPU()` and `wceGPU()`. These system requirements
are **optional at check time**:

* The Python module is imported with `delay_load = TRUE`, so package
  loading never touches Python.
* Every exported function that does need the Python backend
  (`coxphGPU()`, `wceGPU()`, `use_cuda()`) fails with an informative
  `stop()` message if it isn't available, rather than erroring
  cryptically or hanging.
* All `\examples{}` that call these functions are wrapped in
  `\dontrun{}`.
* All tests that call these functions are guarded with
  `testthat::skip_if_not(reticulate::py_module_available("survivalgpu"))`,
  so they skip cleanly rather than error when Python/torch/pykeops aren't
  present.
* The vignettes are pre-rendered: `vignettes/*.Rmd.orig` (with live code)
  are knitted offline into the shipped `vignettes/*.Rmd`, which contain
  the already-executed output as static markdown, not live executable
  chunks. `R CMD check` never needs to run Python to rebuild them.

A user who wants the actual GPU functionality needs Python with `torch`
and `pykeops` installed — see `vignette("python_connect")` for setup
instructions (`reticulate::virtualenv_create()` etc.).

## NOTEs

* **"Found the following (possibly) invalid URLs: ... /issues ... Status:
  404 / 503"** — `https://github.com/jeanfeydy/survivalGPU/issues` is
  intermittently unreachable to automated HTTP checks (a 404 in local
  testing, a 503 via win-builder) despite the repository and its Issues
  page being directly reachable in a browser. We believe this is a
  transient issue on GitHub's side with automated requests rather than a
  problem with the URL itself.
* **"possibly misspelled word WCE"** — WCE (Weighted Cumulative Exposure)
  is a modeling method implemented by the package and is spelled out on
  first use in the `Description` field.
