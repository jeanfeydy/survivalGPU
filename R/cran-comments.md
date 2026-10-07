# CRAN comments

This is a new submission.

## Test environments

* Local, Ubuntu 22.04, R 4.6.1: `R CMD check --as-cran` on the built
  tarball, on an account with no Python configured (empty home directory,
  `RETICULATE_PYTHON` unset), as on a CRAN check machine. This machine
  reports one more NOTE, only because HTML Tidy is not installed on it.
* TODO before submitting: add the win-builder (R-devel, R-release) and macOS
  builder results here, and the GitHub Actions result for the submitted
  commit. Remove this line.

## R CMD check results

0 errors | 0 warnings | 1 note

* checking CRAN incoming feasibility ... NOTE

  New submission.

  Possibly misspelled words in DESCRIPTION: Abrahamowicz, Sylvestre, GPUs,
  WCE, Scalable, natively.

  These are spelled correctly. "Abrahamowicz" and "Sylvestre" are the authors
  of the cited reference. "GPUs" and "WCE" are acronyms, both spelled out in
  the Description. "Scalable" and "natively" are English words.

## Python dependency

The computations run in a Python backend that is bundled with the package
and called through 'reticulate'. It needs Python with 'PyTorch' (see
`SystemRequirements`), which CRAN check machines do not provide. The package
is written so that checking it never starts Python:

* The Python module is imported with `delay_load = TRUE`: loading or
  attaching the package does not start Python.
* Examples: `coxphGPU()`, `wceGPU()`, `use_cuda()`, `HR()` and the `predict`
  and `residuals` methods all need a fitted model, hence the Python backend.
  Their examples are therefore in `\dontrun{}`. They all run in our
  environment, with Python and 'PyTorch' installed.
* Tests: every test file that needs the backend starts by calling a helper
  whose first statement is `testthat::skip_on_cran()`, before any call to
  'reticulate'. No Python is started, and nothing is downloaded or written
  outside the check directory. Elsewhere the tests run whenever 'PyTorch' is
  available.
* Vignettes: the two vignettes with results are precomputed. Their sources
  (`vignettes/*.Rmd.orig`, not shipped) are knitted on our machine, and the
  shipped `.Rmd` files contain the output and figures as static content.
  Rebuilding them needs no Python.
* Without Python, or without the required Python packages, the exported
  functions stop with a message that names what is missing and points to
  `vignette("installation")`.

## Code adapted from other packages

The C code in `src/` and parts of the R code (`R/coxph_internals.R`, and the
pre- and post-processing in `coxphGPU()`) are adapted from the 'survival'
package (LGPL (>= 2)). Its author, Terry Therneau, is listed in `Authors@R`
with the roles `ctb` and `cph`.
