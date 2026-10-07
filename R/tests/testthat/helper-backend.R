#' Skip a test if the Python backend can't be used.
#'
#' On CRAN, skip before reticulate is touched at all: starting Python on a
#' machine where none is configured makes reticulate (>= 1.41) download uv and
#' a Python interpreter into the user's cache directory, which must not happen
#' during a check. No CRAN machine has torch anyway.
#'
#' Elsewhere, check for torch rather than for survivalgpu: the Python code is
#' bundled with the R package (inst/python) and loaded from there, so it is
#' only importable by name when it has also been pip-installed.
skip_if_no_backend <- function() {
  testthat::skip_on_cran()
  testthat::skip_if_not(
    reticulate::py_module_available("torch"),
    message = paste(
      "PyTorch is not available to reticulate.",
      "See vignette(\"installation\", package = \"survivalGPU\")."
    )
  )
}
