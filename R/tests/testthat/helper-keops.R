#' Skip a test if the optional pykeops Python dependency isn't installed.
#'
#' Only wceGPU() needs pykeops (not available on Windows); coxphGPU() only
#' needs torch. Files sourced from tests/testthat/helper-*.R run before any
#' test, so this is available to every test file without an explicit source().
skip_if_no_pykeops <- function() {
  testthat::skip_if_not(
    reticulate::py_module_available("pykeops"),
    message = paste(
      "pykeops is not installed (only needed for wceGPU()).",
      "Install it with `pip install survivalgpu[wce]`",
      "(not available on Windows) to run WCE tests."
    )
  )
}
