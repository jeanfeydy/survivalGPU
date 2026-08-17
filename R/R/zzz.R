.onAttach <- function(libname, pkgname) {
  # startup messages
  packageStartupMessage(paste("Please run `use_cuda()` to check CUDA drivers"))
}

utils::globalVariables("survivalgpu")

.onLoad <- function(libname, pkgname) {
  # Python path
  python_path <- system.file("python", package = "survivalGPU")
  assign("survivalgpu",
         reticulate::import_from_path("survivalgpu",
                                       path = python_path,
                                       delay_load = TRUE),
         envir = parent.env(environment()))
}

#' Turn a failed Python/torch/pykeops access into an informative error
#'
#' `survivalgpu` is imported with `delay_load = TRUE`, so nothing actually
#' happens at package load time -- the real import (and any missing Python /
#' torch / pykeops dependency) only surfaces the first time an attribute of
#' the module is accessed. This wraps that access so users get a clear
#' `stop()` instead of a raw reticulate/Python traceback.
#' @noRd
survivalgpu_unavailable_error <- function(e) {
  stop(
    "survivalGPU requires a working Python installation with 'torch' and ",
    "'pykeops' installed, which could not be loaded.\n",
    "See vignette(\"python_connect\", package = \"survivalGPU\") for setup ",
    "instructions, and run use_cuda() for diagnostics.\n",
    "Original error: ", conditionMessage(e),
    call. = FALSE
  )
}
