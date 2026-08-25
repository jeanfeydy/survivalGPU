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
#' happens at package load time -- the real import only surfaces the first
#' time an attribute of the module is accessed. `coxphGPU()` only needs
#' 'torch'; `wceGPU()` additionally needs 'pykeops' (not available on
#' Windows), and that failure only surfaces when `wce_R()` is actually
#' called, since the Python package itself imports fine without pykeops.
#' Both cases are routed through this helper so users get a clear `stop()`
#' instead of a raw reticulate/Python traceback.
#' @noRd
survivalgpu_unavailable_error <- function(e) {
  stop(
    "survivalGPU requires a working Python installation with the necessary ",
    "packages installed ('torch' for coxphGPU(); additionally 'pykeops', ",
    "not available on Windows, for wceGPU()), which could not be loaded.\n",
    "See vignette(\"python_connect\", package = \"survivalGPU\") for setup ",
    "instructions, and run use_cuda() for diagnostics.\n",
    "Original error: ", conditionMessage(e),
    call. = FALSE
  )
}
