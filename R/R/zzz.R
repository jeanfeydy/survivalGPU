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

# Python packages imported by the bundled backend (inst/python). Keep in sync
# with the dependencies in pyproject.toml. 'pykeops' is not listed: only
# wceGPU() needs it.
python_requirements <- c("torch", "numpy", "pandas", "scipy", "matplotlib",
                         "beartype", "jaxtyping")

#' Turn a failed access to the Python backend into an informative error
#'
#' `survivalgpu` is imported with `delay_load = TRUE`, so nothing actually
#' happens at package load time -- the real import only surfaces the first
#' time an attribute of the module is accessed. `wceGPU()` additionally needs
#' 'pykeops' (not available on Windows), and that failure only surfaces when
#' `wce_R()` is actually called, since the Python package itself imports fine
#' without pykeops. All these cases are routed through this helper so users
#' get a clear `stop()` instead of a raw reticulate/Python traceback.
#'
#' The cause is looked for in this order: Python could not be started at all;
#' some Python packages are missing; anything else, in which case the original
#' error is reported as it is.
#'
#' @param e the error caught by `tryCatch()`.
#' @param need_keops `TRUE` when called from `wceGPU()`: 'pykeops' is then
#'   checked too. `coxphGPU()` and `use_cuda()` don't need it.
#' @noRd
survivalgpu_unavailable_error <- function(e, need_keops = FALSE) {
  setup <- paste0(
    "See vignette(\"installation\", package = \"survivalGPU\") for setup ",
    "instructions."
  )

  if (!reticulate::py_available(initialize = TRUE)) {
    stop(
      "survivalGPU could not start Python: reticulate found no usable Python ",
      "environment.\n",
      "Create one and select it before loading survivalGPU. ", setup, "\n",
      "Original error: ", conditionMessage(e),
      call. = FALSE
    )
  }

  required <- c(python_requirements, if (need_keops) "pykeops")
  not_installed <- required[
    !vapply(required, reticulate::py_module_available, logical(1))
  ]

  if (length(not_installed) > 0) {
    stop(
      "survivalGPU needs Python packages that are not installed in the ",
      "Python environment in use: ",
      paste0("'", not_installed, "'", collapse = ", "), ".\n",
      "Python in use: ", reticulate::py_config()$python, "\n",
      "Install them in that environment, or select another one with ",
      "reticulate::use_virtualenv() before loading survivalGPU, in a new R ",
      "session.\n",
      if ("pykeops" %in% not_installed && .Platform$OS.type == "windows") {
        paste0(
          "'pykeops' is not available on Windows: use wceGPU() through WSL2 ",
          "(the Windows Subsystem for Linux).\n"
        )
      },
      setup,
      call. = FALSE
    )
  }

  stop(
    "survivalGPU: the Python backend raised an error.\n",
    "Original error: ", conditionMessage(e),
    call. = FALSE
  )
}
