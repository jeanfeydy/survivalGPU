#' @name survivalGPU-package
#' @aliases survivalGPU
#' @title survivalGPU: Fast survival analysis
#' @description Performs survival analysis on GPU-accelerated hardware using
#' Cox proportional hazards and weighted cumulative exposure (WCE) models,
#' with support for bootstrap resampling and memory management.
#'
#' To learn more about survivalGPU, start with the vignette :
#' `vignette("survivalGPU")`
#'
#' @section Functions:
#'
#' - `use_cuda()`: check CUDA drivers for GPU use.
#' - `coxphGPU()`: fit a Cox proportional hazards regression model.
#' - `wceGPU()`:   fit a Weighted Cumulative Exposure model.
#'
#' @author
#' - [Jean FEYDY](https://www.jeanfeydy.com)
#' - Alexis van STRAATEN
#' @useDynLib survivalGPU, .registration = TRUE
"_PACKAGE"



#' Import of survivalgpu python package
#' @noRd
use_survivalGPU <- function() {
  # Python path
  python_path <- system.file("python", package = "survivalGPU")
  survivalgpu <- reticulate::import_from_path("survivalgpu", path = python_path)

  return(survivalgpu)
}


#' CUDA utilisation
#'
#' @description
#' Specifies whether you are using GPUs or not. If TRUE, CUDA drivers are
#' detected, and you are using GPU.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' use_cuda()
#' }
use_cuda <- function() {
  # survivalgpu <- use_survivalGPU()
  return(tryCatch(survivalgpu$use_cuda, error = survivalgpu_unavailable_error))
}
