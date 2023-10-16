#' @name survivalGPU-package
#' @aliases survivalGPU
#' @docType package
#' @title survivalGPU: Fast survival analysis
#' @description Allows to perform survivals analysis on GPU with coxph and WCE
#' models, and several features to use bootstrap and manage memory.
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
NULL



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
#' Specifies wether you are using GPUs or not. If TRUE, CUDA drivers are
#' detected, and you are using GPU.
#'
#' @export
#'
#' @examples
#' use_cuda()
use_cuda <- function() {
  # survivalgpu <- use_survivalGPU()
  return(survivalgpu$use_cuda)
}
