#' @name survivalGPU-package
#' @aliases survivalGPU
#' @docType package
#' @title survivalGPU: Fast survival analysis
#' @description Allows to perform survivals analysis on GPU with coxph and WCE
#' models, and several features to use bootstrap and manage memory.
#'
#' To learn more about survivalGPU, see vignettes :
#' `vignette("coxPH")`
#' `vignette("WCE")`
#' `vignette("python_connect")`
#'
#' @section Functions:
#'
#' - `use_cuda()`: check CUDA drivers for GPU use.
#' - `coxphGPU()`: fit a Cox proportional hazards regression model.
#' - `wceGPU()`:   fit a Weighted Cumulative Exposure model.
#'
#' @author
#' - [Jean FEYDY](https://www.jeanfeydy.com)
#' - [Alexis van STRAATEN](https://alexis-vs.github.io/Portfolio/)
#' @useDynLib survivalGPU, .registration = TRUE
NULL


#' CUDA utilisation
#'
#' @description
#' Specifies wether you are using GPUs or not. If TRUE, CUDA drivers are
#' detected, and you are using GPU.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' use_cuda()
#' }
use_cuda <- function() {
  return(survivalgpu$use_cuda)
}
