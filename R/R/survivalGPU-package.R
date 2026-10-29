#' @name survivalGPU-package
#' @aliases survivalGPU
#' @title survivalGPU: Fast Survival Analysis
#' @description Fits Cox proportional hazards and weighted cumulative exposure
#' (WCE) models on graphics processing units (GPUs), with bootstrap confidence
#' intervals computed in batches.
#'
#' To learn more about survivalGPU, start with `vignette("survivalGPU")`.
#'
#' @section Functions:
#'
#' - [use_cuda()]: tells whether computations run on a GPU.
#' - [coxphGPU()]: fits a Cox proportional hazards model.
#' - [wceGPU()]: fits a weighted cumulative exposure model.
#' - [bootstrap()]: adds bootstrap confidence intervals to a fitted model.
#' - [HR()]: hazard ratio between two exposure histories, for a WCE model.
#'
"_PACKAGE"



#' Import of survivalgpu python package
#' @noRd
use_survivalGPU <- function() {
  # Python path
  python_path <- system.file("python", package = "survivalGPU")
  survivalgpu <- reticulate::import_from_path("survivalgpu", path = python_path)

  return(survivalgpu)
}


#' Check Whether a GPU Is Used
#'
#' @description
#' Tells whether computations run on a GPU, i.e. whether 'PyTorch' detects
#' CUDA drivers and a CUDA-capable GPU.
#'
#' @return A single logical value: `TRUE` if 'PyTorch' detects a CUDA-capable
#'   GPU, in which case computations run on it by default; `FALSE` if they run
#'   on the CPU. An error is raised if the Python backend cannot be loaded.
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
