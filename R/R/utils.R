#' Skip if no python
#'
#' @description
#' helper function to skip tests if no python
#'
#' @importFrom reticulate py_available
#' @importFrom testthat skip
#' @noRd
skip_if_no_python <- function() {
  have_python <- reticulate::py_available(initialize = TRUE)
  if(!have_python) testthat::skip("Python not available on system for testing")
}




#' Skip if no module
#'
#' @description
#' helper function to skip tests if we don't have the survivalGPU python dependencies
#'
#' @importFrom reticulate py_module_available
#' @importFrom testthat skip
#' @noRd
skip_if_no_modules <- function() {
  survivalgpu_python_dep <- c("torch", "torch_scatter",
                              "pykeops", "matplotlib",
                              "beartype", "jaxtyping")

  py_module <- lapply(survivalgpu_python_dep, reticulate::py_module_available)
  if(any(py_module == FALSE)){
    testthat::skip("one or more modules not available for testing")
  }
}
