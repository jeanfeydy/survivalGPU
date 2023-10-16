.onAttach <- function(libname, pkgname) {
  # startup messages
  packageStartupMessage(paste("Please run `use_cuda()` to check CUDA drivers"))
}

.onLoad <- function(libname, pkgname) {
  # Python path
  python_path <- system.file("python", package = "survivalGPU")
  survivalgpu <<- reticulate::import_from_path("survivalgpu",
                                               path = python_path,
                                               delay_load = TRUE)
}
