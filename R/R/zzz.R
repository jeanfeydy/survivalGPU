.onAttach <- function(libname, pkgname) {
  # startup messages
  packageStartupMessage(paste("Please run `use_cuda()` to check CUDA drivers"))
}
