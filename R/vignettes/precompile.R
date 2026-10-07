#!/usr/bin/env Rscript
#
# Precompiles the vignettes that need a live Python environment (coxPH, WCE,
# survivalGPU) from their *.Rmd.orig source into static *.Rmd files, with the
# code already executed and its output baked in. This way R CMD check never
# has to run Python/reticulate itself: it just formats the already-computed
# *.Rmd files.
#
# Requires the `survivalGPU` reticulate virtualenv from
# vignette("installation") to already exist and be usable.
#
# Usage (from the R/ package directory):
#   Rscript vignettes/precompile.R
#
# Run this whenever a *.Rmd.orig file or the survivalGPU API changes, then
# commit the regenerated *.Rmd (and any new files under figures/).

local({
  orig_files <- list.files("vignettes", pattern = "\\.Rmd\\.orig$", full.names = TRUE)

  old_wd <- setwd("vignettes")
  on.exit(setwd(old_wd))

  # Keep KeOps' start-up messages (e.g. "CUDA libraries not found") out of
  # the vignettes. Must be set before Python starts.
  Sys.setenv(KEOPS_VERBOSE = "0")

  for (f in basename(orig_files)) {
    out <- sub("\\.orig$", "", f)
    # Not "<name>_files/": rmarkdown::render() deletes that directory when it
    # builds the vignette, so the figures would be missing from the package.
    knitr::opts_chunk$set(fig.path = paste0("figures/", tools::file_path_sans_ext(out), "-"))
    knitr::knit(f, out)

    # knitr leaves trailing spaces on some output lines (e.g. "n= 227,
    # number of events= 164 "): strip them, so that the pre-commit
    # trailing-whitespace hook doesn't have to modify the files.
    lines <- readLines(out)
    writeLines(sub("[ \t]+$", "", lines), out)
  }
})
