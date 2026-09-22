#!/usr/bin/env Rscript
#
# Builds the R package's existing pkgdown site (R/_pkgdown.yml) and nests it
# into the unified Sphinx-built docs site, at r/reference/. Nothing under
# doc/source/r/reference/ is a Sphinx source document, so sphinx-build never
# writes there -- this script must simply run *after* sphinx-build, never
# before, so its output is never at risk of being clobbered.
#
# Requires the survivalGPU R package itself to be installed (pkgdown runs
# roxygen2 @examples when rendering Reference pages), and -- like
# doc/scripts/knit_r_pages.R -- the `survivalGPU` reticulate virtualenv from
# vignette("python_connect") to already exist and be usable.
#
# Usage (from the repository root, after building the Sphinx site):
#   Rscript doc/scripts/build_pkgdown.R

pkgdown::build_site(
  pkg = "R",
  override = list(destination = file.path(getwd(), "doc/_build/html/r/reference")),
  preview = FALSE
)
