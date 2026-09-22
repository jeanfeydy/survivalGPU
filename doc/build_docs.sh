#!/usr/bin/env bash
#
# Builds the full survivalGPU documentation site end to end: knits the R
# user-guide pages, builds the Sphinx site (which also executes the Python
# user-guide pages), then builds the R API reference (pkgdown) nested into
# the Sphinx output. See doc/README.md for requirements and a lighter
# Python-only alternative.
#
# Usage (from the repository root):
#   bash doc/build_docs.sh

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

rm -rf doc/_build

# 1. Knit the R vignette sources into plain, static Markdown pages. Needs a
#    live reticulate -> Python(+torch, +pykeops) environment, exactly like
#    R/vignettes/precompile.R already does.
Rscript doc/scripts/knit_r_pages.R

# 2. Build the Sphinx site: executes the MyST-NB Python user-guide pages,
#    runs autodoc/autosummary against python/survivalgpu, and ingests the
#    already-static R pages from step 1 as plain content (no execution).
sphinx-build -W -b html doc/source doc/_build/html

# 3. Build the R API reference (pkgdown) *last*, nested directly into the
#    Sphinx output tree at r/reference/. No Sphinx source document lives at
#    that path, so step 2 never touches it -- running pkgdown afterwards
#    means its output can't be clobbered.
Rscript doc/scripts/build_pkgdown.R

echo "Documentation built at doc/_build/html/"
