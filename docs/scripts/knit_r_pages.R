#!/usr/bin/env Rscript
#
# Knits the live-chunk vignette sources (R/vignettes/*.Rmd.orig) a *second*
# time, into plain Markdown pages for the Sphinx docs site
# (docs/source/r/user_guide/*.md), with the R (and, via reticulate, Python)
# code already executed and its real output baked in. Sphinx's MyST parser
# then treats these as ordinary static pages -- Sphinx itself never executes
# R.
#
# This is independent from R/vignettes/precompile.R (which knits the same
# *.Rmd.orig sources into CRAN-facing *.Rmd files): that script's output is
# committed to git because CRAN check machines have no live Python, but the
# docs build always has one, so the output of *this* script is regenerated
# on every build and gitignored instead.
#
# Requires the `survivalGPU` reticulate virtualenv from
# vignette("python_connect") to already exist and be usable.
#
# Usage (from the repository root):
#   Rscript docs/scripts/knit_r_pages.R

library(knitr)
render_markdown()

src_dir <- "R/vignettes"
out_dir <- "docs/source/r/user_guide"

# survivalGPU.Rmd.orig is deliberately excluded: it overlaps heavily with
# coxPH + WCE combined, and docs/source/r/index.md already serves as the
# section's lightweight overview instead.
pages <- c("coxPH", "WCE")

# knit() leaves the source's Pandoc YAML front matter (title/output/vignette
# fields) and its "please edit *.Rmd.orig" HTML comment as literal text --
# rmarkdown::render() would normally interpret the former and drop both, but
# nothing downstream in the Sphinx pipeline does. Replace the YAML block with
# a plain Markdown heading using its "title" field, and drop the comment.
promote_title <- function(path) {
  lines <- readLines(path)
  delims <- which(lines == "---")
  if (length(delims) < 2 || delims[1] != 1) {
    return(invisible())
  }
  front_matter <- lines[seq(delims[1], delims[2])]
  title <- sub('^title:\\s*"?(.*?)"?\\s*$', "\\1", grep("^title:", front_matter, value = TRUE)[1])

  body <- lines[(delims[2] + 1):length(lines)]
  body <- body[!grepl("^<!--.*\\.Rmd\\.orig produces.*-->$", body)]

  writeLines(c(paste0("# ", title), body), path)
}

old_wd <- setwd(out_dir)
on.exit(setwd(old_wd))

for (name in pages) {
  # cwd is now out_dir, so a bare relative fig.path is correct both as the
  # physical write location and as the image reference inside the .md file
  # (relative to its own directory) -- see R/vignettes/precompile.R, which
  # uses the same setwd() trick for the same reason.
  opts_chunk$set(fig.path = paste0(name, "_files/"))
  out_file <- paste0(name, ".md")
  knit(file.path(old_wd, src_dir, paste0(name, ".Rmd.orig")), out_file)
  promote_title(out_file)
}
