# survivalGPU documentation

This is the source for the unified Sphinx documentation site covering both
the Python (`survivalgpu`) and R (`survivalGPU`) packages. It lives at the
repository root, alongside `python/` and `R/`, because it documents both of
them in one site rather than belonging to either.

## Layout

```
docs/
├── build_docs.sh        # orchestrates the full build (see below)
├── requirements_docs.txt  # Python doc-build dependencies
├── scripts/
│   ├── knit_r_pages.R   # knits R/vignettes/*.Rmd.orig -> docs/source/r/user_guide/*.md
│   └── build_pkgdown.R  # builds the R API reference, nested into the Sphinx output
└── source/               # the Sphinx project itself (conf.py, pages, ...)
    ├── python/           # Python section: installation, user guide, API
    └── r/                 # R section: installation, user guide, API reference
```

The `python/user_guide/*.md` and `r/user_guide/*.md` pages all show **real
executed output** — nothing is hand-copied. R and Python code are executed
by different mechanisms (see below), since Sphinx itself can only execute
Python.

## Full build

Requires a working R + [reticulate](https://rstudio.github.io/reticulate/) →
Python(+torch, +optionally pykeops) environment — the same one
`vignette("python_connect")` in the R package sets up, and that
`R/vignettes/precompile.R` already relies on. From the repository root:

```bash
pip install -r docs/requirements_docs.txt
# Install the R package itself, plus its doc-build Suggests
# (knitr, rmarkdown, pkgdown, survminer):
Rscript -e 'devtools::install("R", dependencies = TRUE)'

bash docs/build_docs.sh
```

Then serve the result locally:

```bash
python -m http.server -d docs/_build/html 8000
```

## Python-only partial build

If you don't have a working R + reticulate environment set up, you can still
build and iterate on the Python section alone:

```bash
pip install -r docs/requirements_docs.txt
sphinx-build -b html docs/source docs/_build/html
```

This skips `docs/scripts/knit_r_pages.R` and `docs/scripts/build_pkgdown.R`, so
the Python section (installation, user guide, API reference) builds and
works normally, but the R section's user-guide pages and its `r/reference/`
API reference will be missing/broken links. That's expected — not a bug —
when iterating on Python-only changes.

## How the R and Python pages get executed

- **Python** (`python/user_guide/*.md`): executed directly by Sphinx itself,
  via `myst-nb`, during the normal `sphinx-build` step.
- **R** (`r/user_guide/*.md`): Sphinx never executes R. Instead,
  `docs/scripts/knit_r_pages.R` knits the *same* live-chunk sources already
  used for the R package's CRAN vignettes (`R/vignettes/{coxPH,WCE}.Rmd.orig`)
  a second time, into plain Markdown with the output already baked in, as a
  separate step that must run *before* `sphinx-build`. Sphinx then just
  treats that Markdown as ordinary static content.
- **R API reference** (`r/reference/`): built separately by `pkgdown` (using
  the existing `R/_pkgdown.yml` config, unchanged), nested into the Sphinx
  output at `docs/_build/html/r/reference/`. This step must run *after*
  `sphinx-build`, since nothing in the Sphinx source tree targets that path
  — running it first would risk it being wiped by a later clean.
