# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

project = "survivalGPU"
copyright = "2026, Jean Feydy, Antoine Poirot-Bourdain, Alexis van Straaten"
author = "Jean Feydy, Antoine Poirot-Bourdain, Alexis van Straaten"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "myst_nb",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# myst-nb: parses .md as MyST (subsumes myst_parser) and executes pages that
# declare a Jupyter `kernelspec` in their front matter. Plain .md files
# (e.g. ones knitted from R) have no kernelspec, so they're never executed.
nb_execution_mode = "cache"
nb_execution_timeout = 120

# autodoc / autosummary / napoleon: survivalgpu's docstrings are Google-style.
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = False

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Sphinx's own default ("python3") would otherwise apply Python syntax
# highlighting to fenced code blocks with no explicit language tag -- the
# only such blocks in this site are the R knit pipeline's plain-text output
# blocks (doc/scripts/knit_r_pages.R), which should render as plain text.
highlight_language = "none"
