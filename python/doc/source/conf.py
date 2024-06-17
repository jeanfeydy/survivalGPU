from __future__ import annotations

import importlib.metadata
import locale
import os
import sys
import warnings
from pathlib import Path

# Allow to import local modules
sys.path.insert(0, str(Path().resolve()))
# conf_module is where we define dynamic_scraper and reset_pyvista
# pyvista configuration
# See: https://github.com/pyvista/pyvista/blob/main/doc/source/conf.py
import pyvista
from conf_module import dynamic_scraper, reset_pyvista
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.utilities.docs import (  # noqa: F401
    linkcode_resolve,
    pv_html_page_context,
)

# Otherwise VTK reader issues on some systems, causing sphinx to crash. See also #226.
# locale.setlocale(locale.LC_ALL, "en_US.UTF-8")


# Manage errors
pyvista.set_error_output_file("errors.txt")
# Ensure that offscreen rendering is used for docs generation
pyvista.OFF_SCREEN = True  # Not necessary - simply an insurance policy
# Preferred plotting style for documentation
pyvista.set_plot_theme("document")
pyvista.global_theme.window_size = [1024, 768]
pyvista.global_theme.font.size = 22
pyvista.global_theme.font.label_size = 22
pyvista.global_theme.font.title_size = 22
pyvista.global_theme.return_cpos = False
pyvista.set_jupyter_backend(None)
# Save figures in specified directory
pyvista.FIGURE_PATH = Path("./images/", "auto-generated/")
if not Path.exists(pyvista.FIGURE_PATH):
    Path.mkdir(pyvista.FIGURE_PATH, parents=True)

# necessary when building the sphinx gallery
pyvista.BUILDING_GALLERY = True
os.environ['PYVISTA_BUILDING_GALLERY'] = 'true'

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.",
)

# Prevent deprecated features from being used in examples
warnings.filterwarnings(
    "error",
    category=PyVistaDeprecationWarning,
)

project = "survivalGPU"
copyright = "2024, The survivalGPU team"
author = "The survivalGPU team"
version = release = "test" #importlib.metadata.version("survivalgpu") # TODO metadata vesrion

extensions = [
    "myst_parser",
    "sphinx_design",
    "pyvista.ext.plot_directive",
    "pyvista.ext.viewer_directive",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_math_dollar",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    'sphinx_gallery.gen_gallery',
]


pygments_style = 'sphinx'

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

source_suffix = [".rst", ".md"]
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]

html_theme = "furo"

myst_enable_extensions = [
    "colon_fence",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
]



sphinx_gallery_conf = {
    'examples_dirs': '../../examples',   # path to your example scripts
    'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
    'filename_pattern': '/plot_', # execute only files that start with `plot_`
    "ignore_pattern": "/utils_",
    "image_scrapers": (dynamic_scraper, "matplotlib"), # dynamic_scraper is defined in conf_module.py
    "first_notebook_cell": "%matplotlib inline",
    "backreferences_dir": None,
    # Reset module did not work with sphinx-gallery 0.16.0
    # we assume that documentation settings are not modified in examples
    "reset_modules": (reset_pyvista,), # reset_pyvista is defined in conf_module.py
    "reset_modules_order": "both",
}

suppress_warnings = ["config.cache"]
always_document_param_types = False
