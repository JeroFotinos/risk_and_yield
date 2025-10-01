# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# --- We're not installing the package in the tox venv that tests docstyle,
# --- so we need to add the package root to sys.path here.
sys.path.insert(0, os.path.abspath("../.."))

import risknyield  # after sys.path insert

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "risknyield"
copyright = "2025, Jeronimo Fotinos"
author = "Jeronimo Fotinos"
release = risknyield.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # NumPy/Google docstrings
    "sphinx.ext.autosummary",  # summary tables for APIs
    "sphinx.ext.intersphinx",  # cross-link to NumPy, SciPy, etc.
    "sphinx.ext.viewcode",  # link to highlighted source
    "sphinx_autodoc_typehints",  # render type hints nicely
    # Optional:
    # "myst_parser",                # Markdown support (MyST)
    "nbsphinx",                   # Jupyter notebooks
    "sphinx_copybutton",
    "sphinx_design",
]

# Autosummary: generate stub pages automatically
autosummary_generate = True

# Autodoc defaults: don’t include members with no docstrings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": False,   # <- hide the empty, auto-annotated attributes
    "show-inheritance": True,
    "inherited-members": False,
    "imported-members": False,  # globally avoid double-documenting reexports
}

# Napoleon: use the "Attributes" section in your docstrings as the source of
# truth
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_attr_annotations = False   # <- don't duplicate attributes
napoleon_use_ivar = True            # <- render attributes as :ivar:

# Type hints: prefer in the description for cleaner signatures
autodoc_typehints = "description"

# Mock heavy/optional imports at build time (add if needed)
autodoc_mock_imports = []

# Sphinx 8+ compatible intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}


templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# --- HTML theme
html_theme = "pydata_sphinx_theme"
# html_theme = "furo"

html_theme_options = {
    # left “Section navigation”
    "show_nav_level": 2,        # show 2 levels expanded in the left nav
    "navigation_depth": 4,      # how deep the tree can go in the sidebar
    "collapse_navigation": True,

    # right “On this page”
    "secondary_sidebar_items": ["page-toc"],  # add "edit-this-page" later if you want

    # misc niceties
    "show_prev_next": True,
    "navigation_with_keys": True,
}

# optional: icons/links in the header (add your repo if you want)
html_theme_options["icon_links"] = [
    {"name": "GitHub", "url": "https://github.com/JeroFotinos/risk_and_yield", "icon": "fab fa-github"},
]

# code highlighting similar to SciPy
pygments_style = "default"
pygments_dark_style = "github-dark"  # nice dark mode pairing

# We have custom static files already:
html_static_path = ["_static"]

# If at some point we use MyST:
# myst_enable_extensions = ["colon_fence", "deflist"]

# Fail on warnings (CI-friendly; keep off locally if noisy)
# nitpicky = True
