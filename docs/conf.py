"""Sphinx configuration for the cosmock documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "cosmock"
copyright = "2026, Ivan Espinoza"
author = "Ivan Espinoza"
release = "0.0.1"

root_doc = "index"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_show_sphinx = False

autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
