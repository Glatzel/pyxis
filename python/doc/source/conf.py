# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import sys
from pathlib import Path

import sphinx_autosummary_accessors

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
# Add  directory

sys.path.insert(0, str(Path(__file__).parents[2]))
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
# import pyxis

project = "pyxis"
copyright = "2024, Glatzel"
author = "Glatzel"
# release = pyxis.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Sphinx extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    # Third-party extensions
    "autodocsumm",
    "sphinx_copybutton",
]
maximum_signature_line_length = 88
templates_path = ["_templates"]
exclude_patterns = []

# Below setting is used by
# sphinx-autosummary-accessors - build docs for namespace accessors like `Series.str`
# https://sphinx-autosummary-accessors.readthedocs.io/en/stable/
templates_path = ["_templates", sphinx_autosummary_accessors.templates_path]

# -- Extension settings  -----------------------------------------------------

# sphinx.ext.intersphinx - link to other projects' documentation
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pyproj": ("https://pyproj4.github.io/pyproj/stable/", None),
    "python": ("https://docs.python.org/3", None),
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
