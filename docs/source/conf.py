# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath('../../src'))
sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'RM-Gen'
copyright = '2025, Artur Padovesi Piratelli'
author = 'Artur Padovesi Piratelli'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx_rtd_theme",
              "sphinx.ext.napoleon",
              "sphinx_autodoc_typehints",
              "sphinx.ext.viewcode",
              "sphinx.ext.mathjax",
              "sphinx_copybutton",
              "nbsphinx",
             ]
autodoc_typehints = "description"
typehints_fully_qualified = True
simplify_optional_unions = True
nbsphinx_execute = 'never'

autosummary_generate = True
autosummary_imported_members = True

# Ensure recursive structure works
autosummary_generate_overwrite = True

# Recommended with nested autosummary to avoid duplicate index warnings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"


html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": False,
    "navigation_depth": 5,
}
html_show_sourcelink = False
pygments_style = "default"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]
# html_css_files = ['css/sionna.css']

# napoleon_custom_sections = [("Input shape", "params_style"),
#                             ("Output shape", "params_style"),
#                             ("Attributes", "params_style"),
#                             ("Input", "params_style"),
#                             ("Output", "params_style"),
#                             ("Keyword Arguments", "params_style"),
#                             ]
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_keyword = True
numfig = True
