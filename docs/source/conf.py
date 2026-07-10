# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ATLP'
copyright = '2026, Phakhawanon'
author = 'Phakhawanon'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      # Core library for capturing docstrings
    'sphinx.ext.napoleon',     # Lets you use Google or NumPy style docstrings
    'sphinx.ext.viewcode',     # Adds links to the highlighted source code
    'sphinx.ext.todo'
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

import os
import sys

sys.path.insert(0, os.path.abspath('../../src'))

# docs/source/conf.py

# Add a list of the module names that are causing import errors
autodoc_mock_imports = [
    "transformers",
    "torch",
    "av",
    "pinocchio",
    "cv2",
    "matplotlib",
    "numpy",
    "pandas",
    "PIL",
]
todo_include_todos = True

# Break a function/method signature onto one line per parameter once it
# exceeds this many characters. Sphinx treats 0 as "unlimited" (never
# breaks), so use 1 to force wrapping on virtually every signature.
maximum_signature_line_length = 1
