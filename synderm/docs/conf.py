import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

project = "synderm"
copyright = "2026, Manrai Lab"
author = "Thomas Buckley, Luke Sagers, James Diao, Luke Melas-Kyriazi, Arjun Manrai"
release = "0.2.0"

extensions = ["nbsphinx", "sphinx.ext.autodoc", "sphinx.ext.viewcode", "sphinx.ext.napoleon"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
nbsphinx_execute = "never"  # notebooks are executed with papermill on a GPU box, not by the docs build
autodoc_mock_imports = ["torch", "torchvision", "diffusers", "transformers", "timm", "peft", "accelerate",
                        "bitsandbytes", "webdataset", "datasets", "scipy", "pandas", "numpy", "PIL", "tqdm", "requests"]
html_theme = "sphinx_rtd_theme"
html_static_path = []
