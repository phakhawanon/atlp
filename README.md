# Automatic Task Labelling Pipeline
*Version 0.25* Updated 08/06/2026

ATLP is a module that enables automatic labelling for Galbot G1 teleoperation dataset.

Currently, ATLP can only label RGB head camera video from Galbot G1 using Qwen3-VL-2B-Instruct.

## Installation
First, clone this repo by
```bash
git clone https://github.com/phakhawanon/atlp/  
```
Then, run these commands to set up the conda environment for running the module.
```bash
# Inside the root of the repo
conda create -n atlp-env pip
conda activate atlp-env

# If you only need `import atlp` (dataset management, tagging/evaluation, plotting):
pip install -r requirements-core.txt

# (Optional) If you also need auto-labelling, 3D simulation, or the GUI backend:
pip install -r requirements-full.txt

# Finally, install this package
pip install -e .
```
Don't forget to use `conda deactivate` when you are done using this module.

## Documentation

The documentation can be accessed via [https://phakhawanon.github.io/atlp/](https://phakhawanon.github.io/atlp/)

## Usage
Users are encouraged to import this module inside Jupyter notebook and use it as advised in the example of the documentation.

## Uninstall
Delete all directories from this repo and the downloaded Qwen3 model inside `~/.cache/huggingface/hub/`.

Also run this command to remove the conda environment.
```bash
conda env remove --name atlp-env
```
