# Automatic Task Labelling Pipeline
*Version 0.25* Updated 06/08/2026

ATLP is a module that enables automatic labelling for Galbot G1 teleoperation dataset.

Currently, ATLP can only label RGB head camera video from Galbot G1 using Qwen3-VL-2B-Instruct.

## Documentation
Run index.html inside '/docs/build/html/', or by using
```bash
# Inside the root of the repo
python -m http.server 8000
```
and access the 'http://localhost:8000/' in your browser.

## Installation
```bash
# Inside the root of the repo
conda create -n atlp-env pip
conda activate atlp-env
pip install -r requirements.txt
```
Don't forget to use 'conda deactivate' when you are done using this module.

## Usage
Users are encouraged to import this module inside Jupyter notebook and use it as advised in the example of the documentation.

## Uninstall
Delete all directories from this repo and the downloaded Qwen3 model inside '~/.cache/huggingface/hub/'.

Also run this command to remove the conda environment.
```bash
conda env remove –name atlp-env
```
