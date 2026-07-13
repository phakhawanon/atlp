#!/bin/bash
# Requires the atlp/atlp_gui packages to be installed first: pip install -e .
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

trap "kill 0" EXIT

# python -m http.server 5500 --directory src/atlp_gui &
python src/atlp_gui/serve.py &
python -m uvicorn atlp_gui.backend:app --reload --port 8000
