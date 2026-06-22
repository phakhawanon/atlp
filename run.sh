#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

trap "kill 0" EXIT

# python -m http.server 5500 --directory src/atlp_gui &
python src/atlp_gui/serve.py &
python -m uvicorn src.atlp_gui.backend:app --reload --port 8000
