#!/bin/sh
# Double-click (macOS) or run (Linux) to set up and open the Folio Processor web app.
# First run may need: chmod +x run_folio_app.command
cd "$(dirname "$0")" || exit 1
python3 run_app.py || python run_app.py
