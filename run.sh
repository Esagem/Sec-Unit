#!/usr/bin/env bash
# Sec-Unit pipeline runner — sets up a venv, installs deps, and runs the pipeline.
#
# Usage:
#   ./run.sh                          # Run all 9 input combinations from inputs/
#   ./run.sh <pdf1> <pdf2>            # Run a single pair
#   ./run.sh --build                  # Build the PyInstaller binary only
#
# Requirements: python3 (>= 3.10), kubescape on PATH (for Task 3).

set -e

VENV_DIR="comp5700-venv"
PY="python3"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[+] Creating virtual environment in $VENV_DIR"
  $PY -m venv "$VENV_DIR"
fi

# Activate venv (portable across bash/zsh, git-bash, WSL)
if [[ -f "$VENV_DIR/Scripts/activate" ]]; then
  # Windows (git-bash)
  source "$VENV_DIR/Scripts/activate"
else
  # Linux/macOS
  source "$VENV_DIR/bin/activate"
fi

echo "[+] Installing dependencies"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

if [[ "${1:-}" == "--build" ]]; then
  echo "[+] Building PyInstaller binary"
  python build.py
  echo "[+] Binary at dist/sec-unit"
  exit 0
fi

if [[ $# -eq 2 ]]; then
  echo "[+] Running pipeline on $1 and $2"
  python main.py "$1" "$2"
else
  echo "[+] Running pipeline on all 9 input combinations"
  python main.py --all
fi
