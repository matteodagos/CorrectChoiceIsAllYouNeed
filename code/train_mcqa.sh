#!/usr/bin/env bash
# File: train_mcqa.sh
# Usage:  bash train_mcqa.sh

set -euo pipefail                     # fail fast on any error or undefined var
IFS=$'\n\t'

#### 1. Create & activate a fresh virtual environment
VENV_DIR="train_mcqa/venv-mcqa"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

#### 2. Upgrade pip + install project requirements
python -m pip install --upgrade pip
python -m pip install -r train_mcqa/requirements.txt

#### 3. Run the training script
python train_mcqa/train_mcqa_model.py
