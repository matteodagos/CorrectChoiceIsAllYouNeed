#!/usr/bin/env bash

# DPO Training Script for MNLP M3 Final Submission
# Team: Intelligent Agents

cd train_dpo

# Install the required packages
pip install -r requirements.txt

# For training the DPO model to get ciacco/MNLP_M3_dpo_model, run:
python train_dpo.py

# Note: Training takes 11-12 hours and may crash with OOM near the end
# Check logs with: tensorboard --logdir logs/
# See README for manual steps regarding authentication, merging, and reproducibility
