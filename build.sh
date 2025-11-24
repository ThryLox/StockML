#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "Installing dependencies..."
pip install -r backend/requirements.txt

echo "Training models..."
python backend/train_models.py

echo "Build completed successfully!"
