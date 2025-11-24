#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "Installing dependencies..."
pip install -r backend/requirements.txt

echo "Removing old models (if any)..."
rm -rf backend/models
rm -rf models

echo "Training models..."
python backend/train_models.py

echo "Build completed successfully!"
