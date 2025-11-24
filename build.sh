#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "Installing dependencies..."
pip install -r backend/requirements.txt

# Only train if models don't exist (saves time on deploys)
if [ ! -d "backend/models" ] || [ -z "$(ls -A backend/models 2>/dev/null)" ]; then
    echo "Training models..."
    python backend/train_models.py
else
    echo "Models already exist, skipping training"
fi

echo "Build completed successfully!"
