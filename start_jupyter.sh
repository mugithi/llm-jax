#!/bin/bash

# Exit on error
set -e

# Define Poetry path
POETRY_PATH="$HOME/.local/bin/poetry"

# Check if Poetry is installed
if [ ! -f "$POETRY_PATH" ]; then
    echo "ERROR: Poetry is not installed. Please run setup.sh first."
    exit 1
fi

echo "Starting Jupyter Notebook server..."
echo "The server will be accessible at http://localhost:8888"
echo "Press Ctrl+C to stop the server when you're done."

# Start Jupyter Notebook server
"$POETRY_PATH" run jupyter notebook