#!/bin/bash

# Exit on error
set -e

echo "Setting up JAX environment..."

# Define Poetry path
POETRY_PATH="$HOME/.local/bin/poetry"

# Install Poetry if not already installed
if [ ! -f "$POETRY_PATH" ]; then
    echo "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -

    # Verify Poetry installation
    if [ ! -f "$POETRY_PATH" ]; then
        echo "Poetry installation failed. Trying alternative method..."
        pip install poetry
    fi

    echo "Poetry installed successfully"
else
    echo "Poetry is already installed"
fi

# Verify Poetry is available
if [ ! -f "$POETRY_PATH" ]; then
    echo "ERROR: Poetry is still not available. Please install it manually:"
    echo "pip install poetry"
    exit 1
fi

# Install project dependencies (including absl-py) without installing the project itself
echo "Installing project dependencies..."
"$POETRY_PATH" install --no-root

# Install JAX with TPU support
echo "Installing JAX with TPU support..."
"$POETRY_PATH" run pip install "jax[tpu]>=0.4.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Install Jupyter Notebook
echo "Installing Jupyter Notebook..."
"$POETRY_PATH" run pip install jupyter notebook

echo "Setup completed successfully!"
echo "To run the scripts, use: $POETRY_PATH run python <script_name>.py"
echo ""
echo "To start Jupyter Notebook server, run: $POETRY_PATH run jupyter notebook"
echo "This will launch a Jupyter server and open a browser window with the Jupyter interface."