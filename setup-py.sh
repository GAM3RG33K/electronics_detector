#!/bin/bash

# setup-py.sh - Create and activate a Python virtual environment

# Get the current directory
CURRENT_DIR=$(pwd)
PY_ENV_DIR="${CURRENT_DIR}/.py_env"

echo "Creating Python virtual environment in ${PY_ENV_DIR}"

# Check if Python is installed
if ! command -v python3.11 &> /dev/null; then
    echo "Error: Python 3.11 is not installed or not in PATH"
    exit 1
fi

# Create the virtual environment
python3.11 -m venv "${PY_ENV_DIR}"

# Check if venv creation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment"
    exit 1
fi

echo "Python virtual environment created successfully at ${PY_ENV_DIR}"

# Upgrade pip and setuptools
"${PY_ENV_DIR}/bin/pip" install --upgrade pip setuptools

echo ""
echo "To activate the virtual environment, run:"
echo "  source ${PY_ENV_DIR}/bin/activate"
echo ""
echo "You can activate it now with:"
echo "  source .py_env/bin/activate"
echo ""
echo "When you're done, deactivate it with:"
echo "  deactivate"