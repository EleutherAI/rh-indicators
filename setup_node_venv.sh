#!/bin/bash
# Setup node-specific virtual environment for shared storage clusters
# 
# This script creates a venv in a node-specific location to avoid
# hardcoded Python interpreter paths that break across nodes.
#
# Usage:
#   ./setup_node_venv.sh [python_executable]
#
# If python_executable is not provided, it will try to find Python 3.11+
# or fall back to python3

set -e

# Get node name for venv directory
NODE_NAME=$(hostname)
VENV_DIR=".venv-${NODE_NAME}"

# Determine Python executable
if [ -n "$1" ]; then
    PYTHON="$1"
elif command -v python3.11 &> /dev/null; then
    PYTHON="python3.11"
elif command -v python3.12 &> /dev/null; then
    PYTHON="python3.12"
elif command -v python3.13 &> /dev/null; then
    PYTHON="python3.13"
else
    PYTHON="python3"
fi

# Check Python version
PYTHON_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
echo "Using Python: $PYTHON ($PYTHON_VERSION)"
echo "Creating venv in: $VENV_DIR"

# Check if Python version meets requirements (>=3.11,<3.14)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]); then
    echo "WARNING: Project requires Python >=3.11,<3.14, but found $PYTHON_VERSION"
    echo "The venv will be created, but installation may fail."
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Remove old venv if it exists
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing venv: $VENV_DIR"
    rm -rf "$VENV_DIR"
fi

# Create new venv
echo "Creating virtual environment..."
$PYTHON -m venv "$VENV_DIR"

# Activate and upgrade pip
echo "Upgrading pip..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

# Install project
echo "Installing project dependencies..."
pip install -e ".[dev]"

# Install djinn if path exists
DJINN_PATH="/mnt/ssd-1/david/djinn"
if [ -d "$DJINN_PATH" ]; then
    echo "Installing djinn from: $DJINN_PATH"
    pip install -e "$DJINN_PATH"
fi

echo ""
echo "✓ Virtual environment created successfully!"
echo ""
echo "To activate this venv, run:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Or add this to your shell config:"
echo "  alias activate-venv='source $(pwd)/$VENV_DIR/bin/activate'"

