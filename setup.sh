#!/bin/bash

# GCA Project Setup Script
# This script sets up the environment using uv package manager

set -e

echo "🚀 Setting up GCA project with Gemini API..."

# Check if GEMINI_API_KEY is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠️  WARNING: GEMINI_API_KEY environment variable is not set!"
    echo "Please export your Gemini API key:"
    echo "  export GEMINI_API_KEY=your_api_key_here"
    echo ""
    read -p "Do you want to continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python version
echo "🔍 Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
if [[ "$PYTHON_VERSION" != "3.12" ]]; then
    echo "⚠️  WARNING: Python 3.12 is required, but found Python $PYTHON_VERSION"
    echo "Please install Python 3.12 before continuing."
    read -p "Do you want to continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ Python 3.12 detected"
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "✅ uv installed successfully!"
    echo "⚠️  Please restart your shell or run: source ~/.bashrc"
    exit 0
fi

echo "✅ uv is already installed"

# Create virtual environment with Python 3.12
echo "🔧 Creating virtual environment with Python 3.12..."
uv venv --python 3.12

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
echo "  - PyTorch 2.4.0+cu124 from PyTorch index"
echo "  - DGL 2.4.0+cu124 from DGL wheels"
echo "  - google-generativeai and other dependencies"
uv pip install -e .

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To verify the setup, you can run:"
echo "  python -c 'import google.generativeai as genai; print(\"Gemini API module loaded successfully!\")'"
echo ""
echo "Make sure to export your GEMINI_API_KEY before running any scripts:"
echo "  export GEMINI_API_KEY=your_api_key_here"
echo ""
echo "Happy coding! 🎉"
