#!/bin/bash
# Helper script to run ai-git-squash.py in a virtual environment

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install gitpython openai python-dotenv

# Run the script with all arguments passed to this script
echo "Running ai-git-squash.py..."
python3 ai-git-squash.py "$@"

# Deactivate virtual environment
deactivate
