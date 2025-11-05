#!/bin/bash

echo "========================================"
echo "Electronics Detection System"
echo "========================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "Python version:"
python3 --version
echo

# Check if dependencies are installed
python3 -c "import cv2" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies"
        exit 1
    fi
    echo
fi

echo "Starting detector..."
echo
python3 electronics_detector.py

echo
echo "Detector stopped."
