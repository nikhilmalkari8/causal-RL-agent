#!/bin/bash

# Setup and Run Script for Causal RL Project
echo "🚀 Setting up Causal RL Project..."

# Clean up old files
echo "🧹 Cleaning up old files..."
find . -name "*.pth" -delete 2>/dev/null || true
rm -rf results/ plots/ models/ __pycache__/ 2>/dev/null || true

# Create fresh directories
echo "📁 Creating directories..."
mkdir -p results plots models

# Install dependencies
echo "📦 Installing dependencies..."
pip install torch numpy gymnasium matplotlib seaborn scipy pandas tqdm

# Run the master training script
echo "🎯 Starting comprehensive training and evaluation..."
python master_train_evaluate.py

echo "✅ Complete! Check the results/ and plots/ directories for outputs."