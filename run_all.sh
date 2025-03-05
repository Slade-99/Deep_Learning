#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh  # Ensure Conda is recognized
conda activate Final  # Activate your Conda environment

# Specify the directory containing the Python scripts
SCRIPT_DIR="/home/azwad/Works/Deep_Learning/Implementation_Phase/KD_Training_scripts"  # Change this to your actual folder path

# Navigate to the directory
cd "$SCRIPT_DIR" || { echo "Directory not found!"; exit 1; }

# Run all Python scripts in the directory
for script in *.py; do
    echo "Running $script..."
    python3 "$script"
    echo "$script finished!"
done

# Deactivate Conda environment
conda deactivate
