#!/bin/bash
# run_all.sh - Script to run preprocessing, CF-Recsys, and A-LLMRec sequentially

# Step 1: Preprocessing and Feature Extraction
echo "Running preprocessing and feature extraction..."
python guilherme/src/data/preprocessing_and_features.py
if [ $? -ne 0 ]; then
    echo "Preprocessing failed. Exiting."
    exit 1
fi

# Step 2: CF-Recsys (Phase 0)
echo "Setting up CF-Recsys (Phase 0)..."
python guilherme/src/the_sauce/a_llmrec/pre_train/sasrec/main.py
if [ $? -ne 0 ]; then
    echo "CF-Recsys phase 0 failed. Exiting."
    exit 1
fi

# Step 3: A-LLMRec (Phase 1 and beyond)
echo "Running A-LLMRec..."
# If you're using Windows paths, you might need to adjust or use forward slashes.
python "C:/Users/gbert/startup/axl/guilherme/src/the_sauce/a_llmrec/main.py"
if [ $? -ne 0 ]; then
    echo "A-LLMRec failed. Exiting."
    exit 1
fi

echo "All scripts executed successfully."
