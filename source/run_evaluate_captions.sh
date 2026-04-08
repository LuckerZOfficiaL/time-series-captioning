#!/bin/bash

# Base path for generated captions
BASE_PATH="/home/ubuntu/thesis/data/samples/new_samples_no_overlap/generated_captions/new_prompt"

# List of model folders to evaluate
MODELS=(
    "chatts_14b"
    "timeomni1_7b" 
)

# Change to source directory
cd /home/ubuntu/thesis/source

# Run evaluation for each model
for model in "${MODELS[@]}"; do
    FOLDER_PATH="${BASE_PATH}/${model}"

    # Check if folder exists
    if [ -d "$FOLDER_PATH" ]; then
        echo "=============================================="
        echo "Evaluating: $model"
        echo "=============================================="
        python evaluate_captions.py --generated_captions_folder_path "$FOLDER_PATH"
        echo ""
    else
        echo "Skipping $model - folder does not exist: $FOLDER_PATH"
    fi
done

echo "All evaluations complete!"
