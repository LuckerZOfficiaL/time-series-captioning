#!/bin/bash

# Script to run evaluate_captions_stat_inf_only.py for multiple models

# Base path for generated captions
BASE_PATH="/home/ubuntu/thesis/data/samples/new_samples_no_overlap/generated_captions/new_prompt"

# List of models to evaluate
MODELS=(
    "claude-3-haiku_text"
    "OpenAI GPT-4o_text"
    "gemini-2.0-flash_text"
    "smolvlm"
    "llava_7b"
    "internvl2b"
    "llama_vlm_11b"
    
    "gemma_27b"
    "qwenvl"
    "qwenvl_pal"
    "gemma_12b"
    "internvl38b"
    "internvl8b"
    "idefics2_8b"
    "phi4"
)

# Python script path
EVAL_SCRIPT="/home/ubuntu/thesis/source/evaluate_captions_stat_inf_only.py"

# Log file
LOG_FILE="/home/ubuntu/thesis/evaluation_batch_$(date +%Y%m%d_%H%M%S).log"

echo "Starting batch evaluation at $(date)" | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Counter for tracking progress
total=${#MODELS[@]}
current=0

# Loop through each model
for model in "${MODELS[@]}"; do
    current=$((current + 1))
    folder_path="${BASE_PATH}/${model}"

    echo "" | tee -a "$LOG_FILE"
    echo "[$current/$total] Processing: $model" | tee -a "$LOG_FILE"
    echo "Folder: $folder_path" | tee -a "$LOG_FILE"
    echo "Started at: $(date)" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"

    # Check if folder exists
    if [ ! -d "$folder_path" ]; then
        echo "WARNING: Folder does not exist: $folder_path" | tee -a "$LOG_FILE"
        echo "Skipping..." | tee -a "$LOG_FILE"
        continue
    fi

    # Run the evaluation script
    python "$EVAL_SCRIPT" --generated_captions_folder_path "$folder_path" 2>&1 | tee -a "$LOG_FILE"

    # Check exit status
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "SUCCESS: Completed $model" | tee -a "$LOG_FILE"
    else
        echo "ERROR: Failed to process $model" | tee -a "$LOG_FILE"
    fi

    echo "Finished at: $(date)" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Batch evaluation completed at $(date)" | tee -a "$LOG_FILE"
echo "All results logged to: $LOG_FILE"
