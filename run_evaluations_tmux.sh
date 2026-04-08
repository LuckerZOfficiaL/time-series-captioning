#!/bin/bash

# List of models to evaluate
  #"claude-3-haiku_text"
  #"OpenAI GPT-4o_text"
  #"gemini-2.0-flash_text"
  #"smolvlm"
  #"phi4"

models=(
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
  )

# Base path
base_path="/home/ubuntu/thesis/data/samples/new_samples_no_overlap/generated_captions/new_prompt"

# Available GPU devices - EDIT THIS LIST to change available GPUs
AVAILABLE_GPUS=(4 5 6)

# Maximum number of parallel jobs
MAX_PARALLEL=10

# Function to get a random GPU from the available list
get_random_gpu() {
  local num_gpus=${#AVAILABLE_GPUS[@]}
  local random_index=$((RANDOM % num_gpus))
  echo "${AVAILABLE_GPUS[$random_index]}"
}

echo "Starting evaluations for ${#models[@]} models with max $MAX_PARALLEL in parallel..."
echo ""

# Track running sessions
running_count=0

for model in "${models[@]}"; do
  # Wait if we have MAX_PARALLEL sessions running
  while [ $running_count -ge $MAX_PARALLEL ]; do
    sleep 5
    # Count how many eval sessions are still running
    running_count=$(tmux list-sessions 2>/dev/null | grep -c "^eval_" || echo 0)
  done

  # Create a session name from the model name (replace spaces and special chars)
  session_name="eval_${model//[ \/]/_}"

  # Get a random GPU for this evaluation
  gpu_id=$(get_random_gpu)

  # Build the full path for this model
  model_path="${base_path}/${model}"

  echo "Starting tmux session: $session_name for model: $model (GPU: $gpu_id)"
  echo "  Path: $model_path"

  # Create tmux session and run the evaluation
  tmux new-session -d -s "$session_name" \
    "CUDA_VISIBLE_DEVICES=$gpu_id python /home/ubuntu/thesis/source/evaluate_captions.py \
    --generated_captions_folder_path \"$model_path\"; \
    echo ''; \
    echo 'Evaluation completed for: $model'; \
    echo 'Press any key to close this tmux session...'; \
    read"

  running_count=$((running_count + 1))
  sleep 1
done

echo ""
echo "============================================"
echo "All evaluation sessions launched!"
echo "============================================"
echo ""
echo "Active tmux sessions:"
tmux list-sessions 2>/dev/null | grep "^eval_"
echo ""
echo "To attach to a session, use: tmux attach -t <session_name>"
echo "To list all sessions: tmux ls"
echo "To kill all eval sessions: tmux kill-session -t eval_"
