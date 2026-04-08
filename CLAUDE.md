# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A **time series captioning and evaluation framework** that:
- Generates natural language captions for time series data across 11 domains (air quality, crime, COVID-19, etc.)
- Creates and evaluates reasoning tasks (caption retrieval, time series retrieval, plot matching, Q&A)
- Benchmarks VLMs and LLMs on time series understanding
- Contains a separate `MM/` research subproject with 12 model merging techniques

Data lives at `/home/ubuntu/thesis/data` (symlinked to `/shared/tsqa`). API keys are in `.credentials/`.

## Common Commands

```bash
# Generate captions using API models (OpenAI, Gemini, Claude)
python source/generate_captions_baseline.py

# Generate with auto-retry on failures
python source/auto_script.py

# Evaluate generated captions (full metrics)
python source/evaluate_captions.py \
  --generated_captions_folder_path <path> \
  --gt_captions_folder_path <path>

# Evaluate statistical inference only
python source/evaluate_captions_stat_inf_only.py \
  --generated_captions_folder_path <path>

# Run batch evaluation for all models sequentially
bash run_evaluations_batch.sh

# Run parallel evaluation across GPUs (uses tmux sessions)
bash run_evaluations_tmux.sh

# Generate task datasets
python -m source.tasks.caption_retrieval
python -m source.tasks.ts_retrieval
python -m source.tasks.plot_retrieval

# Run local model inference
CUDA_VISIBLE_DEVICES=0 python -m source.inference.llava_infer
CUDA_VISIBLE_DEVICES=0 python -m source.inference.internvl_infer
python -m source.inference.qwenvl_infer

# Score task results
python source/score_unified.py subsample/tasks.json <inference_results_folder>

# Fine-tune Qwen
python source/qwen_fine_tune.py

# Count words in dataset
python source/count_words.py
```

## Architecture

### Configuration

All behavior is driven by `source/configs/config.yaml`:
- `data.dataset_names` — which domains to process
- `model.used_models` — which models generate captions (values must match `model.all_models`)
- `path.*` — all input/output paths; `generated_captions_folder_path` and `gt_captions_folder_path` are the key ones for evaluation
- `eval.evaluated_model` — model being evaluated (used to label output)
- `eval.use_img_input` — whether to pass plot images to the model

### Core Modules (`source/`)

| File | Role |
|------|------|
| `helpers.py` | Central utility library (~4000 lines): API calls (OpenAI/Gemini/Claude/Bedrock/Ollama), caption generation, fact extraction, all evaluation metrics (BLEU, ROUGE, METEOR, BERTScore, SimCSE) |
| `generate_captions_baseline.py` | Calls API/local VLMs to produce baseline captions; outputs `.txt` files per time series |
| `evaluate_captions.py` | Full evaluation: text similarity + statistical inference accuracy |
| `evaluate_captions_stat_inf_only.py` | Evaluation focused on numeric accuracy (min/max/mean/std) |
| `auto_script.py` | Wraps generation scripts with exponential backoff retry logic |
| `multi_gpu_utils.py` | Batch processing across multiple GPUs |
| `simcse.py` | Sentence-level semantic similarity (SimCSE) evaluation |

### Inference Scripts (`source/inference/`)

Each script targets a specific local model: `llava_infer.py`, `internvl_infer.py`, `phi4_infer.py`, `base_qwen_infer.py`, `fine_tuned_qwen_infer.py`, `dsmath_infer.py`. They share utilities from `inference_utils.py`.

### Task Generation (`source/tasks/`)

Tasks are multiple-choice datasets. `caption_retrieval.py`, `ts_retrieval.py`, and `plot_retrieval.py` generate the main task types. `perturbed_*.py` variants create robustness-testing versions with modified data.

### Data Flow

1. Raw time series CSVs + metadata → `plot_generation.py` produces JPEG plots
2. `generate_captions_baseline.py` sends (time series data + optional plot) to model → `.txt` caption files
3. Optional: `caption_refinement.py`, `extract_facts.py`, `fact_bank.py` enrich/filter captions
4. `evaluate_captions.py` compares generated captions against GT captions using multiple metrics
5. `tasks/` scripts package data into MCQ JSON datasets; `score_unified.py` evaluates model answers

### Output Structure

Generated captions are saved as individual `.txt` files per time series under a model-named folder (e.g., `generated_captions/new_prompt/gemini-2.0-flash_text/`). Evaluation results go to `data/evaluation_results/`.

### Model Merging Subproject (`MM/`)

Each subdirectory is an independent research project implementing a published merging technique (AdaMerging, DARE, TALL Masks, Ties-Merging, TransFusion, RotationSymmetry, etc.). They have their own dependencies and are not integrated with the main captioning pipeline.

## API Model Names

Model name strings in config must match exactly. Key mappings:
- `"OpenAI GPT-4o"` → GPT-4o via OpenAI API
- `"Google Gemini-2.0-Flash"` → Gemini via `google.genai`
- `"Anthropic Claude-3.5"` → Claude via AWS Bedrock (`boto3`)
- `"Ollama <model>"` → local Ollama instance
- `"bedrock/us.anthropic.claude-3-7-sonnet-..."` → Claude 3.7 via Bedrock (used in `qa_model`)