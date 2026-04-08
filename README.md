## Overview

This project provides tools to:
- Generate descriptive captions for time series data across multiple domains
- Create and evaluate various reasoning tasks (retrieval, comparison, QA)
- Support multiple state-of-the-art models with multi-GPU inference
- Benchmark model performance on time series understanding tasks

## Project Structure

```
source/
├── inference/          # Model-specific inference scripts
├── tasks/              # Task generation and evaluation
│   ├── caption_retrieval.py      # Caption matching tasks
│   ├── ts_retrieval.py           # Time series retrieval tasks
│   ├── plot_retrieval.py         # Plot-based retrieval tasks
│   ├── perturbed_*.py            # Perturbed data tasks
│   └── task_helpers.py           # Common task utilities
├── multi_gpu_utils.py  # Multi-GPU parallel processing
├── helpers.py          # Core utilities and prompt generation
├── generate_captions_baseline.py  # Baseline caption generation
├── score_unified.py    # Unified scoring and evaluation
└── evaluate_*.py       # Various evaluation scripts
```

## Supported Models

### Vision Language Models
- **Meta Llama 3.2-11B-Vision-Instruct** - Text + image input
- **InternVL 2.5-8B** - Multimodal understanding
- **Qwen-VL** - Alibaba's vision-language model
- **Microsoft Phi-4** - Compact multimodal model
- **LLaVA** - Popular open-source VLM
- **SmolVLM** - Efficient vision-language model
- **IDEFICS** - Flamingo-style multimodal model

### Text-Only Models
- **Qwen** series - For text-only caption generation
- **DeepSeek-Math-7B** - Mathematical reasoning
- Fine-tuned variants of above models

New models can be added in `source/inference/` to evaluate on tasks or generate captions.

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd time-series-captioning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage Examples


### 2. Create Tasks

**Generate new time series Q&A tasks:**
```bash
python -m source.tasks.caption_retrieval
python -m source.tasks.ts_retrieval
python -m source.tasks.plot_retrieval
```

**Run model inference:**
```bash
python -m source.inference.llama_infer
python -m source.inference.qwenvl_infer
```

### 3. Score Model Performance

**Evaluate model results:**
```bash
python source/score_unified.py subsample/tasks.json llama_inference_results --output llama_results.json
```

**Score model performance on Q&A tasks:**
```bash
python source/score_unified.py subsample/tasks.json llama_inference_results
```

### 4. Generate Baseline Captions

**Generate captions using API models:**
```bash
python source/generate_captions_baseline.py
```

### 5. Fine-tune Models on Caption Data

**Fine-tune Qwen model:**
```bash
python source/qwen_fine_tune.py
```

### Task Types

1. **Caption Retrieval** - Multiple choice selection of correct captions
2. **Time Series Retrieval** - Finding matching time series from descriptions
3. **Plot Retrieval** - Visual plot matching tasks
4. **Comparison Tasks** - Comparing time series characteristics
5. **Perturbed Tasks** - Robustness testing with modified data

### Domain Coverage

The framework supports time series from multiple domains:
- Air Quality
- Crime Statistics
- Border Crossings
- Demographics
- Road Injuries
- COVID-19 data
- CO2 emissions
- Dietary patterns
- Online Retail
- Walmart sales
- Agriculture

### Evaluation Metrics

- Accuracy on multiple-choice tasks
- JSON response parsing and validation
- Answer format standardization (A/B/C/D, True/False)
- Performance analysis across domains and task types

## Configuration

### Data Paths
Update paths in individual scripts or use environment variables:
- `DATA_DIR` - Input data directory
- `OUT_DIR` - Output results directory
- `MODEL_PATH` - Model checkpoint paths

### Model Parameters
Adjust model-specific settings in inference scripts:
- Batch size (default: 10)
- Temperature for generation
- Max new tokens
- GPU allocation

### GPU Configuration
Configure in `multi_gpu_utils.py`:
- `NUM_GPUS_TO_USE` - Number of GPUs to utilize
- `BATCH_SIZE` - Processing batch size per GPU

## Output Format

Generated captions and task results are saved as:
- Individual `.txt` files for each time series
- JSON task files with prompts and ground truth
- Evaluation results in JSON format
- Performance metrics and accuracy scores
