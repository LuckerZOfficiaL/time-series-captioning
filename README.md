# CaTS-Bench: Can Language Models Describe Time Series?

**Findings of ACL 2026** | [Paper](https://arxiv.org/abs/2509.20823)

> *Luca Zhou, Pratham Yashwante, Marshall Fisher, Alessio Sampieri, Zihao Zhou, Fabio Galasso, Rose Yu*

---

## Abstract

Time series captioning, the task of describing time series in natural language, requires numeric and temporal reasoning, trend interpretation, and contextual understanding. Existing benchmarks, however, often rely on fully synthetic or generic captions, and typically neglect metadata and visual representations. We introduce **CaTS-Bench**, a comprehensive benchmark for **C**ontext-**a**ware **T**ime **S**eries reasoning across 11 diverse domains, centered on a gold-standard evaluation set of 1,746 human-rewritten captions that measure how effectively models translate numeric trends into immediately interpretable narratives. To address the scarcity of human-annotated data, we also propose a scalable pipeline for generating high-fidelity synthetic captions. We evaluate leading Vision-Language Models on our benchmark, revealing that even proprietary models struggle to capture numeric nuances in temporal descriptions, while finetuning open-source models on synthetic data yields substantial performance gains. Finally, we release a diagnostic suite of 910 multiple-choice questions and tailored numeric metrics to gauge time-series-specific reasoning capabilities.

---

## Repository Structure

```
source/
├── configs/
│   └── config.yaml                  # Central configuration (models, paths, eval settings)
├── inference/                       # Local model inference scripts
│   ├── llava_infer.py
│   ├── internvl_infer.py
│   ├── phi4_infer.py
│   ├── base_qwen_infer.py
│   ├── fine_tuned_qwen_infer.py
│   ├── dsmath_infer.py
│   └── inference_utils.py
├── qa_tasks/                        # Diagnostic task generation
│   ├── caption_retrieval.py
│   ├── ts_retrieval.py
│   ├── plot_retrieval.py
│   ├── perturbed_caption_retrieval.py
│   ├── perturbed_ts_retrieval.py
│   └── task_helpers.py
├── helpers.py                       # Core utilities: API calls, metrics (BLEU, ROUGE, BERTScore, SimCSE)
├── generate_captions_baseline.py    # Caption generation via API/local VLMs
├── evaluate_captions.py             # Full evaluation (text similarity + numeric accuracy)
├── evaluate_captions_stat_inf_only.py  # Numeric accuracy evaluation only
├── auto_script.py                   # Generation with exponential backoff retry
├── plot_generation.py               # Generate time series plots (JPEG)
├── paraphrase_captions.py           # Synthetic caption augmentation
├── mix_captions.py                  # Caption mixing utilities
├── qwen_fine_tune.py                # Fine-tuning Qwen on synthetic captions
├── simcse.py                        # SimCSE semantic similarity
└── multi_gpu_utils.py               # Multi-GPU batch processing
data/
└── processed/                       # Source time series data (11 domains, JSON format)
run_evaluations_batch.sh             # Sequential evaluation across all models
run_evaluations_tmux.sh              # Parallel evaluation across GPUs (tmux)
```

---

## Domains

CaTS-Bench covers 11 diverse real-world time series domains:

| Domain | File |
|--------|------|
| Air Quality | `aq.json` |
| Agricultural Productivity | `agricultural_productivity.json` |
| Border Crossing | `border_crossing.json` |
| CO₂ Emissions | `co2.json` |
| COVID-19 | `covid.json` |
| Crime Statistics | `crime.json` |
| Demographics | `demographics.json` |
| Diet | `diet.json` |
| Online Retail | `online_retail.json` |
| Road Injuries | `road_injuries.json` |
| Walmart Sales | `walmart.json` |

---

## Installation

```bash
git clone https://github.com/LuckerZOfficiaL/time-series-captioning.git
cd time-series-captioning
pip install -r requirements.txt
```

Set up API credentials (OpenAI, Google Gemini, Anthropic) in `.credentials/`.

---

## Configuration

All behavior is controlled by [`source/configs/config.yaml`](source/configs/config.yaml):

- `data.dataset_names` — which domains to process
- `model.used_models` — which models generate captions
- `path.gt_captions_folder_path` — ground-truth captions for evaluation
- `path.generated_captions_folder_path` — model-generated captions to evaluate
- `eval.use_img_input` — whether to pass plot images to the model

---

## Usage

### Generate captions

```bash
# Using API models (configure model.used_models in config.yaml)
python source/generate_captions_baseline.py

# With automatic retry on failures
python source/auto_script.py
```

### Evaluate captions

```bash
# Full evaluation (text similarity + numeric accuracy)
python source/evaluate_captions.py \
  --generated_captions_folder_path <path> \
  --gt_captions_folder_path <path>

# Numeric accuracy only (min/max/mean/std)
python source/evaluate_captions_stat_inf_only.py \
  --generated_captions_folder_path <path>

# Run all models sequentially
bash run_evaluations_batch.sh

# Run all models in parallel across GPUs
bash run_evaluations_tmux.sh
```

### Generate diagnostic tasks

```bash
python -m source.qa_tasks.caption_retrieval
python -m source.qa_tasks.ts_retrieval
python -m source.qa_tasks.plot_retrieval
```

### Run local model inference

```bash
CUDA_VISIBLE_DEVICES=0 python -m source.inference.llava_infer
CUDA_VISIBLE_DEVICES=0 python -m source.inference.internvl_infer
python -m source.inference.qwenvl_infer
```

### Fine-tune on synthetic captions

```bash
python source/qwen_fine_tune.py
```

---

## Supported Models

**Proprietary (via API):**
- OpenAI GPT-4o / GPT-5
- Google Gemini 2.0 Flash / 2.5 Pro
- Anthropic Claude 3.5 / 3.7 (via AWS Bedrock)

**Open-source (local inference):**
- LLaVA, InternVL 2.5, Qwen-VL, Phi-4, SmolVLM, IDEFICS
- DeepSeek-Math, Llama 3.2 Vision

**Ollama (local):**
- Llama, Mixtral, Gemma, Qwen, DeepSeek-R1, Phi-4, and others

---

## Citation

```bibtex
@inproceedings{zhou2026cats,
  title     = {CaTS-Bench: Can Language Models Describe Time Series?},
  author    = {Zhou, Luca and Yashwante, Pratham and Fisher, Marshall and
               Sampieri, Alessio and Zhou, Zihao and Galasso, Fabio and Yu, Rose},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  year      = {2026}
}
```
