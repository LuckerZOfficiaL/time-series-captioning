import os
import json
import torch
from tqdm import tqdm
from helpers import(
    load_config,
    generate_prompt_for_baseline,
    save_file,
    get_response
)
from claude_api import(
    get_claude_response,
    get_claude_image_response
)

MCQ_FILE = "/home/ubuntu/thesis/data/datematch/mcqs_final.jsonl"
#MCQ_FILE = "/home/ubuntu/thesis/data/datematch/mcqs_mid2_short_len6.jsonl"

# List of models to run inference on
MODELS = [
    #"OpenAI GPT-4o",
    #"OpenAI GPT-5",
    #"claude-3-haiku",
    "gemini-2.0-flash"
]


def build_prompt(mcq):
    """Build the prompt for the MCQ task."""
    ts_str = ", ".join(str(v) for v in mcq["time_series"])
    options_str = "\n".join(
        f"{k}. {v}" for k, v in sorted(mcq["options"].items())
    )

    return f"""
You are given an hourly time series.

Time series:
[{ts_str}]

The series starts at:
{mcq["starting_time"]}

Question:
What is the value of the time series at the following time?

{mcq["question_time"]}

Options:
{options_str}

You MUST answer using exactly one capital letter.
Final answer (one letter only):
""".strip()


def get_model_response(prompt, model):
    """Get response from the specified model (text-only)."""
    if "claude" in model.lower():
        # Claude models
        if "haiku" in model.lower():
            return get_claude_response(prompt, model="bedrock/us.anthropic.claude-3-haiku-20240307-v1:0")
        else:
            return get_claude_response(prompt, model=model)

    elif "gpt" in model.lower() or "openai" in model.lower():
        # GPT-4o
        return get_response(prompt=prompt, model="OpenAI GPT-4o")

    elif "gemini" in model.lower():
        # Gemini 2.0 Flash
        return get_response(prompt=prompt, model="Google Gemini-2.0-Flash")

    else:
        raise ValueError(f"Unsupported model: {model}")


def evaluate_model(model_name):
    """Run MCQ evaluation for a single model."""
    correct = 0
    total = 0

    # Output file will be named based on model
    model_name_safe = model_name.replace(" ", "_").replace("-", "_").lower()
    out_file = f"/home/ubuntu/thesis/data/datematch/answers/{model_name_safe}_{MCQ_FILE.split("/")[-1].split(".")[0]}_predictions.jsonl"

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    print(f"\nEvaluating model: {model_name}")
    print(f"Output file: {out_file}\n")

    with open(MCQ_FILE) as f, open(out_file, "w") as out:
        for line in tqdm(f, desc=f"Evaluating {model_name}"):
            mcq = json.loads(line)
            prompt = build_prompt(mcq)

            # Get model response
            response = get_model_response(prompt, model_name)

            # Extract prediction - find first capital letter A/B/C/D
            pred = None
            for ch in response:
                if ch in {"A", "B", "C", "D"}:
                    pred = ch
                    break

            gold = mcq["answer"]
            is_correct = (pred == gold)

            correct += int(is_correct)
            total += 1

            # Save result
            out.write(json.dumps({
                "series_id": mcq["series_id"],
                "bucket": mcq["bucket"],
                "gold": gold,
                "pred": pred,
                "correct": is_correct,
                "raw": response
            }) + "\n")

    accuracy = correct / total if total > 0 else 0
    print(f"\nResults for {model_name}:")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4f}")
    print("="*80)

    return {"model": model_name, "total": total, "correct": correct, "accuracy": accuracy}


def main():
    """Main function to run MCQ evaluation for all models."""
    print(f"Running inference for {len(MODELS)} models")
    print(f"MCQ file: {MCQ_FILE}")
    print("="*80)

    all_results = []

    for model in MODELS:
        print(f"Running {model}...")
        result = evaluate_model(model)
        all_results.append(result)

    # Print summary of all results
    print("\n" + "="*80)
    print("SUMMARY OF ALL MODELS")
    print("="*80)
    for result in all_results:
        print(f"{result['model']:30s} | Accuracy: {result['accuracy']:.4f} ({result['correct']}/{result['total']})")
    print("="*80)


if __name__ == "__main__":
    main()
