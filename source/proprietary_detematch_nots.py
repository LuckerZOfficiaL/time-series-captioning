import os
import json
from tqdm import tqdm
from helpers import get_response
from claude_api import get_claude_image_response

MCQ_FILE = "/home/ubuntu/thesis/data/datematch/mcqs_final.jsonl" #"/home/ubuntu/thesis/data/datematch/nots/mcqs_mid2_short_len6.jsonl"
IMG_DIR = "/home/ubuntu/thesis/data/datematch/nots/plots_start_only"




# List of models to run inference on
MODELS = [
    "OpenAI GPT-4o",
    "OpenAI GPT-5",
    "claude-3-haiku",
    "gemini-2.0-flash"
]

def build_prompt(mcq):
    """Build the prompt for the MCQ task."""
    options = "\n".join(
        f"{k}. {v}" for k, v in sorted(mcq["options"].items())
    )

    return f"""
You are given an hourly time series shown as a plot.

Only the starting time is labeled on the x-axis.
Each point represents one hour.

Start time:
{mcq["starting_time"]}

Question:
What is the value of the time series at the following time?

{mcq["question_time"]}

Options:
{options}

You MUST answer using exactly one capital letter.
Final answer (one letter only):
""".strip()


def get_model_response(image_path, prompt, model):
    """Get response from the specified model (vision models)."""
    if "claude" in model.lower():
        # Claude vision models
        if "haiku" in model.lower():
            return get_claude_image_response(
                image_path=image_path,
                prompt=prompt,
                model="bedrock/us.anthropic.claude-3-haiku-20240307-v1:0"
            )
        elif "3.7" in model.lower() or "3-7" in model.lower():
            return get_claude_image_response(
                image_path=image_path,
                prompt=prompt,
                model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0"
            )
        else:
            return get_claude_image_response(
                image_path=image_path,
                prompt=prompt,
                model=model
            )

    elif "gpt" in model.lower() or "openai" in model.lower():
        # GPT-4o vision - need to implement vision support in helpers
        # For now, using the standard get_response (you may need to enhance this)
        return get_response(prompt=prompt, model="OpenAI GPT-4o")

    elif "gemini" in model.lower():
        # Gemini 2.0 Flash vision - need to implement vision support
        return get_response(prompt=prompt, model="Google Gemini-2.0-Flash")

    else:
        raise ValueError(f"Unsupported model: {model}")


def run_single_mcq(mcq, model):
    """Run a single MCQ with the specified model."""
    img_path = os.path.join(IMG_DIR, f"{mcq['series_id']}.jpeg")
    if not os.path.exists(img_path):
        return None, "Missing image"

    prompt = build_prompt(mcq)
    response = get_model_response(img_path, prompt, model)

    # Extract prediction - find first capital letter A/B/C/D
    pred = next((c for c in response if c in {"A", "B", "C", "D"}), None)
    return pred, response

def evaluate_model(model_name):
    """Run MCQ evaluation for a single model."""
    correct = 0
    total = 0

    # Output file will be named based on model
    model_name_safe = model_name.replace(" ", "_").replace("-", "_").lower()
    out_file = f"/home/ubuntu/thesis/data/datematch/answers/nots/long/{model_name_safe}_{MCQ_FILE.split('/')[-1].split('.')[0]}_vision_predictions.jsonl"

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    print(f"\nEvaluating model: {model_name}")
    print(f"Output file: {out_file}\n")

    with open(MCQ_FILE) as f, open(out_file, "w") as out:
        for line in tqdm(f, desc=f"Evaluating {model_name}"):
            mcq = json.loads(line)

            pred, raw = run_single_mcq(mcq, model_name)
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
                "raw": raw
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
    print(f"Image directory: {IMG_DIR}")
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