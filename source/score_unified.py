#!/usr/bin/env python3
"""
Unified scoring script for CaTSBench tasks and Q&A.

Usage:
    python score_unified.py <tasks_json_path> <model_answers_dir> [--output results.json]
    python score_unified.py <tasks_json_path> <model_answers_dir> --subsample-correct <output_tasks.json> [--samples 100] [--seed 42]

Examples:
    python score_unified.py easy_subsample/tasks.json finetuned_llama_inference_easy --output llama_results.json
    python score_unified.py starter_tasks/tasks.json qwen_inference_results_subset --subsample-correct correct_subset.json --samples 100
"""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

# Allowed answer formats
_ALLOWED_ANSWERS = {"a", "b", "c", "d", "true", "false"}

def _parse_start(s: str):
    """Extract the first valid answer token from a string."""
    if not s:
        return None
    # Grab the first contiguous token after leading whitespace
    m = re.match(r"\s*(\S+)", s)
    if not m:
        print("String is empty or whitespace only.")
        return None
    token = m.group(1).lower()
    if token not in _ALLOWED_ANSWERS:
        print(f"Unrecognised answer '{m.group(1)}' in: {s!r}")
        return None
    return token.title()

def _attempt_parse_str(s):
    """Attempt to parse answer from string using various heuristics."""
    s = s.lower()
    
    # Handle true/false
    assert not (("false" in s) and ("true" in s))
    if "false" in s:
        return 'False'
    if "true" in s:
        return 'True'
    
    # Look for JSON-like answer format
    pattern = re.compile(r'\s*[\'"]answer[\'"]\s*:\s*[\'"]([A-Da-d])[\'"]\s*')
    match_ = pattern.search(s) 
    if match_:
        return _parse_start(match_.group(1).upper())
    
    # Lots of hardcoded reformats necessary from different LLM answers
    str_reformat = s.replace('.', '').replace('"', '').replace("'", '')
    str_reformat = str_reformat.replace(")", "").replace("(", "").replace("answer:", "").split(' ')
    for x in str_reformat:
        if x in _ALLOWED_ANSWERS:
            return _parse_start(x)
    
    print(f"INVALID ANSWER:\n {s}")
    return ""

def extract_choice(json_str):
    """Extract answer choice from model response (JSON or text format)."""
    orig_str = json_str
    json_str = json_str.replace('```', '').replace('json', '').replace('.', '')
    
    # Try to extract JSON
    if "{" in json_str and "}" in json_str:
        json_str = json_str.split('}')[0] + '}'
    json_str = re.sub(r'("answer"\s*:\s*)([A-Za-z])', r'\1"\2"', json_str)
    
    try:
        data = json.loads(json_str)
    except:
        return _attempt_parse_str(json_str)
    
    try:
        answer = data.get("answer", "").strip()
    except AttributeError:
        # handle LLM answering only 'true' or 'false'
        if type(data) == bool:
            return str(data)
        return _attempt_parse_str(str(data))

    return _parse_start(answer)

def load_tasks(tasks_json_path):
    """Load tasks from unified tasks.json file."""
    print(f"Loading tasks from {tasks_json_path}...")
    with open(tasks_json_path, 'r') as f:
        tasks = json.load(f)
    print(f"Loaded {len(tasks)} tasks")
    return tasks

def load_model_answers(answers_dir):
    """Load model answers from directory of task_id.txt files."""
    print(f"Loading model answers from {answers_dir}...")
    answers = {}
    
    if not os.path.exists(answers_dir):
        print(f"Warning: Answers directory {answers_dir} does not exist")
        return answers
    
    for filename in os.listdir(answers_dir):
        if filename.endswith('.txt'):
            task_id = filename[:-4]  # Remove .txt extension
            filepath = os.path.join(answers_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    answer_text = f.read().strip()
                    parsed_answer = extract_choice(answer_text)
                    answers[task_id] = parsed_answer
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                continue
    
    print(f"Loaded {len(answers)} model answers")
    return answers

def calculate_accuracy(tasks, model_answers):
    """Calculate overall and per-task-type accuracy."""
    results = {}
    task_type_results = defaultdict(list)
    
    # Group tasks by task_type for detailed breakdown
    tasks_by_id = {task['task_id']: task for task in tasks}
    
    correct_count = 0
    total_count = 0
    missing_answers = []
    
    for task in tasks:
        task_id = task['task_id']
        ground_truth = str(task['ground_truth'])
        task_type = task['task_type']
        
        if task_id in model_answers:
            predicted = str(model_answers[task_id])
            is_correct = (predicted == ground_truth)
            
            results[task_id] = {
                'predicted': predicted,
                'ground_truth': ground_truth,
                'correct': is_correct,
                'task_type': task_type
            }
            
            task_type_results[task_type].append(is_correct)
            
            if is_correct:
                correct_count += 1
            total_count += 1
        else:
            missing_answers.append(task_id)
            results[task_id] = {
                'predicted': None,
                'ground_truth': ground_truth,
                'correct': False,
                'task_type': task_type,
                'missing': True
            }
            total_count += 1
    
    # Calculate overall accuracy
    overall_accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    # Calculate per-task-type accuracy
    task_type_accuracy = {}
    for task_type, correct_list in task_type_results.items():
        if correct_list:
            accuracy = sum(correct_list) / len(correct_list)
            task_type_accuracy[task_type] = {
                'accuracy': accuracy,
                'correct': sum(correct_list),
                'total': len(correct_list)
            }
    
    return {
        'overall_accuracy': overall_accuracy,
        'correct_count': correct_count,
        'total_count': total_count,
        'missing_answers': missing_answers,
        'task_type_accuracy': task_type_accuracy,
        'detailed_results': results
    }

def print_results(results):
    """Print formatted results to console."""
    print("\n" + "="*60)
    print("SCORING RESULTS")
    print("="*60)
    
    print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    print(f"Correct: {results['correct_count']}/{results['total_count']}")
    
    if results['missing_answers']:
        print(f"Missing answers: {len(results['missing_answers'])}")
        if len(results['missing_answers']) <= 10:
            print(f"Missing task IDs: {results['missing_answers']}")
        else:
            print(f"First 10 missing: {results['missing_answers'][:10]}...")
    
    print("\nPer-Task-Type Accuracy:")
    print("-" * 40)
    for task_type, stats in sorted(results['task_type_accuracy'].items()):
        accuracy = stats['accuracy']
        correct = stats['correct']
        total = stats['total']
        print(f"{task_type:35}: {accuracy:.4f} ({correct}/{total})")

def save_results(results, output_path):
    """Save results to JSON file."""
    print(f"\nSaving results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

def identify_correct_answers(tasks, model_answers):
    """Identify which tasks were answered correctly by the model."""
    correct_by_type = defaultdict(list)
    
    for task in tasks:
        task_id = task['task_id']
        ground_truth = str(task['ground_truth'])
        task_type = task['task_type']
        
        if task_id in model_answers:
            predicted = str(model_answers[task_id])
            if predicted == ground_truth:
                correct_by_type[task_type].append(task)
    
    return correct_by_type

def subsample_correct_tasks(correct_by_type, samples_per_type=100, seed=42):
    """Randomly subsample correct tasks by type."""
    random.seed(seed)
    subsampled_tasks = []
    
    print(f"\nSubsampling {samples_per_type} correctly answered tasks per type:")
    
    for task_type, correct_tasks in sorted(correct_by_type.items()):
        available = len(correct_tasks)
        
        if available < samples_per_type:
            print(f"  {task_type}: Only {available} correct answers available (requested {samples_per_type})")
            sampled = correct_tasks  # Take all available
        else:
            sampled = random.sample(correct_tasks, samples_per_type)
            print(f"  {task_type}: {len(sampled)} sampled from {available} correct answers")
        
        subsampled_tasks.extend(sampled)
    
    return subsampled_tasks

def main():
    parser = argparse.ArgumentParser(description="Score model performance on CaTSBench tasks")
    parser.add_argument("tasks_json", help="Path to tasks.json file")
    parser.add_argument("answers_dir", help="Directory containing model answer files (task_id.txt)")
    parser.add_argument("--output", "-o", help="Output JSON file for results", default="scoring_results.json")
    parser.add_argument("--subsample-correct", help="Create subset of correctly answered tasks (output path)")
    parser.add_argument("--samples", "-s", type=int, default=100, 
                        help="Number of samples per task type for subsampling (default: 100)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for subsampling (default: 42)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.tasks_json):
        print(f"Error: Tasks file {args.tasks_json} does not exist")
        sys.exit(1)
    
    if not os.path.exists(args.answers_dir):
        print(f"Error: Answers directory {args.answers_dir} does not exist")
        sys.exit(1)
    
    # Load data
    tasks = load_tasks(args.tasks_json)
    model_answers = load_model_answers(args.answers_dir)
    
    # Calculate accuracy
    results = calculate_accuracy(tasks, model_answers)
    
    # Check if we're in subsample mode
    if args.subsample_correct:
        # Identify correctly answered tasks
        print(f"Identifying correctly answered questions...")
        correct_by_type = identify_correct_answers(tasks, model_answers)
        
        # Print statistics
        print(f"\nCorrect answers by task type:")
        total_correct = 0
        for task_type, correct_tasks in sorted(correct_by_type.items()):
            count = len(correct_tasks)
            total_correct += count
            print(f"  {task_type}: {count}")
        
        print(f"\nTotal correctly answered: {total_correct}")
        
        # Check if we have enough correct answers
        insufficient_types = []
        for task_type, correct_tasks in correct_by_type.items():
            if len(correct_tasks) < args.samples:
                insufficient_types.append((task_type, len(correct_tasks)))
        
        if insufficient_types:
            print(f"\nWarning: Some task types have fewer than {args.samples} correct answers:")
            for task_type, count in insufficient_types:
                print(f"  {task_type}: {count} correct answers")
        
        # Subsample correct tasks
        subsampled_tasks = subsample_correct_tasks(correct_by_type, args.samples, args.seed)
        
        # Save subsampled tasks
        print(f"\nSaving {len(subsampled_tasks)} subsampled tasks to {args.subsample_correct}...")
        with open(args.subsample_correct, 'w') as f:
            json.dump(subsampled_tasks, f, indent=2)
        
        # Final summary
        final_counts = defaultdict(int)
        for task in subsampled_tasks:
            final_counts[task['task_type']] += 1
        
        print(f"\nFinal subsampled counts:")
        for task_type, count in sorted(final_counts.items()):
            print(f"  {task_type}: {count}")
        
        print(f"\nDone! Subsampled tasks saved to {args.subsample_correct}")
    else:
        # Regular scoring mode
        # Display results
        print_results(results)
        
        # Save results
        save_results(results, args.output)
        
        print(f"\nDone! Results saved to {args.output}")

if __name__ == "__main__":
    main()