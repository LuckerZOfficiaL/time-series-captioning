#!/usr/bin/env python3
"""
Script to randomly subsample 10% of questions from each task_type in tasks.json
with a fixed seed for reproducibility.
"""

import json
import random
from collections import defaultdict

def subsample_tasks(input_file, output_file, hard_questions_file=None, sample_rate=0.1, min_samples=200, seed=42):
    """
    Subsample tasks by task_type with fixed seed, filtering out hard questions.
    
    Args:
        input_file: Path to input tasks.json
        output_file: Path to output subsampled tasks.json
        hard_questions_file: Path to hard questions file to filter out (optional)
        sample_rate: Fraction of tasks to sample (default 0.1 for 10%)
        min_samples: Minimum number of samples per task type (default 200)
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Load hard questions to filter out if provided
    hard_task_ids = set()
    if hard_questions_file:
        print(f"Loading hard questions from {hard_questions_file}...")
        with open(hard_questions_file, 'r') as f:
            hard_questions = json.load(f)
        hard_task_ids = {task['task_id'] for task in hard_questions}
        print(f"Found {len(hard_task_ids)} hard questions to filter out")
    
    # Load tasks
    print(f"Loading tasks from {input_file}...")
    with open(input_file, 'r') as f:
        tasks = json.load(f)
    
    # Filter out hard questions if specified
    if hard_task_ids:
        original_count = len(tasks)
        tasks = [task for task in tasks if task['task_id'] not in hard_task_ids]
        filtered_count = len(tasks)
        print(f"Filtered out {original_count - filtered_count} hard questions ({original_count} -> {filtered_count})")
    
    # Group tasks by task_type
    tasks_by_type = defaultdict(list)
    for task in tasks:
        tasks_by_type[task['task_type']].append(task)
    
    # Print original counts and check minimum requirements
    print("\nOriginal task counts:")
    for task_type, task_list in sorted(tasks_by_type.items()):
        print(f"  {task_type}: {len(task_list)}")
        if len(task_list) < min_samples:
            raise ValueError(f"Task type '{task_type}' has only {len(task_list)} tasks after filtering hard questions, "
                           f"but minimum required is {min_samples}")
    
    # Subsample each task type
    subsampled_tasks = []
    print(f"\nSubsampling {min_samples} from each task type:")
    
    for task_type, task_list in sorted(tasks_by_type.items()):
        n_samples = min_samples  # Always sample exactly min_samples
        sampled = random.sample(task_list, n_samples)
        subsampled_tasks.extend(sampled)
        actual_rate = n_samples / len(task_list) * 100
        print(f"  {task_type}: {len(task_list)} -> {n_samples} ({actual_rate:.1f}%)")
    
    # Save subsampled tasks
    print(f"\nSaving {len(subsampled_tasks)} subsampled tasks to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(subsampled_tasks, f, indent=2)
    
    # Print final summary
    final_counts = defaultdict(int)
    for task in subsampled_tasks:
        final_counts[task['task_type']] += 1
    
    print("\nFinal subsampled counts:")
    for task_type, count in sorted(final_counts.items()):
        print(f"  {task_type}: {count}")
    
    print(f"\nTotal subsampled tasks: {len(subsampled_tasks)}")

if __name__ == "__main__":
    input_path = "/shared/tsqa/CaTSBench/all_questions/tasks.json"
    hard_questions_path = "/shared/tsqa/CaTSBench/hard_questions/tasks.json"
    output_path = "subsampled_tasks.json"
    
    subsample_tasks(input_path, output_path, hard_questions_file=None, min_samples=300)
    print(f"\nDone! Subsampled tasks saved to {output_path}")
