#!/usr/bin/env python
"""
Script to explore the S1.1K dataset structure.
"""

import json
from datasets import load_dataset
import config

def main():
    """Explore the dataset structure."""
    # Load the dataset
    print("Loading dataset...")
    dataset = load_dataset(config.DATASET_NAME, config.DATASET_SUBSET)
    
    # Print dataset info
    print("\nDataset info:")
    print(f"Number of examples: {len(dataset['train'])}")
    print(f"Features: {dataset['train'].features}")
    
    # Print a sample example
    print("\nSample example:")
    example = dataset['train'][0]
    
    # Print all fields in the example
    for key, value in example.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"{key}: {value[:100]}...")
        else:
            print(f"{key}: {value}")
    
    # Count examples with different CoT fields
    print("\nCounting examples with different CoT fields:")
    cot_fields = ["gemini_cot", "deepseek_cot", "claude_cot", "gpt4_cot"]
    for field in cot_fields:
        count = sum(1 for example in dataset['train'] if field in example and example[field] and example[field].strip())
        print(f"{field}: {count} examples")
    
    # Count examples with solution field
    solution_count = sum(1 for example in dataset['train'] if "solution" in example and example["solution"] and example["solution"].strip())
    print(f"solution: {solution_count} examples")
    
    # Count examples with answer field
    answer_count = sum(1 for example in dataset['train'] if "answer" in example and example["answer"] and example["answer"].strip())
    print(f"answer: {answer_count} examples")
    
    # Save a few examples to a JSON file for inspection
    print("\nSaving sample examples to sample_examples.json...")
    sample_examples = [dataset['train'][i] for i in range(min(5, len(dataset['train'])))]
    with open("sample_examples.json", "w") as f:
        json.dump(sample_examples, f, indent=2)
    
    print("Exploration complete!")

if __name__ == "__main__":
    main() 