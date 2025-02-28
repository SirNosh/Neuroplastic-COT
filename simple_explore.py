#!/usr/bin/env python
"""
Simple script to explore the S1.1K dataset structure.
"""

import json
from datasets import load_dataset

def main():
    """Explore the dataset structure."""
    # Load the dataset
    print("Loading dataset...")
    dataset = load_dataset("simplescaling/s1K-1.1")
    
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
    
    # Save a few examples to a JSON file for inspection
    print("\nSaving sample examples to sample_examples.json...")
    sample_examples = [dataset['train'][i] for i in range(min(5, len(dataset['train'])))]
    with open("sample_examples.json", "w") as f:
        json.dump(sample_examples, f, indent=2)
    
    print("Exploration complete!")

if __name__ == "__main__":
    main() 