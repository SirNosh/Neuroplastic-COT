#!/usr/bin/env python
"""
Script for evaluating the finetuned Qwen 2.5 3B model on the S1.1K Chain of Thought dataset.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Optional, Union

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from peft import PeftModel

import config
from data_utils import load_s1_1k_dataset


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate finetuned Qwen 2.5 3B model")
    parser.add_argument(
        "--model_path",
        type=str,
        default=config.OUTPUT_DIR,
        help="Path to the finetuned model",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=config.MODEL_NAME,
        help="Base model name or path",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_results.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help="Number of examples to evaluate (None for all)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p for generation",
    )
    return parser.parse_args()


def load_finetuned_model(args):
    """Load the finetuned model."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    # Load LoRA weights if using PEFT
    if os.path.exists(os.path.join(args.model_path, "adapter_config.json")):
        logger.info("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(model, args.model_path)
    
    # Set generation config
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.temperature > 0,
    )
    
    return model, tokenizer, generation_config


def evaluate_model(model, tokenizer, generation_config, args):
    """Evaluate the model on the test set."""
    # Load test dataset
    dataset = load_dataset(config.DATASET_NAME, config.DATASET_SUBSET)
    
    # Create a test split
    dataset = dataset.shuffle(seed=config.SEED)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=config.SEED)
    dataset = split_dataset["test"]
    
    # Limit number of examples if specified
    if args.num_examples is not None:
        dataset = dataset.select(range(min(args.num_examples, len(dataset))))
    
    results = []
    
    # Evaluate each example
    for i, example in enumerate(dataset):
        logger.info(f"Evaluating example {i+1}/{len(dataset)}")
        
        # Format the input using the prompt template
        prompt = config.PROMPT_TEMPLATE.format(problem=example["question"])
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config,
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the model's answer (everything after the prompt)
        model_output = generated_text[len(prompt):]
        
        # Get reference CoT if available
        reference_cot = None
        cot_fields = ["gemini_cot", "deepseek_cot", "claude_cot", "gpt4_cot"]
        for field in cot_fields:
            if field in example and example[field] and example[field].strip():
                reference_cot = example[field]
                break
        
        # If no CoT field is available, use the solution directly
        if not reference_cot and "solution" in example and example["solution"]:
            reference_cot = example["solution"]
        
        # Store results
        results.append({
            "question": example["question"],
            "reference_cot": reference_cot,
            "reference_answer": example.get("answer", ""),
            "model_output": model_output,
        })
    
    return results


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer, generation_config = load_finetuned_model(args)
    
    # Evaluate model
    logger.info("Starting evaluation...")
    results = evaluate_model(model, tokenizer, generation_config, args)
    
    # Save results
    logger.info(f"Saving results to {args.output_file}...")
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main() 