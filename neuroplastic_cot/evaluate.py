"""
Evaluation script for the finetuned Qwen 2.5 3B model on the S1.1K Chain of Thought dataset.
"""

import argparse
import json
import logging
import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

from neuroplastic_cot.config import (
    MODEL_NAME,
    DATASET_NAME,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate finetuned model on S1.1K dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to finetuned model")
    parser.add_argument("--base_model", type=str, default=MODEL_NAME, help="Base model name")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", help="Output file for results")
    parser.add_argument("--num_examples", type=int, default=None, help="Number of examples to evaluate (None for all)")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=TOP_P, help="Top-p for generation")
    
    return parser.parse_args()


def load_finetuned_model(model_path, base_model):
    """
    Load the finetuned model and tokenizer.
    
    Args:
        model_path: Path to the finetuned model
        base_model: Name of the base model
        
    Returns:
        Tokenizer and model
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Check if model_path contains adapter weights (LoRA)
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        # Load LoRA weights
        model = PeftModel.from_pretrained(model, model_path)
        logging.info("Loaded LoRA weights")
    else:
        # Load full model weights
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        logging.info("Loaded full model weights")
    
    return tokenizer, model


def evaluate_model(model, tokenizer, test_dataset, args):
    """
    Evaluate the model on the test set.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for the model
        test_dataset: Test dataset
        args: Command line arguments
        
    Returns:
        Dictionary of evaluation results
    """
    results = []
    
    # Limit number of examples if specified
    if args.num_examples is not None:
        test_dataset = test_dataset.select(range(min(args.num_examples, len(test_dataset))))
    
    # Evaluate on each example
    for example in tqdm(test_dataset, desc="Evaluating"):
        # Format input
        input_text = f"Question: {example['question']}\nAnswer with step-by-step reasoning:"
        
        # Tokenize input
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        
        # Generate response
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.temperature > 0,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode response
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        # Get reference answer
        if "cot_gpt4" in example and example["cot_gpt4"]:
            reference = example["cot_gpt4"]
        elif "cot_palm" in example and example["cot_palm"]:
            reference = example["cot_palm"]
        elif "cot_claude" in example and example["cot_claude"]:
            reference = example["cot_claude"]
        else:
            reference = example["answer"]
        
        # Store result
        results.append({
            "question": example["question"],
            "generated_cot": response,
            "reference_cot": reference,
            "answer": example["answer"]
        })
    
    return results


def main():
    """Main function for evaluation."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Load model and tokenizer
    logging.info(f"Loading model from {args.model_path}")
    tokenizer, model = load_finetuned_model(args.model_path, args.base_model)
    
    # Load test dataset
    logging.info(f"Loading dataset {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)
    test_dataset = dataset["test"]
    
    # Evaluate model
    logging.info("Evaluating model")
    results = evaluate_model(model, tokenizer, test_dataset, args)
    
    # Save results
    logging.info(f"Saving results to {args.output_file}")
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logging.info("Evaluation complete")


if __name__ == "__main__":
    main() 