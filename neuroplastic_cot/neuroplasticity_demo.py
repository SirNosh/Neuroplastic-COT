"""
Demo script for showcasing neuroplasticity techniques for Chain of Thought reasoning.

This script demonstrates the effects of:
1. Adaptive Learning Rate (ALR) - Shows how learning rates adapt to reasoning complexity
2. Elastic Weight Consolidation (EWC) - Visualizes parameter importance for reasoning
3. Hebbian Learning - Demonstrates strengthening of reasoning pathways
"""

import argparse
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import Dataset
import os
import random
from tqdm import tqdm

# Import neuroplasticity components
from neuroplastic_cot.neuroplasticity import (
    AdaptiveReasoningRate,
    ElasticWeightConsolidation,
    HebbianLearning,
    NeuroplasticityTrainer
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Demo for neuroplasticity techniques")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct", 
                        help="Model to use for the demo")
    parser.add_argument("--use_alr", action="store_true", help="Use Adaptive Learning Rate")
    parser.add_argument("--use_ewc", action="store_true", help="Use Elastic Weight Consolidation")
    parser.add_argument("--use_hebbian", action="store_true", help="Use Hebbian Learning")
    parser.add_argument("--use_all", action="store_true", help="Use all neuroplasticity techniques")
    parser.add_argument("--num_examples", type=int, default=10, 
                        help="Number of examples to use for the demo")
    parser.add_argument("--output_dir", type=str, default="neuroplasticity_demo_results",
                        help="Directory to save demo results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_demo_dataset(num_examples=10):
    """
    Create a small dataset for the demo with varying complexity.
    
    Returns:
        A small dataset with mathematical problems of varying complexity
    """
    # Simple problems
    simple_problems = [
        {
            "question": "What is 5 + 7?",
            "answer": "12",
            "cot": "To find the sum of 5 and 7, I add them together: 5 + 7 = 12. The answer is 12."
        },
        {
            "question": "What is 8 × 4?",
            "answer": "32",
            "cot": "To find the product of 8 and 4, I multiply them: 8 × 4 = 32. The answer is 32."
        },
        {
            "question": "What is 20 - 13?",
            "answer": "7",
            "cot": "To find the difference between 20 and 13, I subtract: 20 - 13 = 7. The answer is 7."
        },
    ]
    
    # Medium complexity problems
    medium_problems = [
        {
            "question": "If a rectangle has a length of 12 cm and a width of 5 cm, what is its area?",
            "answer": "60 square cm",
            "cot": "To find the area of a rectangle, I multiply its length by its width. Area = length × width = 12 cm × 5 cm = 60 square cm. The answer is 60 square cm."
        },
        {
            "question": "A train travels at 60 km/h. How far will it travel in 2.5 hours?",
            "answer": "150 km",
            "cot": "To find the distance traveled, I multiply the speed by the time. Distance = speed × time = 60 km/h × 2.5 h = 150 km. The answer is 150 km."
        },
        {
            "question": "If 3x + 7 = 22, what is the value of x?",
            "answer": "5",
            "cot": "To solve for x, I need to isolate it. Starting with 3x + 7 = 22, I subtract 7 from both sides: 3x = 15. Then I divide both sides by 3: x = 5. The answer is 5."
        },
    ]
    
    # Complex problems
    complex_problems = [
        {
            "question": "The sum of three consecutive integers is 51. What are these integers?",
            "answer": "16, 17, 18",
            "cot": "Let's call the first integer n. Then the three consecutive integers are n, n+1, and n+2. According to the problem, their sum is 51. So, n + (n+1) + (n+2) = 51. Simplifying: 3n + 3 = 51. Subtracting 3 from both sides: 3n = 48. Dividing both sides by 3: n = 16. Therefore, the three consecutive integers are 16, 17, and 18. The answer is 16, 17, 18."
        },
        {
            "question": "A mixture of 40 liters contains water and alcohol in the ratio 3:1. How much alcohol should be added to make the ratio 1:1?",
            "answer": "20 liters",
            "cot": "The ratio of water to alcohol is 3:1, so out of 40 liters, water is 3/4 × 40 = 30 liters, and alcohol is 1/4 × 40 = 10 liters. To make the ratio 1:1, the amount of water and alcohol should be equal. Since we have 30 liters of water, we need 30 liters of alcohol. We already have 10 liters of alcohol, so we need to add 30 - 10 = 20 more liters of alcohol. The answer is 20 liters."
        },
        {
            "question": "The probability of getting at least one head when flipping a fair coin n times is 0.875. What is the value of n?",
            "answer": "3",
            "cot": "The probability of getting at least one head is 1 minus the probability of getting no heads. The probability of getting no heads in n flips is (1/2)^n. So, 1 - (1/2)^n = 0.875. Solving for (1/2)^n: (1/2)^n = 1 - 0.875 = 0.125 = 1/8. Since (1/2)^3 = 1/8, n = 3. The answer is 3."
        },
        {
            "question": "Find the derivative of f(x) = x^3 - 4x^2 + 5x - 2.",
            "answer": "3x^2 - 8x + 5",
            "cot": "To find the derivative of f(x) = x^3 - 4x^2 + 5x - 2, I'll use the power rule and linearity of differentiation. The power rule states that d/dx(x^n) = n*x^(n-1). So, d/dx(x^3) = 3x^2, d/dx(-4x^2) = -8x, d/dx(5x) = 5, and d/dx(-2) = 0. Adding these terms: f'(x) = 3x^2 - 8x + 5. The answer is 3x^2 - 8x + 5."
        },
    ]
    
    # Combine problems based on the number of examples needed
    all_problems = simple_problems + medium_problems + complex_problems
    selected_problems = all_problems[:min(num_examples, len(all_problems))]
    
    # Create a dataset
    dataset = Dataset.from_dict({
        "question": [p["question"] for p in selected_problems],
        "answer": [p["answer"] for p in selected_problems],
        "cot": [p["cot"] for p in selected_problems],
        "complexity": [1] * len(simple_problems) + [2] * len(medium_problems) + [3] * len(complex_problems[:min(num_examples - len(simple_problems) - len(medium_problems), len(complex_problems))])
    })
    
    return dataset


def visualize_complexity_vs_learning_rate(alr, dataset, output_dir):
    """
    Visualize how learning rate adapts to problem complexity.
    
    Args:
        alr: AdaptiveReasoningRate instance
        dataset: Dataset with problems of varying complexity
        output_dir: Directory to save the visualization
    """
    # Create a tokenizer for processing the examples
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Process examples
    complexities = []
    lr_multipliers = []
    
    for i in range(len(dataset)):
        # Prepare input
        prompt = f"Question: {dataset[i]['question']}\nAnswer with step-by-step reasoning:"
        target = dataset[i]['cot']
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        targets = tokenizer(target, return_tensors="pt", padding=True, truncation=True)
        
        # Create batch with labels
        batch = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": targets["input_ids"]
        }
        
        # Get learning rate multiplier
        lr_multiplier = alr.get_lr_multipliers(batch).item()
        
        # Store data for visualization
        complexities.append(dataset[i]['complexity'])
        lr_multipliers.append(lr_multiplier)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(complexities, lr_multipliers, c=complexities, cmap='viridis', s=100, alpha=0.7)
    plt.colorbar(label='Problem Complexity')
    plt.xlabel('Problem Complexity')
    plt.ylabel('Learning Rate Multiplier')
    plt.title('Adaptive Learning Rate Based on Problem Complexity')
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'complexity_vs_learning_rate.png'))
    plt.close()
    
    logging.info(f"Saved complexity vs. learning rate visualization to {output_dir}")


def visualize_ewc_parameter_importance(ewc, output_dir):
    """
    Visualize parameter importance as determined by EWC.
    
    Args:
        ewc: ElasticWeightConsolidation instance
        output_dir: Directory to save the visualization
    """
    # Get parameter importance from Fisher information
    param_names = []
    importance_values = []
    
    for name, fisher in ewc.fisher_information.items():
        if 'layer' in name and 'weight' in name:
            # Take the mean importance for each layer's weight
            param_names.append(name.split('.')[-3])  # Extract layer name
            importance_values.append(fisher.abs().mean().item())
    
    # Sort by importance
    sorted_indices = np.argsort(importance_values)
    param_names = [param_names[i] for i in sorted_indices[-20:]]  # Top 20 layers
    importance_values = [importance_values[i] for i in sorted_indices[-20:]]
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    bars = plt.barh(param_names, importance_values, color='skyblue')
    plt.xlabel('Mean Parameter Importance')
    plt.ylabel('Layer')
    plt.title('EWC Parameter Importance by Layer')
    plt.grid(True, alpha=0.3)
    
    # Add values to the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                 ha='left', va='center')
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'ewc_parameter_importance.png'))
    plt.close()
    
    logging.info(f"Saved EWC parameter importance visualization to {output_dir}")


def train_with_neuroplasticity(args):
    """
    Train a model with neuroplasticity techniques on a small dataset.
    
    Args:
        args: Command line arguments
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, 'demo.log'))
        ]
    )
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataset
    logging.info("Creating demo dataset...")
    dataset = create_demo_dataset(args.num_examples)
    
    # Load model and tokenizer
    logging.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Use a tiny model for the demo to make it run faster
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Determine which neuroplasticity techniques to use
    use_alr = args.use_alr or args.use_all
    use_ewc = args.use_ewc or args.use_all
    use_hebbian = args.use_hebbian or args.use_all
    
    # Initialize neuroplasticity components
    logging.info("Initializing neuroplasticity components...")
    
    if use_alr:
        alr = AdaptiveReasoningRate(
            base_lr=5e-5,
            max_lr_factor=3.0,
            complexity_measure="token_length"
        )
        logging.info("Initialized Adaptive Learning Rate")
        
        # Visualize complexity vs. learning rate
        logging.info("Generating ALR visualization...")
        visualize_complexity_vs_learning_rate(alr, dataset, args.output_dir)
    
    if use_ewc:
        ewc = ElasticWeightConsolidation(
            model=model,
            importance_factor=1000.0,
            fisher_sample_size=min(args.num_examples, 5),
            fisher_update_freq=1
        )
        logging.info("Initialized Elastic Weight Consolidation")
    
    if use_hebbian:
        hebbian = HebbianLearning(
            model=model,
            hebbian_factor=0.01,
            activation_threshold=0.5,
            target_layers=["mlp", "attention"]
        )
        logging.info("Initialized Hebbian Learning")
    
    # Prepare dataset for training
    def tokenize_function(examples):
        prompts = [f"Question: {q}\nAnswer with step-by-step reasoning:" for q in examples["question"]]
        inputs = tokenizer(prompts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        targets = tokenizer(examples["cot"], padding="max_length", truncation=True, max_length=384, return_tensors="pt")
        
        # Create labels
        inputs["labels"] = targets["input_ids"]
        
        return inputs
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        num_train_epochs=1,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=1,
        save_steps=10,
        save_total_limit=2,
        remove_unused_columns=False,
    )
    
    # Create trainer with neuroplasticity components
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_dataset,
        "use_alr": use_alr,
        "use_ewc": use_ewc,
        "use_hebbian": use_hebbian,
    }
    
    if use_alr:
        trainer_kwargs["alr_config"] = {"base_lr": 5e-5, "max_lr_factor": 3.0}
    
    if use_ewc:
        trainer_kwargs["ewc_config"] = {"importance_factor": 1000.0, "fisher_sample_size": min(args.num_examples, 5)}
        
        # Update Fisher information with the dataset
        logging.info("Calculating initial Fisher information...")
        ewc.update_fisher_information(torch.utils.data.DataLoader(tokenized_dataset, batch_size=2))
        
        # Visualize parameter importance
        logging.info("Generating EWC visualization...")
        visualize_ewc_parameter_importance(ewc, args.output_dir)
    
    if use_hebbian:
        trainer_kwargs["hebbian_config"] = {"hebbian_factor": 0.01, "activation_threshold": 0.5}
    
    trainer = NeuroplasticityTrainer(**trainer_kwargs)
    
    # Train for a few steps
    logging.info("Training model with neuroplasticity techniques...")
    trainer.train()
    
    logging.info(f"Demo completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    train_with_neuroplasticity(args) 