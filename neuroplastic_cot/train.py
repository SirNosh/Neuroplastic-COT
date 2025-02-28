"""
Main script for fine-tuning the Qwen 2.5 3B model on the S1.1K Chain of Thought dataset.
"""

import argparse
import logging
import os
import random
import sys
import torch
import wandb
from transformers import set_seed

from neuroplastic_cot.model_utils import load_model, load_tokenizer, get_training_arguments
from neuroplastic_cot.data_utils import get_dataloaders
from neuroplastic_cot.neuroplasticity import get_neuroplasticity_trainer
from neuroplastic_cot.config import (
    MODEL_NAME, 
    LORA_ENABLED, 
    LORA_R, 
    LORA_ALPHA, 
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
    WANDB_ENABLED,
    WANDB_PROJECT,
    WANDB_NAME,
    SEED,
    USE_ALR,
    USE_EWC,
    USE_HEBBIAN,
    ALR_CONFIG,
    EWC_CONFIG,
    HEBBIAN_CONFIG
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune Qwen 2.5 3B on S1.1K Chain of Thought dataset")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--wandb", action="store_true", default=WANDB_ENABLED, help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default=WANDB_PROJECT, help="W&B project name")
    parser.add_argument("--wandb_name", type=str, default=WANDB_NAME, help="W&B run name")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    
    # LoRA arguments
    parser.add_argument("--lora", action="store_true", default=LORA_ENABLED, help="Use LoRA")
    parser.add_argument("--lora_r", type=int, default=LORA_R, help="LoRA r")
    parser.add_argument("--lora_alpha", type=int, default=LORA_ALPHA, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=LORA_DROPOUT, help="LoRA dropout")
    
    # Neuroplasticity arguments
    parser.add_argument("--use_alr", action="store_true", default=USE_ALR, help="Use Adaptive Learning Rate")
    parser.add_argument("--use_ewc", action="store_true", default=USE_EWC, help="Use Elastic Weight Consolidation")
    parser.add_argument("--use_hebbian", action="store_true", default=USE_HEBBIAN, help="Use Hebbian Learning")
    
    return parser.parse_args()


def main():
    """Main function for fine-tuning."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.output_dir, "train.log"))
        ]
    )
    
    # Set random seed
    set_seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb
    if args.wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name)
    
    # Load tokenizer
    tokenizer = load_tokenizer(args.model_name)
    
    # Load model
    model = load_model(
        model_name=args.model_name,
        use_lora=args.lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=LORA_TARGET_MODULES
    )
    
    # Log number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Number of trainable parameters: {trainable_params}")
    
    # Prepare dataloaders
    train_dataloader, eval_dataloader = get_dataloaders(tokenizer)
    
    # Get training arguments
    training_args = get_training_arguments(output_dir=args.output_dir)
    
    # Create trainer
    trainer = get_neuroplasticity_trainer(
        model=model,
        args=training_args,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        use_alr=args.use_alr,
        use_ewc=args.use_ewc,
        use_hebbian=args.use_hebbian,
        alr_config=ALR_CONFIG,
        ewc_config=EWC_CONFIG,
        hebbian_config=HEBBIAN_CONFIG
    )
    
    # Train model
    logging.info("Starting training...")
    trainer.train()
    
    # Save final model and tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logging.info(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main() 