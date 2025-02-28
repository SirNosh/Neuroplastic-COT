#!/usr/bin/env python
"""
Main script for finetuning Qwen 2.5 3B on the S1.1K Chain of Thought dataset.
"""

import os
import sys
import logging
from datetime import datetime
import random
import numpy as np
import torch
import wandb
from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
    HfArgumentParser,
)

import config
from data_utils import get_dataloaders
from model_utils import load_model, load_tokenizer, get_training_arguments
from neuroplasticity import get_neuroplasticity_trainer


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to train the model.
    """
    # Set random seed for reproducibility
    set_seed(config.SEED)
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    
    # Initialize wandb if enabled
    if config.USE_WANDB:
        wandb.init(
            project=config.WANDB_PROJECT,
            config=vars(config),
            name=f"qwen-2.5-3b-cot-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        )
    
    # Load tokenizer and model
    logger.info(f"Loading tokenizer from {config.MODEL_NAME}")
    tokenizer = load_tokenizer(config.MODEL_NAME)
    
    logger.info(f"Loading model from {config.MODEL_NAME}")
    model = load_model(config.MODEL_NAME)
    
    # Log number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params / all_params:.2%} of {all_params:,})")
    
    # Prepare dataloaders
    logger.info("Preparing dataloaders")
    train_dataloader, eval_dataloader = get_dataloaders(
        tokenizer=tokenizer,
        dataset_name=config.DATASET_NAME,
        dataset_subset=config.DATASET_SUBSET,
        max_length=config.MAX_LENGTH,
        batch_size=config.BATCH_SIZE,
        prompt_template=config.PROMPT_TEMPLATE,
    )
    
    # Get training arguments
    training_args = get_training_arguments()
    
    # Create trainer
    logger.info("Creating trainer")
    
    if config.USE_NEUROPLASTICITY:
        logger.info("Using neuroplasticity techniques for training")
        
        # Configure ALR
        alr_config = None
        if config.USE_ALR:
            alr_config = {
                "base_lr": config.ALR_BASE_LR,
                "max_lr_factor": config.ALR_MAX_FACTOR,
                "complexity_measure": config.ALR_COMPLEXITY_MEASURE,
            }
            
        # Configure EWC
        ewc_config = None
        if config.USE_EWC:
            ewc_config = {
                "importance_factor": config.EWC_IMPORTANCE_FACTOR,
                "fisher_sample_size": config.EWC_FISHER_SAMPLE_SIZE,
                "fisher_update_freq": config.EWC_FISHER_UPDATE_FREQ,
            }
            
        # Configure Hebbian Learning
        hebbian_config = None
        if config.USE_HEBBIAN:
            hebbian_config = {
                "hebbian_factor": config.HEBBIAN_FACTOR,
                "activation_threshold": config.HEBBIAN_ACTIVATION_THRESHOLD,
                "target_layers": config.HEBBIAN_TARGET_LAYERS,
            }
            
        # Create neuroplasticity trainer
        trainer = get_neuroplasticity_trainer(
            model=model,
            args=training_args,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            use_alr=config.USE_ALR,
            use_ewc=config.USE_EWC,
            use_hebbian=config.USE_HEBBIAN,
            alr_config=alr_config,
            ewc_config=ewc_config,
            hebbian_config=hebbian_config,
        )
    else:
        # Use standard Trainer if neuroplasticity is disabled
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataloader.dataset,
            eval_dataset=eval_dataloader.dataset,
            data_collator=lambda x: x,  # Identity function as data is already processed
        )
    
    # Train the model
    logger.info("Starting training")
    trainer.train()
    
    # Save the final model and tokenizer
    logger.info(f"Saving model to {config.OUTPUT_DIR}")
    trainer.save_model(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)
    
    logger.info("Training completed")


if __name__ == "__main__":
    main() 