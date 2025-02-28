"""
Demo script to showcase neuroplasticity techniques for Chain of Thought reasoning.

This script demonstrates the effects of Adaptive Learning Rate (ALR),
Elastic Weight Consolidation (EWC), and Hebbian Learning on a small subset
of the S1.1K dataset.
"""

import argparse
import logging
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

import config
from data_utils import get_dataloaders
from model_utils import load_tokenizer, load_model, get_training_arguments
from neuroplasticity import (
    AdaptiveReasoningRate,
    ElasticWeightConsolidation,
    HebbianLearning,
    get_neuroplasticity_trainer,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Neuroplasticity demo for CoT reasoning")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./neuroplasticity_demo_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=10,
        help="Number of examples to use for the demo",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--use_alr",
        action="store_true",
        help="Use Adaptive Learning Rate",
    )
    parser.add_argument(
        "--use_ewc",
        action="store_true",
        help="Use Elastic Weight Consolidation",
    )
    parser.add_argument(
        "--use_hebbian",
        action="store_true",
        help="Use Hebbian Learning",
    )
    parser.add_argument(
        "--use_all",
        action="store_true",
        help="Use all neuroplasticity techniques",
    )
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_small_dataset(dataloader, num_examples):
    """Create a small dataset for the demo."""
    small_dataset = []
    for batch in dataloader:
        small_dataset.append(batch)
        if len(small_dataset) * dataloader.batch_size >= num_examples:
            break
    return small_dataset


def visualize_complexity(alr, dataset, output_dir):
    """Visualize the complexity of examples in the dataset."""
    complexities = []
    
    for batch in dataset:
        complexity = alr.measure_complexity(batch)
        complexities.extend(complexity.cpu().numpy())
    
    plt.figure(figsize=(10, 6))
    plt.hist(complexities, bins=20, alpha=0.7)
    plt.title("Distribution of Reasoning Complexity")
    plt.xlabel("Complexity Score")
    plt.ylabel("Number of Examples")
    plt.grid(True, alpha=0.3)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "complexity_distribution.png"))
    plt.close()
    
    logger.info(f"Complexity visualization saved to {output_dir}/complexity_distribution.png")


def visualize_learning_rates(alr, dataset, output_dir):
    """Visualize the learning rates for examples in the dataset."""
    lr_multipliers = []
    
    for batch in dataset:
        multiplier = alr.get_lr_multipliers(batch)
        lr_multipliers.extend(multiplier.cpu().numpy())
    
    plt.figure(figsize=(10, 6))
    plt.hist(lr_multipliers, bins=20, alpha=0.7)
    plt.title("Distribution of Learning Rate Multipliers")
    plt.xlabel("Learning Rate Multiplier")
    plt.ylabel("Number of Examples")
    plt.grid(True, alpha=0.3)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "lr_multiplier_distribution.png"))
    plt.close()
    
    logger.info(f"Learning rate visualization saved to {output_dir}/lr_multiplier_distribution.png")


def visualize_ewc_importance(ewc, output_dir):
    """Visualize the importance of parameters according to EWC."""
    # Get the average Fisher information for each parameter
    avg_fisher = {}
    for name, fisher in ewc.fisher_information.items():
        avg_fisher[name] = fisher.mean().item()
    
    # Sort parameters by importance
    sorted_params = sorted(avg_fisher.items(), key=lambda x: x[1], reverse=True)
    
    # Plot top 20 most important parameters
    top_params = sorted_params[:20]
    param_names = [name.split('.')[-2] + '.' + name.split('.')[-1] for name, _ in top_params]
    importances = [imp for _, imp in top_params]
    
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(param_names)), importances, alpha=0.7)
    plt.xticks(range(len(param_names)), param_names, rotation=90)
    plt.title("Top 20 Most Important Parameters (EWC)")
    plt.xlabel("Parameter")
    plt.ylabel("Average Fisher Information")
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "ewc_parameter_importance.png"))
    plt.close()
    
    logger.info(f"EWC importance visualization saved to {output_dir}/ewc_parameter_importance.png")


def train_with_neuroplasticity(args):
    """Train a model with neuroplasticity techniques."""
    # Set random seed
    set_seed(args.seed)
    
    # Determine which techniques to use
    use_alr = args.use_alr or args.use_all
    use_ewc = args.use_ewc or args.use_all
    use_hebbian = args.use_hebbian or args.use_all
    
    if not (use_alr or use_ewc or use_hebbian):
        logger.info("No neuroplasticity techniques selected. Using all techniques.")
        use_alr = use_ewc = use_hebbian = True
    
    # Load tokenizer and model
    logger.info(f"Loading tokenizer from {config.MODEL_NAME}")
    tokenizer = load_tokenizer(config.MODEL_NAME)
    
    logger.info(f"Loading model from {config.MODEL_NAME}")
    model = load_model(config.MODEL_NAME)
    
    # Log number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {trainable_params}")
    
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
    
    # Create small dataset for the demo
    logger.info(f"Creating small dataset with {args.num_examples} examples")
    small_train_dataset = create_small_dataset(train_dataloader, args.num_examples)
    small_eval_dataset = create_small_dataset(eval_dataloader, args.num_examples // 2)
    
    # Initialize neuroplasticity components
    if use_alr:
        logger.info("Initializing Adaptive Learning Rate")
        alr = AdaptiveReasoningRate(
            base_lr=config.ALR_BASE_LR,
            max_lr_factor=config.ALR_MAX_FACTOR,
            complexity_measure=config.ALR_COMPLEXITY_MEASURE,
        )
        
        # Visualize complexity and learning rates
        logger.info("Visualizing complexity and learning rates")
        visualize_complexity(alr, small_train_dataset, args.output_dir)
        visualize_learning_rates(alr, small_train_dataset, args.output_dir)
    
    if use_ewc:
        logger.info("Initializing Elastic Weight Consolidation")
        ewc = ElasticWeightConsolidation(
            model=model,
            importance_factor=config.EWC_IMPORTANCE_FACTOR,
            fisher_sample_size=config.EWC_FISHER_SAMPLE_SIZE,
            fisher_update_freq=config.EWC_FISHER_UPDATE_FREQ,
        )
        
        # Update Fisher information
        logger.info("Updating Fisher information")
        ewc.update_fisher_information(small_train_dataset)
        
        # Visualize parameter importance
        logger.info("Visualizing parameter importance")
        visualize_ewc_importance(ewc, args.output_dir)
    
    if use_hebbian:
        logger.info("Initializing Hebbian Learning")
        hebbian = HebbianLearning(
            model=model,
            hebbian_factor=config.HEBBIAN_FACTOR,
            activation_threshold=config.HEBBIAN_ACTIVATION_THRESHOLD,
            target_layers=config.HEBBIAN_TARGET_LAYERS,
        )
    
    # Get training arguments
    training_args = get_training_arguments()
    training_args.num_train_epochs = args.num_epochs
    training_args.output_dir = args.output_dir
    
    # Configure neuroplasticity components for the trainer
    alr_config = None
    if use_alr:
        alr_config = {
            "base_lr": config.ALR_BASE_LR,
            "max_lr_factor": config.ALR_MAX_FACTOR,
            "complexity_measure": config.ALR_COMPLEXITY_MEASURE,
        }
        
    ewc_config = None
    if use_ewc:
        ewc_config = {
            "importance_factor": config.EWC_IMPORTANCE_FACTOR,
            "fisher_sample_size": config.EWC_FISHER_SAMPLE_SIZE,
            "fisher_update_freq": config.EWC_FISHER_UPDATE_FREQ,
        }
        
    hebbian_config = None
    if use_hebbian:
        hebbian_config = {
            "hebbian_factor": config.HEBBIAN_FACTOR,
            "activation_threshold": config.HEBBIAN_ACTIVATION_THRESHOLD,
            "target_layers": config.HEBBIAN_TARGET_LAYERS,
        }
    
    # Create neuroplasticity trainer
    logger.info("Creating neuroplasticity trainer")
    trainer = get_neuroplasticity_trainer(
        model=model,
        args=training_args,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        use_alr=use_alr,
        use_ewc=use_ewc,
        use_hebbian=use_hebbian,
        alr_config=alr_config,
        ewc_config=ewc_config,
        hebbian_config=hebbian_config,
    )
    
    # Train the model
    logger.info("Starting training")
    trainer.train()
    
    # Save the final model and tokenizer
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Training completed")
    
    # Log which techniques were used
    techniques_used = []
    if use_alr:
        techniques_used.append("Adaptive Learning Rate")
    if use_ewc:
        techniques_used.append("Elastic Weight Consolidation")
    if use_hebbian:
        techniques_used.append("Hebbian Learning")
    
    logger.info(f"Neuroplasticity techniques used: {', '.join(techniques_used)}")


if __name__ == "__main__":
    args = parse_args()
    train_with_neuroplasticity(args) 