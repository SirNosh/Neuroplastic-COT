"""
Avalanche-based continual learning training script.

Integrates the Avalanche library (industry-standard continual learning framework)
with our canonical EWC and SI implementations for direct comparison.

Reference: Lomonaco et al., "Avalanche: An end-to-end library for continual learning" (2021)
"""

import argparse
import logging
import os
import json
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple

# Avalanche imports
from avalanche.benchmarks.utils import AvalancheTensorDataset
from avalanche.benchmarks.generators import dataset_benchmark
from avalanche.training.plugins import EWCPlugin, SynapticIntelligencePlugin
from avalanche.training.supervised import Naive
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.templates import SupervisedTemplate

from dataset_loaders import ContinualDatasetLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContinualLearningDataset(Dataset):
    """Dataset wrapper for continual learning phases."""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Format: Question + Answer
        if "question" in item and "answer" in item:
            text = f"Question: {item['question']}\nAnswer: {item['answer']}"
        else:
            text = item.get("problem", "") + "\n" + item.get("solution", "")

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0).clone()
        }


class AvalancheCompatibleModel(torch.nn.Module):
    """
    Wrapper to make HuggingFace models compatible with Avalanche.

    Avalanche expects:
    - forward(x) returns logits
    - Standard classification-style interface
    """

    def __init__(self, hf_model, tokenizer):
        super().__init__()
        self.model = hf_model
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass compatible with both HF and Avalanche."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        # Return in Avalanche-compatible format
        if labels is not None:
            return outputs.logits, outputs.loss
        return outputs.logits


def setup_model(model_name: str, device: str) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Setup model with LoRA and 4-bit quantization.

    Args:
        model_name: HuggingFace model name
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_name}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # Base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Prepare for k-bit training
    base_model = prepare_model_for_kbit_training(base_model)

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Apply LoRA
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # Wrap for Avalanche compatibility
    avalanche_model = AvalancheCompatibleModel(model, tokenizer)

    return avalanche_model, tokenizer


def create_avalanche_benchmark(
    phases_data: Dict[str, List[Dict]],
    tokenizer,
    max_length: int = 512
):
    """
    Create Avalanche benchmark from continual learning phases.

    Args:
        phases_data: Dictionary mapping phase names to data lists
        tokenizer: Tokenizer for encoding
        max_length: Maximum sequence length

    Returns:
        Avalanche benchmark
    """
    logger.info("Creating Avalanche benchmark...")

    train_datasets = []
    test_datasets = []

    for phase_name, data in phases_data.items():
        logger.info(f"Processing phase: {phase_name} ({len(data)} samples)")

        # Split train/test (90/10)
        split_idx = int(0.9 * len(data))
        train_data = data[:split_idx]
        test_data = data[split_idx:]

        # Create datasets
        train_ds = ContinualLearningDataset(train_data, tokenizer, max_length)
        test_ds = ContinualLearningDataset(test_data, tokenizer, max_length)

        train_datasets.append(train_ds)
        test_datasets.append(test_ds)

    # Create benchmark
    benchmark = dataset_benchmark(
        train_datasets=train_datasets,
        test_datasets=test_datasets
    )

    return benchmark


def train_with_avalanche(
    model,
    tokenizer,
    benchmark,
    strategy_name: str,
    output_dir: str,
    num_epochs: int = 2,
    learning_rate: float = 2e-5,
    ewc_lambda: float = 0.4,
    si_lambda: float = 0.4
):
    """
    Train using Avalanche continual learning strategies.

    Args:
        model: The model to train
        tokenizer: Tokenizer
        benchmark: Avalanche benchmark
        strategy_name: One of ['naive', 'ewc', 'si']
        output_dir: Directory to save results
        num_epochs: Epochs per phase
        learning_rate: Learning rate
        ewc_lambda: EWC regularization strength
        si_lambda: SI regularization strength
    """
    logger.info(f"Training with Avalanche strategy: {strategy_name}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Setup plugins based on strategy
    plugins = []
    if strategy_name == "ewc":
        plugins.append(EWCPlugin(ewc_lambda=ewc_lambda, mode='separate'))
    elif strategy_name == "si":
        plugins.append(SynapticIntelligencePlugin(si_lambda=si_lambda))

    # Setup logging
    loggers = [
        InteractiveLogger(),
        TextLogger(open(os.path.join(output_dir, "log.txt"), "w"))
    ]

    # Create strategy
    # Note: Avalanche's Naive strategy is the base class for continual learning
    strategy = Naive(
        model=model,
        optimizer=optimizer,
        train_mb_size=2,
        train_epochs=num_epochs,
        eval_mb_size=4,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        plugins=plugins,
        evaluator=None,  # We'll do custom evaluation
        eval_every=1
    )

    # Results tracking
    results = {
        "strategy": strategy_name,
        "phases": [],
        "performance_matrix": {},
        "forgetting_scores": {}
    }

    # Training loop across phases
    for phase_idx, experience in enumerate(benchmark.train_stream):
        phase_name = ["arithmetic", "algebra", "geometry"][phase_idx]
        logger.info(f"\n{'='*60}")
        logger.info(f"Phase {phase_idx + 1}: {phase_name}")
        logger.info(f"{'='*60}")

        results["phases"].append(phase_name)

        # Train on current phase
        logger.info(f"Training on {phase_name}...")
        strategy.train(experience)

        # Evaluate on all phases seen so far
        logger.info(f"Evaluating after {phase_name} phase...")
        for eval_idx in range(phase_idx + 1):
            eval_experience = benchmark.test_stream[eval_idx]
            eval_phase_name = ["arithmetic", "algebra", "geometry"][eval_idx]

            # Evaluate
            eval_loss = evaluate_on_experience(strategy.model, eval_experience)

            key = f"after_{phase_name}_test_{eval_phase_name}"
            results["performance_matrix"][key] = eval_loss
            logger.info(f"  {eval_phase_name}: loss = {eval_loss:.4f}")

        # Save checkpoint
        phase_dir = os.path.join(output_dir, f"phase_{phase_idx + 1}_{phase_name}")
        os.makedirs(phase_dir, exist_ok=True)

        # Save model (LoRA adapter only)
        if hasattr(strategy.model, 'model'):
            strategy.model.model.save_pretrained(phase_dir)

        logger.info(f"Checkpoint saved: {phase_dir}")

    # Compute forgetting metrics
    logger.info("\nComputing forgetting metrics...")
    for phase_idx, phase_name in enumerate(results["phases"][:-1]):  # Exclude last phase
        initial_key = f"after_{phase_name}_test_{phase_name}"
        final_key = f"after_{results['phases'][-1]}_test_{phase_name}"

        if initial_key in results["performance_matrix"] and final_key in results["performance_matrix"]:
            initial_loss = results["performance_matrix"][initial_key]
            final_loss = results["performance_matrix"][final_key]

            # For loss: increase = forgetting (lower is better)
            forgetting_pct = ((final_loss - initial_loss) / initial_loss) * 100

            results["forgetting_scores"][phase_name] = {
                "initial_loss": float(initial_loss),
                "final_loss": float(final_loss),
                "forgetting_percentage": float(forgetting_pct)
            }

            logger.info(f"{phase_name}: {forgetting_pct:.2f}% forgetting")

    # Calculate average forgetting
    if results["forgetting_scores"]:
        avg_forgetting = np.mean([
            v["forgetting_percentage"] for v in results["forgetting_scores"].values()
        ])
        results["average_forgetting"] = float(avg_forgetting)
        logger.info(f"\nAverage Forgetting: {avg_forgetting:.2f}%")

    # Save results
    results_path = os.path.join(output_dir, "continual_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved: {results_path}")
    logger.info(f"{'='*60}")

    return results


def evaluate_on_experience(model, experience) -> float:
    """
    Evaluate model on an Avalanche experience.

    Args:
        model: Model to evaluate
        experience: Avalanche experience

    Returns:
        Average loss
    """
    model.eval()
    device = next(model.parameters()).device

    total_loss = 0.0
    total_samples = 0

    # Create dataloader from experience
    dataloader = DataLoader(
        experience.dataset,
        batch_size=4,
        shuffle=False
    )

    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            if hasattr(model, 'model'):
                # Our wrapped model
                outputs = model.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            else:
                # Direct model
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

            # Accumulate loss
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[1]
            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    model.train()
    return total_loss / total_samples if total_samples > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Avalanche-based continual learning")

    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--max_length", type=int, default=512)

    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_samples_per_phase", type=int, default=1000)

    # Strategy arguments
    parser.add_argument(
        "--strategy",
        type=str,
        default="naive",
        choices=["naive", "ewc", "si"],
        help="Continual learning strategy"
    )
    parser.add_argument("--ewc_lambda", type=float, default=0.4)
    parser.add_argument("--si_lambda", type=float, default=0.4)

    # Output arguments
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info("="*60)
    logger.info("Avalanche-Based Continual Learning")
    logger.info("="*60)
    logger.info(f"Model: {args.model_name_or_path}")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("="*60)

    # Load datasets
    logger.info("\nLoading datasets...")
    data_loader = ContinualDatasetLoader(seed=args.seed)

    phases_data = {
        "arithmetic": data_loader.load_arithmetic_data(args.max_samples_per_phase),
        "algebra": data_loader.load_algebra_data(args.max_samples_per_phase),
        "geometry": data_loader.load_geometry_data(args.max_samples_per_phase)
    }

    for phase, data in phases_data.items():
        logger.info(f"  {phase}: {len(data)} samples")

    # Setup model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = setup_model(args.model_name_or_path, device)

    # Create Avalanche benchmark
    benchmark = create_avalanche_benchmark(
        phases_data,
        tokenizer,
        args.max_length
    )

    # Train
    results = train_with_avalanche(
        model=model,
        tokenizer=tokenizer,
        benchmark=benchmark,
        strategy_name=args.strategy,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        ewc_lambda=args.ewc_lambda,
        si_lambda=args.si_lambda
    )

    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    logger.info(f"Results: {args.output_dir}/continual_results.json")


if __name__ == "__main__":
    main()
