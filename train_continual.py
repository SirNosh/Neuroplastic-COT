"""
Continual Learning Training Script

Trains a model sequentially across multiple mathematical domains (arithmetic, algebra, geometry)
and measures catastrophic forgetting with and without neuroplasticity mechanisms.
"""

import os
import logging
import argparse
import torch
import json
from tqdm import tqdm
import gc
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.utils.data import Dataset, DataLoader
from dataset_loaders import ContinualDatasetLoader
from neuroplasticity import (
    AdaptiveLearningRateScheduler,
    ElasticWeightConsolidation,
    HebbianLearning,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Continual Learning for Neuroplasticity Testing")

    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--output_dir", type=str, default="./output_continual")

    # Training arguments
    parser.add_argument("--num_epochs_per_phase", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_samples_per_phase", type=int, default=1000, help="Max training samples per phase")

    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Neuroplasticity arguments
    parser.add_argument("--use_alr", action="store_true")
    parser.add_argument("--use_ewc", action="store_true")
    parser.add_argument("--use_hebbian", action="store_true")
    parser.add_argument("--ewc_lambda", type=float, default=0.4, help="EWC importance (higher = more retention)")
    parser.add_argument("--hebb_lambda", type=float, default=0.01)
    parser.add_argument("--ewc_sample_size", type=int, default=100)

    # Misc
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


class ContinualDataset(Dataset):
    """Dataset for continual learning."""

    def __init__(self, data, tokenizer, max_length=2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["prompt"] + item["completion"]

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        }


def evaluate_phase(model, tokenizer, test_data, domain_name, max_length=2048):
    """Evaluate model on a specific domain."""
    model.eval()

    dataset = ContinualDataset(test_data, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {domain_name}", leave=False):
            batch = {k: v.to(model.device) for k, v in batch.items()}

            outputs = model(**batch)
            total_loss += outputs.loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

    model.train()
    return avg_loss


def train_continual(args):
    """Main continual learning training loop."""

    # Set seeds
    torch.manual_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    logger.info(f"Loading base model: {args.model_name_or_path}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load datasets for all phases
    loader = ContinualDatasetLoader(seed=args.seed)
    phases_data = loader.load_all_phases(max_samples_per_phase=args.max_samples_per_phase)

    # Format data for training
    formatted_phases = {}
    for phase_name, (train_data, test_data) in phases_data.items():
        formatted_train = loader.format_for_training(train_data, phase_name)
        formatted_test = loader.format_for_training(test_data, phase_name)
        formatted_phases[phase_name] = (formatted_train, formatted_test)

    # Initialize neuroplasticity mechanisms
    ewc = None
    alr_scheduler = None
    hebbian = None

    if args.use_ewc:
        ewc = ElasticWeightConsolidation(
            model=model,
            lambda_ewc=args.ewc_lambda
        )
        logger.info(f"EWC enabled with lambda={args.ewc_lambda}")

    # Track results across phases
    continual_results = {
        "phases": ["arithmetic", "algebra", "geometry"],
        "performance_matrix": {},  # performance[phase_trained_after][phase_tested_on]
        "forgetting_scores": {},
        "args": vars(args)
    }

    phase_names = ["arithmetic", "algebra", "geometry"]

    # Sequential training across phases
    for phase_idx, phase_name in enumerate(phase_names):
        logger.info(f"\n{'='*60}")
        logger.info(f"PHASE {phase_idx + 1}/3: {phase_name.upper()}")
        logger.info(f"{'='*60}\n")

        train_data, test_data = formatted_phases[phase_name]

        # Create dataset and dataloader
        train_dataset = ContinualDataset(train_data, tokenizer, args.max_length)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True
        )

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

        # Learning rate scheduler
        total_steps = len(train_dataloader) * args.num_epochs_per_phase // args.gradient_accumulation_steps
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=min(100, total_steps // 10),
            num_training_steps=total_steps
        )

        # Adaptive LR
        if args.use_alr:
            alr_scheduler = AdaptiveLearningRateScheduler(
                optimizer=optimizer,
                patience=2,
                factor=0.5
            )

        # Hebbian
        if args.use_hebbian:
            hebbian = HebbianLearning(
                model=model,
                lambda_hebb=args.hebb_lambda,
                update_interval=100
            )

        # Training loop for this phase
        model.train()
        global_step = 0

        for epoch in range(args.num_epochs_per_phase):
            logger.info(f"Epoch {epoch + 1}/{args.num_epochs_per_phase}")

            progress_bar = tqdm(train_dataloader, desc=f"{phase_name} - Epoch {epoch+1}")

            for step, batch in enumerate(progress_bar):
                batch = {k: v.to(model.device) for k, v in batch.items()}

                outputs = model(**batch)
                loss = outputs.loss

                # Add EWC loss if enabled and this isn't the first phase
                if ewc and phase_idx > 0:
                    ewc_loss = ewc.ewc_loss()
                    loss += ewc_loss

                loss = loss / args.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()

                    if alr_scheduler:
                        alr_scheduler.step(loss.item())

                    if hebbian:
                        hebbian.step()

                    optimizer.zero_grad()
                    global_step += 1

                    progress_bar.set_postfix({"loss": loss.item() * args.gradient_accumulation_steps})

        # After training on this phase, evaluate on ALL phases seen so far
        logger.info(f"\nEvaluating after {phase_name} training...")

        for eval_phase_idx in range(phase_idx + 1):
            eval_phase_name = phase_names[eval_phase_idx]
            _, eval_test_data = formatted_phases[eval_phase_name]

            eval_loss = evaluate_phase(model, tokenizer, eval_test_data, eval_phase_name, args.max_length)

            # Store result
            key = f"after_{phase_name}_test_{eval_phase_name}"
            continual_results["performance_matrix"][key] = eval_loss

            logger.info(f"  {eval_phase_name}: loss = {eval_loss:.4f}")

        # Compute Fisher Information for EWC (protect this phase's knowledge)
        if ewc and phase_idx < len(phase_names) - 1:  # Not on last phase
            logger.info(f"Computing Fisher Information for {phase_name}...")

            # Use subset of training data
            fisher_data = train_data[:args.ewc_sample_size]
            fisher_dataset = ContinualDataset(fisher_data, tokenizer, args.max_length)
            fisher_dataloader = DataLoader(fisher_dataset, batch_size=args.batch_size, shuffle=False)

            ewc.compute_fisher_information(fisher_dataloader)
            logger.info("Fisher Information computed")

        # Save checkpoint after each phase
        checkpoint_dir = os.path.join(args.output_dir, f"phase_{phase_idx+1}_{phase_name}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        logger.info(f"Checkpoint saved to {checkpoint_dir}")

        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Compute forgetting metrics
    logger.info("\n" + "="*60)
    logger.info("COMPUTING FORGETTING METRICS")
    logger.info("="*60 + "\n")

    for phase_idx in range(len(phase_names) - 1):
        phase_name = phase_names[phase_idx]

        # Score right after learning this phase
        initial_key = f"after_{phase_name}_test_{phase_name}"
        initial_score = continual_results["performance_matrix"][initial_key]

        # Score after learning all subsequent phases (final score)
        final_phase = phase_names[-1]
        final_key = f"after_{final_phase}_test_{phase_name}"
        final_score = continual_results["performance_matrix"][final_key]

        # Forgetting = (initial - final) / initial
        # For loss, lower is better, so we want final_score ≈ initial_score
        # Forgetting metric: (final - initial) / initial (lower = less forgetting)
        forgetting = (final_score - initial_score) / initial_score if initial_score > 0 else 0

        continual_results["forgetting_scores"][phase_name] = {
            "initial_loss": initial_score,
            "final_loss": final_score,
            "forgetting_percentage": forgetting * 100
        }

        logger.info(f"{phase_name}:")
        logger.info(f"  Initial loss: {initial_score:.4f}")
        logger.info(f"  Final loss: {final_score:.4f}")
        logger.info(f"  Forgetting: {forgetting*100:.2f}%")

    # Save final results
    results_path = os.path.join(args.output_dir, "continual_results.json")
    with open(results_path, "w") as f:
        json.dump(continual_results, f, indent=2)

    logger.info(f"\nResults saved to {results_path}")

    # Save final model
    final_dir = os.path.join(args.output_dir, "final-model")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Cleanup
    if hebbian:
        hebbian.cleanup()

    logger.info("Continual learning training complete!")


if __name__ == "__main__":
    args = parse_args()
    train_continual(args)
