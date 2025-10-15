import os
import logging
import argparse
import torch
import wandb
from tqdm import tqdm
import gc
import json
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
from data_processor import CoTDataProcessor
from neuroplasticity import (
    AdaptiveLearningRateScheduler,
    ElasticWeightConsolidation,
    HebbianLearning,
    DynamicTemperatureScheduler,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-7B with neuroplasticity mechanisms")

    # Model and dataset arguments
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B", help="Path to pretrained model")
    parser.add_argument("--dataset_name", type=str, default="simplescaling/s1K-1.1", help="Dataset name")
    parser.add_argument("--cot_format", type=str, default="deepseek_thinking_trajectory", help="CoT format to use")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluation steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save steps")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # Neuroplasticity arguments
    parser.add_argument("--use_alr", action="store_true", help="Use Adaptive Learning Rate")
    parser.add_argument("--use_ewc", action="store_true", help="Use Elastic Weight Consolidation")
    parser.add_argument("--use_hebbian", action="store_true", help="Use Hebbian Learning")
    parser.add_argument("--ewc_lambda", type=float, default=0.1, help="EWC lambda")
    parser.add_argument("--hebb_lambda", type=float, default=0.01, help="Hebbian learning lambda")
    parser.add_argument("--hebb_update_interval", type=int, default=100, help="Hebbian update interval")
    
    # Memory optimization arguments
    parser.add_argument("--ewc_sample_size", type=int, default=50, help="Number of samples for EWC computation")
    parser.add_argument("--memory_efficient", action="store_true", help="Use memory-efficient training", default=True)
    
    # Misc arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="cot-neuroplasticity", help="W&B project name")
    
    return parser.parse_args()

def setup_wandb(args):
    """Initialize Weights & Biases logging"""
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"qwen2.5-3b-cot-{'alr' if args.use_alr else ''}-{'ewc' if args.use_ewc else ''}-{'hebb' if args.use_hebbian else ''}",
        )
        logger.info("Initialized Weights & Biases logging")

def load_model_and_tokenizer(args):
    """Load model and tokenizer with quantization"""
    logger.info(f"Loading model: {args.model_name_or_path}")
    
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Model loaded with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    
    return model, tokenizer

def train(args):
    """Main training function"""
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Setup W&B logging
    setup_wandb(args)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and process dataset
    data_processor = CoTDataProcessor(
        model_name_or_path=args.model_name_or_path,
        max_length=args.max_length,
        dataset_name=args.dataset_name,
        cot_format=args.cot_format,
        batch_size=args.batch_size,
    )
    
    train_dataloader, eval_dataloader = data_processor.get_dataloaders()
    
    if train_dataloader is None:
        logger.error("No training data available. Exiting.")
        return
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Setup learning rate scheduler
    total_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps
    warmup_steps = min(args.warmup_steps, total_steps // 10)
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Setup neuroplasticity mechanisms
    neuroplasticity_mechanisms = []
    
    # Adaptive Learning Rate
    alr_scheduler = None
    if args.use_alr:
        logger.info("Using Adaptive Learning Rate")
        alr_scheduler = AdaptiveLearningRateScheduler(optimizer)
        neuroplasticity_mechanisms.append("ALR")
    
    # Elastic Weight Consolidation
    ewc = None
    if args.use_ewc:
        logger.info("Using Elastic Weight Consolidation")
        ewc = ElasticWeightConsolidation(model, ewc_lambda=args.ewc_lambda)
        
        # Create a smaller reference dataloader for EWC
        logger.info(f"Creating reference dataloader with {args.ewc_sample_size} samples for EWC")
        reference_dataloader = data_processor.get_reference_dataloader(num_samples=args.ewc_sample_size)
        
        if reference_dataloader:
            # Clear CUDA cache before computing Fisher information
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Compute Fisher Information Matrix on reference data
            ewc.compute_fisher_information(reference_dataloader, model.device)
        
        neuroplasticity_mechanisms.append("EWC")
    
    # Hebbian Learning
    hebbian = None
    if args.use_hebbian:
        logger.info("Using Hebbian Learning")
        hebbian = HebbianLearning(
            model, 
            hebb_lambda=args.hebb_lambda, 
            update_interval=args.hebb_update_interval
        )
        neuroplasticity_mechanisms.append("Hebbian")
    
    logger.info(f"Training with neuroplasticity mechanisms: {', '.join(neuroplasticity_mechanisms)}")

    # Initialize metrics tracking
    training_metrics = {
        "losses": [],
        "eval_losses": [],
        "learning_rates": [],
        "ewc_losses": [],
        "epochs": [],
        "steps": []
    }

    # Training loop
    global_step = 0
    model.train()
    
    for epoch in range(args.num_epochs):
        logger.info(f"Starting epoch {epoch+1}/{args.num_epochs}")
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            # Memory optimization: clear cache periodically
            if args.memory_efficient and step % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            # Move batch to device
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Add EWC loss if enabled
            if ewc:
                ewc_loss = ewc.ewc_loss()
                loss += ewc_loss
                
                if args.use_wandb and step % args.logging_steps == 0:
                    wandb.log({"ewc_loss": ewc_loss.item()}, step=global_step)
            
            # Scale loss for gradient accumulation
            loss = loss / args.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights if gradient accumulation is complete
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                
                # Apply ALR if enabled
                if alr_scheduler:
                    alr_scheduler.step(loss.item())
                
                # Apply Hebbian update if enabled
                if hebbian:
                    hebbian.step()
                
                optimizer.zero_grad()
                global_step += 1

                # Track metrics
                current_loss = loss.item() * args.gradient_accumulation_steps
                training_metrics["losses"].append(current_loss)
                training_metrics["learning_rates"].append(optimizer.param_groups[0]["lr"])
                training_metrics["steps"].append(global_step)
                training_metrics["epochs"].append(epoch)
                if ewc:
                    training_metrics["ewc_losses"].append(ewc_loss.item() if ewc_loss else 0)

                # Log metrics
                if args.use_wandb and global_step % args.logging_steps == 0:
                    wandb.log({
                        "loss": current_loss,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }, step=global_step)
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": loss.item() * args.gradient_accumulation_steps,
                    "lr": optimizer.param_groups[0]["lr"],
                    "step": global_step,
                })
                
                # Evaluate if needed
                if eval_dataloader and global_step % args.eval_steps == 0:
                    eval_loss = evaluate(model, eval_dataloader)
                    logger.info(f"Evaluation loss: {eval_loss:.4f}")

                    # Track eval loss
                    training_metrics["eval_losses"].append({
                        "step": global_step,
                        "eval_loss": eval_loss
                    })

                    if args.use_wandb:
                        wandb.log({"eval_loss": eval_loss}, step=global_step)

                    model.train()
                
                # Save model if needed
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    logger.info(f"Model saved to {save_path}")
                    
                # Memory optimization: clear cache after saving
                if args.memory_efficient:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
    
    # Save final model
    final_path = os.path.join(args.output_dir, "final-model")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Final model saved to {final_path}")

    # Save training metrics
    metrics_path = os.path.join(args.output_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(training_metrics, f, indent=2)
    logger.info(f"Training metrics saved to {metrics_path}")

    # Clean up
    if hebbian:
        hebbian.cleanup()

    if args.use_wandb:
        wandb.finish()

def evaluate(model, eval_dataloader):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
    
    return total_loss / len(eval_dataloader)

if __name__ == "__main__":
    args = parse_args()
    train(args) 