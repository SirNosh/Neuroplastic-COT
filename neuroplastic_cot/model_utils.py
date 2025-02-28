"""
Utilities for loading and configuring the Qwen 2.5 3B model.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model

from neuroplastic_cot.config import (
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    NUM_EPOCHS,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    LOGGING_STEPS,
    SAVE_STEPS,
    SAVE_TOTAL_LIMIT,
    EVAL_STEPS
)


def load_tokenizer(model_name):
    """
    Load the tokenizer for the specified model.
    
    Args:
        model_name: Name or path of the model
        
    Returns:
        The loaded tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def load_model(
    model_name,
    use_lora=True,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
):
    """
    Load and configure the model for training.
    
    Args:
        model_name: Name or path of the model
        use_lora: Whether to use LoRA
        lora_r: LoRA r parameter
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout parameter
        lora_target_modules: List of modules to apply LoRA to
        
    Returns:
        The loaded and configured model
    """
    # Configure quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Apply LoRA if enabled
    if use_lora:
        # Configure LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model


def get_training_arguments(output_dir="./output"):
    """
    Get training arguments for the Trainer.
    
    Args:
        output_dir: Directory to save model checkpoints
        
    Returns:
        TrainingArguments object
    """
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        fp16=True,
        remove_unused_columns=False,
        report_to="wandb"
    ) 