"""
Utilities for loading and configuring the Qwen 2.5 3B model.
"""

import os
from typing import Dict, List, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

import config


def load_tokenizer(model_name=None):
    """
    Load the tokenizer.
    
    Args:
        model_name: Name or path of the model to load the tokenizer from.
                   If None, uses the model name from config.
    
    Returns:
        The loaded tokenizer.
    """
    model_name = model_name or config.MODEL_NAME
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def load_model(model_name=None):
    """
    Load the model with quantization and LoRA if enabled.
    
    Args:
        model_name: Name or path of the model to load.
                   If None, uses the model name from config.
    
    Returns:
        The loaded model.
    """
    model_name = model_name or config.MODEL_NAME
    
    # Configure quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Disable KV cache during training
    model.config.use_cache = False
    
    # Apply LoRA if enabled
    if config.USE_LORA:
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            lora_dropout=config.LORA_DROPOUT,
            target_modules=config.LORA_TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model


def get_training_arguments():
    """
    Get training arguments.
    
    Returns:
        TrainingArguments object.
    """
    # Create output directory if it doesn't exist
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        num_train_epochs=config.NUM_EPOCHS,
        warmup_ratio=config.WARMUP_RATIO,
        lr_scheduler_type=config.LR_SCHEDULER_TYPE,
        logging_steps=config.LOGGING_STEPS,
        evaluation_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        fp16=config.FP16,
        bf16=config.BF16,
        report_to="wandb" if config.USE_WANDB else "none",
        seed=config.SEED,
        data_seed=config.SEED,
        remove_unused_columns=False,  # Important for custom dataloaders
    )
    
    return training_args 