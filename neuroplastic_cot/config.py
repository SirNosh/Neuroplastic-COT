"""
Configuration for the Neuroplastic Chain of Thought project.
"""

# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# Dataset configuration
DATASET_NAME = "stanfordnlp/ScienceQA"
TRAIN_SPLIT_RATIO = 0.9
MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 1024

# Training configuration
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 10
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 3
EVAL_STEPS = 100

# LoRA configuration
LORA_ENABLED = True
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Generation configuration
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7
TOP_P = 0.9

# Wandb configuration
WANDB_ENABLED = False
WANDB_PROJECT = "neuroplastic-cot"
WANDB_NAME = "qwen-2.5-3b-cot"

# Random seed
SEED = 42

# Neuroplasticity configuration
USE_ALR = True
USE_EWC = True
USE_HEBBIAN = True

# Adaptive Learning Rate configuration
ALR_CONFIG = {
    "base_lr": 2e-5,
    "max_lr_factor": 3.0,
    "complexity_measure": "token_length"
}

# Elastic Weight Consolidation configuration
EWC_CONFIG = {
    "importance_factor": 1000.0,
    "fisher_sample_size": 100,
    "fisher_update_freq": 500
}

# Hebbian Learning configuration
HEBBIAN_CONFIG = {
    "hebbian_factor": 0.01,
    "activation_threshold": 0.5,
    "target_layers": ["mlp", "attention"]
} 