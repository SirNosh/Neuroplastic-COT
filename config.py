"""
Configuration parameters for finetuning Qwen 2.5 3B on S1.1K Chain of Thought dataset.
"""

# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = "./qwen-2.5-3b-cot-finetuned"

# Dataset configuration
DATASET_NAME = "simplescaling/s1K-1.1"  # The correct S1.1K dataset
DATASET_SUBSET = "default"  # Default subset of the dataset
MAX_LENGTH = 2048
PROMPT_TEMPLATE = """
Problem: {problem}

Think step by step to solve this problem.

Answer:
"""

# Training configuration
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 3
WARMUP_RATIO = 0.03
LR_SCHEDULER_TYPE = "cosine"
LOGGING_STEPS = 10
EVAL_STEPS = 100
SAVE_STEPS = 500

# LoRA configuration
USE_LORA = True
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", 
    "k_proj", 
    "v_proj", 
    "o_proj", 
    "gate_proj", 
    "up_proj", 
    "down_proj"
]

# Mixed precision training
FP16 = True
BF16 = False  # Set to True if your GPU supports BF16

# Neuroplasticity configuration
USE_NEUROPLASTICITY = True

# Adaptive Learning Rate (ALR) configuration
USE_ALR = True
ALR_BASE_LR = 2e-5
ALR_MAX_FACTOR = 3.0  # Maximum multiplier for learning rate
ALR_COMPLEXITY_MEASURE = "token_length"  # Options: "token_length", "reasoning_depth", "entropy"

# Elastic Weight Consolidation (EWC) configuration
USE_EWC = True
EWC_IMPORTANCE_FACTOR = 1000.0  # Scaling factor for EWC penalty
EWC_FISHER_SAMPLE_SIZE = 100  # Number of samples for Fisher information calculation
EWC_FISHER_UPDATE_FREQ = 500  # How often to update Fisher information (in steps)

# Hebbian Learning configuration
USE_HEBBIAN = True
HEBBIAN_FACTOR = 0.01  # Scaling factor for Hebbian updates
HEBBIAN_ACTIVATION_THRESHOLD = 0.5  # Threshold for considering neurons as co-activated
HEBBIAN_TARGET_LAYERS = [  # Layers to apply Hebbian learning to
    "mlp",
    "attention"
]

# Miscellaneous
SEED = 42
USE_WANDB = True
WANDB_PROJECT = "qwen-2.5-3b-cot-finetuning" 