"""
Configuration file for LLM fine-tuning project.
All hyperparameters and paths are defined here for easy modification.
"""

import torch
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODEL_DIR, OUTPUT_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Base model from Hugging Face Hub
BASE_MODEL_NAME = "mistralai/Mistral-7B-v0.1"  # Alternative: "meta-llama/Llama-2-7b-hf"

# Quantization settings for memory efficiency
USE_4BIT_QUANTIZATION = True
BNB_4BIT_COMPUTE_DTYPE = torch.float16
BNB_4BIT_QUANT_TYPE = "nf4"  # Normal Float 4-bit quantization
BNB_4BIT_USE_DOUBLE_QUANT = True  # Nested quantization for additional memory savings

# ============================================================================
# LoRA CONFIGURATION
# ============================================================================
# LoRA: Low-Rank Adaptation - adds trainable rank decomposition matrices
LORA_R = 16  # Rank of the update matrices (higher = more parameters)
LORA_ALPHA = 32  # Scaling factor (typically 2x rank)
LORA_DROPOUT = 0.05  # Dropout probability for LoRA layers
LORA_TARGET_MODULES = [
    "q_proj",  # Query projection in attention
    "k_proj",  # Key projection in attention
    "v_proj",  # Value projection in attention
    "o_proj",  # Output projection in attention
    "gate_proj",  # Gate projection in FFN
    "up_proj",  # Up projection in FFN
    "down_proj",  # Down projection in FFN
]
LORA_BIAS = "none"  # Can be 'none', 'all', or 'lora_only'
LORA_TASK_TYPE = "CAUSAL_LM"  # Task type for PEFT

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
# Training hyperparameters
NUM_EPOCHS = 3
BATCH_SIZE = 4  # Per device batch size
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 4 * 4 = 16
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03  # 3% of training steps for warmup
MAX_GRAD_NORM = 0.3  # Gradient clipping

# Learning rate scheduler
LR_SCHEDULER_TYPE = "cosine"  # Options: linear, cosine, constant

# Optimizer settings
OPTIM = "paged_adamw_32bit"  # Memory-efficient optimizer from bitsandbytes

# Logging and checkpointing
LOGGING_STEPS = 10
SAVE_STEPS = 100
EVAL_STEPS = 100
SAVE_TOTAL_LIMIT = 3  # Keep only last 3 checkpoints

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
# Maximum sequence length (tokens)
MAX_SEQ_LENGTH = 512

# Training/validation split
TRAIN_TEST_SPLIT = 0.1  # 10% for validation

# Random seed for reproducibility
RANDOM_SEED = 42

# Dataset settings
DATASET_NAME = "alpaca"  # Built-in demo dataset
CUSTOM_DATASET_PATH = None  # Set to path for custom dataset

# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================
# Generation parameters
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7  # Higher = more creative, Lower = more deterministic
TOP_P = 0.9  # Nucleus sampling
TOP_K = 50  # Top-k sampling
REPETITION_PENALTY = 1.1  # Penalize repetition
DO_SAMPLE = True  # Use sampling instead of greedy decoding

# ============================================================================
# HARDWARE CONFIGURATION
# ============================================================================
# Automatically detect device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = True  # Use mixed precision training (faster on modern GPUs)

# Enable gradient checkpointing to save memory
GRADIENT_CHECKPOINTING = True

# ============================================================================
# INSTRUCTION TEMPLATE
# ============================================================================
# Prompt template for instruction following
INSTRUCTION_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}"""

INSTRUCTION_TEMPLATE_WITHOUT_RESPONSE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

# ============================================================================
# MODEL OUTPUT NAMES
# ============================================================================
FINAL_MODEL_NAME = "mistral-7b-finetuned"
ADAPTER_NAME = "checkpoint-6"  # Updated to use the latest checkpoint
