"""
Model utilities for loading and configuring the base model with quantization and LoRA.
This module handles all model-related operations for efficient fine-tuning.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from typing import Tuple
import config


def create_bnb_config() -> BitsAndBytesConfig:
    """
    Create BitsAndBytes configuration for 4-bit quantization.
    This dramatically reduces memory usage while maintaining model quality.
    
    Returns:
        BitsAndBytesConfig: Configuration for quantization
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.USE_4BIT_QUANTIZATION,
        bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=config.BNB_4BIT_COMPUTE_DTYPE,
        bnb_4bit_use_double_quant=config.BNB_4BIT_USE_DOUBLE_QUANT,
    )
    return bnb_config


def create_lora_config() -> LoraConfig:
    """
    Create LoRA (Low-Rank Adaptation) configuration.
    LoRA adds trainable rank decomposition matrices to existing weights,
    allowing efficient fine-tuning with minimal additional parameters.
    
    Returns:
        LoraConfig: Configuration for LoRA adapter
    """
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias=config.LORA_BIAS,
        task_type=config.LORA_TASK_TYPE,
    )
    return lora_config


def load_base_model() -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load the base model and tokenizer with quantization.
    
    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: Model and tokenizer
    """
    print(f"Loading base model: {config.BASE_MODEL_NAME}")
    print(f"Device: {config.DEVICE}")
    
    # Create quantization config
    bnb_config = create_bnb_config()
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_NAME,
        quantization_config=bnb_config if config.USE_4BIT_QUANTIZATION else None,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=config.BNB_4BIT_COMPUTE_DTYPE,
    )
    
    # Disable cache for gradient checkpointing
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.BASE_MODEL_NAME,
        trust_remote_code=True,
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Set padding side to right for training
    tokenizer.padding_side = 'right'
    
    print(f"Model loaded successfully")
    print(f"Model dtype: {model.dtype}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    
    return model, tokenizer


def prepare_model_for_training(model: PreTrainedModel) -> PreTrainedModel:
    """
    Prepare the base model for LoRA training.
    
    Args:
        model: Base model to prepare
        
    Returns:
        PreTrainedModel: PEFT model with LoRA adapters
    """
    print("Preparing model for training with LoRA...")
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.GRADIENT_CHECKPOINTING
    )
    
    # Enable gradient checkpointing
    if config.GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
    
    # Create LoRA config
    lora_config = create_lora_config()
    
    # Add LoRA adapters
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    print_trainable_parameters(model)
    
    return model


def print_trainable_parameters(model: PreTrainedModel) -> None:
    """
    Print the number of trainable parameters in the model.
    This helps verify that LoRA is working correctly (should be <1% of total params).
    
    Args:
        model: Model to inspect
    """
    trainable_params = 0
    all_params = 0
    
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    trainable_percent = 100 * trainable_params / all_params
    
    print(f"Trainable params: {trainable_params:,} || "
          f"All params: {all_params:,} || "
          f"Trainable %: {trainable_percent:.4f}%")


def load_trained_model(
    adapter_path: str,
    base_model_name: str = None
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a fine-tuned model with LoRA adapters for inference.
    
    Args:
        adapter_path: Path to the saved LoRA adapter
        base_model_name: Name of the base model (uses config default if None)
        
    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: Model and tokenizer
    """
    if base_model_name is None:
        base_model_name = config.BASE_MODEL_NAME
    
    print(f"Loading fine-tuned model from: {adapter_path}")
    
    # Create quantization config for inference
    bnb_config = create_bnb_config()
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config if config.USE_4BIT_QUANTIZATION else None,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=config.BNB_4BIT_COMPUTE_DTYPE,
    )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Set to evaluation mode
    model.eval()
    
    print("Model loaded successfully for inference")
    
    return model, tokenizer


def merge_and_save_model(
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    output_path: str
) -> None:
    """
    Merge LoRA adapters with base model and save the full model.
    This creates a standalone model without requiring PEFT.
    
    Args:
        model: PEFT model with LoRA adapters
        tokenizer: Tokenizer to save
        output_path: Path to save the merged model
    """
    print(f"Merging LoRA adapters with base model...")
    
    # Merge adapters
    merged_model = model.merge_and_unload()
    
    # Save merged model
    print(f"Saving merged model to: {output_path}")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print("Model saved successfully")
