"""
Training script for fine-tuning LLM with LoRA/QLoRA.
This is the main entry point for model training.
"""

import os
import torch
import random
import numpy as np
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_callback import TrainerCallback
import config
from model_utils import (
    load_base_model,
    prepare_model_for_training,
)
from data_utils import (
    prepare_dataset,
    prepare_tokenized_dataset,
    print_dataset_sample,
)


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LoggingCallback(TrainerCallback):
    """Custom callback for additional logging during training."""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log training metrics."""
        if logs is not None:
            # Filter out None values
            logs = {k: v for k, v in logs.items() if v is not None}
            if logs:
                print(f"Step {state.global_step}: {logs}")


def create_training_arguments() -> TrainingArguments:
    """
    Create training arguments for the Trainer.
    
    Returns:
        TrainingArguments: Configuration for training
    """
    training_args = TrainingArguments(
        output_dir=str(config.OUTPUT_DIR),
        
        # Training hyperparameters
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        max_grad_norm=config.MAX_GRAD_NORM,
        
        # Optimizer
        optim=config.OPTIM,
        lr_scheduler_type=config.LR_SCHEDULER_TYPE,
        
        # Mixed precision training
        fp16=config.USE_FP16 and config.DEVICE == "cuda",
        bf16=False,
        
        # Logging
        logging_dir=str(config.LOGS_DIR),
        logging_steps=config.LOGGING_STEPS,
        logging_first_step=True,
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        
        # Saving
        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        
        # Other
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=config.RANDOM_SEED,
        data_seed=config.RANDOM_SEED,
        report_to="tensorboard",
        remove_unused_columns=False,
        
        # Disable tqdm for cleaner output
        disable_tqdm=False,
    )
    
    return training_args


def train():
    """Main training function."""
    print("="*80)
    print("STARTING LLM FINE-TUNING WITH LoRA")
    print("="*80)
    print(f"Base Model: {config.BASE_MODEL_NAME}")
    print(f"Output Directory: {config.OUTPUT_DIR}")
    print(f"Device: {config.DEVICE}")
    print(f"Mixed Precision (FP16): {config.USE_FP16 and config.DEVICE == 'cuda'}")
    print(f"4-bit Quantization: {config.USE_4BIT_QUANTIZATION}")
    print("="*80)
    
    # Set random seed for reproducibility
    print(f"\nSetting random seed: {config.RANDOM_SEED}")
    set_seed(config.RANDOM_SEED)
    
    # Load dataset
    print("\n" + "="*80)
    print("STEP 1: Loading Dataset")
    print("="*80)
    dataset = prepare_dataset()
    print_dataset_sample(dataset['train'], num_samples=2)
    
    # Load model and tokenizer
    print("\n" + "="*80)
    print("STEP 2: Loading Base Model")
    print("="*80)
    model, tokenizer = load_base_model()
    
    # Prepare model for training
    print("\n" + "="*80)
    print("STEP 3: Preparing Model for LoRA Training")
    print("="*80)
    model = prepare_model_for_training(model)
    
    # Tokenize dataset
    print("\n" + "="*80)
    print("STEP 4: Tokenizing Dataset")
    print("="*80)
    tokenized_dataset = prepare_tokenized_dataset(dataset, tokenizer)
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # Create training arguments
    print("\n" + "="*80)
    print("STEP 5: Setting Up Training")
    print("="*80)
    training_args = create_training_arguments()
    
    print(f"Total epochs: {config.NUM_EPOCHS}")
    print(f"Batch size per device: {config.BATCH_SIZE}")
    print(f"Gradient accumulation steps: {config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective batch size: {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Warmup ratio: {config.WARMUP_RATIO}")
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        data_collator=data_collator,
        callbacks=[LoggingCallback()],
    )
    
    # Start training
    print("\n" + "="*80)
    print("STEP 6: Starting Training")
    print("="*80)
    print("Training in progress... This may take a while depending on your hardware.")
    print("Monitor progress in TensorBoard: tensorboard --logdir logs/")
    print("="*80 + "\n")
    
    # Train the model
    trainer.train()
    
    # Save the final model
    print("\n" + "="*80)
    print("STEP 7: Saving Model")
    print("="*80)
    
    final_output_dir = config.OUTPUT_DIR / config.ADAPTER_NAME
    print(f"Saving LoRA adapters to: {final_output_dir}")
    trainer.model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Model saved to: {final_output_dir}")
    print(f"To use the model for inference, run: python inference.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    train()
