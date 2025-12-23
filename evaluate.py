"""
Evaluation utilities for assessing model performance.
Includes perplexity calculation and sample generation evaluation.
"""

import torch
import numpy as np
from typing import Dict, List
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset
from tqdm import tqdm
import config


def calculate_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    max_samples: int = None
) -> float:
    """
    Calculate perplexity on a dataset.
    Lower perplexity indicates better model performance.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        dataset: Dataset to evaluate on
        max_samples: Maximum number of samples to evaluate (None for all)
        
    Returns:
        float: Perplexity score
    """
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    
    # Limit samples if specified
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Calculating perplexity on {len(dataset)} samples...")
    
    with torch.no_grad():
        for example in tqdm(dataset, desc="Evaluating"):
            # Format and tokenize
            text = config.INSTRUCTION_TEMPLATE.format(
                instruction=example['instruction'],
                response=example['response']
            )
            
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=config.MAX_SEQ_LENGTH,
            )
            
            # Move to device
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            
            # Accumulate
            num_tokens = inputs['input_ids'].numel()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity


def evaluate_sample_generations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_instructions: List[str],
    verbose: bool = True
) -> List[Dict[str, str]]:
    """
    Generate responses for test instructions and display them.
    
    Args:
        model: Model to use for generation
        tokenizer: Tokenizer for the model
        test_instructions: List of instructions to test
        verbose: Whether to print generations
        
    Returns:
        List[Dict[str, str]]: List of instruction-response pairs
    """
    model.eval()
    results = []
    
    if verbose:
        print("\n" + "="*80)
        print("SAMPLE GENERATIONS")
        print("="*80)
    
    for idx, instruction in enumerate(test_instructions):
        # Format prompt
        prompt = config.INSTRUCTION_TEMPLATE_WITHOUT_RESPONSE.format(
            instruction=instruction
        )
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P,
                top_k=config.TOP_K,
                repetition_penalty=config.REPETITION_PENALTY,
                do_sample=config.DO_SAMPLE,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response (remove prompt)
        if "### Response:" in generated_text:
            response = generated_text.split("### Response:")[-1].strip()
        else:
            response = generated_text[len(prompt):].strip()
        
        results.append({
            'instruction': instruction,
            'response': response
        })
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Example {idx + 1}")
            print(f"{'='*80}")
            print(f"Instruction: {instruction}")
            print(f"\nGenerated Response:\n{response}")
            print(f"{'='*80}")
    
    return results


def run_evaluation(
    adapter_path: str = None,
    test_instructions: List[str] = None
) -> Dict[str, float]:
    """
    Run comprehensive evaluation on the fine-tuned model.
    
    Args:
        adapter_path: Path to LoRA adapters (uses default if None)
        test_instructions: Custom test instructions (uses defaults if None)
        
    Returns:
        Dict[str, float]: Evaluation metrics
    """
    from model_utils import load_trained_model
    from data_utils import prepare_dataset
    
    # Default adapter path
    if adapter_path is None:
        adapter_path = str(config.OUTPUT_DIR / config.ADAPTER_NAME)
    
    # Default test instructions
    if test_instructions is None:
        test_instructions = [
            "Explain what a neural network is in simple terms.",
            "Write a Python function to reverse a string.",
            "What are the main causes of climate change?",
            "How does the human immune system work?",
            "Describe the process of making coffee.",
        ]
    
    print("="*80)
    print("STARTING MODEL EVALUATION")
    print("="*80)
    print(f"Loading model from: {adapter_path}\n")
    
    # Load model
    model, tokenizer = load_trained_model(adapter_path)
    
    # Evaluate sample generations
    print("\nGenerating sample responses...")
    evaluate_sample_generations(model, tokenizer, test_instructions, verbose=True)
    
    # Calculate perplexity on validation set
    print("\n" + "="*80)
    print("CALCULATING PERPLEXITY")
    print("="*80)
    
    dataset = prepare_dataset()
    perplexity = calculate_perplexity(
        model,
        tokenizer,
        dataset['test'],
        max_samples=50  # Limit for faster evaluation
    )
    
    print(f"\nValidation Perplexity: {perplexity:.2f}")
    print("(Lower is better - typical range: 5-30 for good models)")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    
    return {
        'perplexity': perplexity
    }


if __name__ == "__main__":
    # Run evaluation with default settings
    run_evaluation()
