"""
Interactive inference script for chat-based interaction with fine-tuned model.
Provides a command-line interface for testing the model.
"""

import torch
from typing import Optional
import config
from model_utils import load_trained_model


class ChatBot:
    """Interactive chatbot using the fine-tuned model."""
    
    def __init__(self, adapter_path: str = None):
        """
        Initialize the chatbot.
        
        Args:
            adapter_path: Path to LoRA adapters (uses default if None)
        """
        # Default adapter path
        if adapter_path is None:
            adapter_path = str(config.OUTPUT_DIR / config.ADAPTER_NAME)
        
        print("Loading fine-tuned model...")
        print(f"Adapter path: {adapter_path}")
        
        self.model, self.tokenizer = load_trained_model(adapter_path)
        
        print("Model loaded successfully!")
        print(f"Device: {self.model.device}")
        print()
    
    def generate_response(
        self,
        instruction: str,
        max_new_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
    ) -> str:
        """
        Generate a response for the given instruction.
        
        Args:
            instruction: User instruction/question
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            str: Generated response
        """
        # Use defaults from config if not specified
        if max_new_tokens is None:
            max_new_tokens = config.MAX_NEW_TOKENS
        if temperature is None:
            temperature = config.TEMPERATURE
        if top_p is None:
            top_p = config.TOP_P
        if top_k is None:
            top_k = config.TOP_K
        
        # Format prompt
        prompt = config.INSTRUCTION_TEMPLATE_WITHOUT_RESPONSE.format(
            instruction=instruction
        )
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=config.REPETITION_PENALTY,
                do_sample=config.DO_SAMPLE,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response (remove prompt)
        if "### Response:" in generated_text:
            response = generated_text.split("### Response:")[-1].strip()
        else:
            response = generated_text[len(prompt):].strip()
        
        return response
    
    def chat(self):
        """Start interactive chat session."""
        print("="*80)
        print("INTERACTIVE CHAT SESSION")
        print("="*80)
        print("Type your instructions/questions below.")
        print("Commands:")
        print("  - Type 'quit' or 'exit' to end the session")
        print("  - Type 'clear' to clear screen")
        print("  - Type 'help' for generation parameter adjustment")
        print("="*80)
        print()
        
        while True:
            # Get user input
            try:
                user_input = input("\n🧑 You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nExiting chat...")
                break
            
            # Handle commands
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for chatting! Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                import os
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
            
            if user_input.lower() == 'help':
                print("\nGeneration Parameters (modify in config.py):")
                print(f"  - Temperature: {config.TEMPERATURE} (higher = more creative)")
                print(f"  - Top-p: {config.TOP_P} (nucleus sampling)")
                print(f"  - Top-k: {config.TOP_K} (top-k sampling)")
                print(f"  - Max tokens: {config.MAX_NEW_TOKENS}")
                continue
            
            # Generate response
            print("\n🤖 Assistant: ", end="", flush=True)
            try:
                response = self.generate_response(user_input)
                print(response)
            except Exception as e:
                print(f"\n❌ Error generating response: {e}")
                print("Please try again with a different instruction.")


def run_single_inference(instruction: str, adapter_path: str = None) -> str:
    """
    Run inference for a single instruction (non-interactive).
    
    Args:
        instruction: Instruction to process
        adapter_path: Path to LoRA adapters
        
    Returns:
        str: Generated response
    """
    chatbot = ChatBot(adapter_path)
    response = chatbot.generate_response(instruction)
    return response


def demo():
    """Run a quick demo with sample instructions."""
    print("="*80)
    print("DEMO MODE - Sample Generations")
    print("="*80)
    print()
    
    chatbot = ChatBot()
    
    sample_instructions = [
        "Explain quantum computing in simple terms.",
        "Write a haiku about technology.",
        "What are the benefits of exercise?",
    ]
    
    for idx, instruction in enumerate(sample_instructions, 1):
        print(f"\n{'='*80}")
        print(f"Demo {idx}/{len(sample_instructions)}")
        print(f"{'='*80}")
        print(f"🧑 Instruction: {instruction}")
        print(f"\n🤖 Response:")
        
        response = chatbot.generate_response(instruction)
        print(response)
    
    print(f"\n{'='*80}")
    print("Demo complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--demo':
            # Run demo mode
            demo()
        else:
            # Single instruction mode
            instruction = ' '.join(sys.argv[1:])
            print(f"Instruction: {instruction}\n")
            response = run_single_inference(instruction)
            print(f"Response: {response}")
    else:
        # Interactive chat mode
        chatbot = ChatBot()
        chatbot.chat()
