"""
Quick start script for training with minimal setup.
This script runs the complete training pipeline with sensible defaults.
"""

import sys
import subprocess
from pathlib import Path


def main():
    """Run the complete training pipeline."""
    print("="*80)
    print("QUICK START - LLM FINE-TUNING")
    print("="*80)
    print()
    
    # Step 1: Verify setup
    print("Step 1: Verifying environment...")
    print("-" * 80)
    try:
        result = subprocess.run(
            [sys.executable, "verify_setup.py"],
            capture_output=False,
            text=True
        )
        if result.returncode != 0:
            print("\n⚠ Setup verification had issues. Continue anyway? (y/n)")
            response = input().strip().lower()
            if response != 'y':
                print("Exiting. Please fix setup issues first.")
                return
    except Exception as e:
        print(f"Error running verification: {e}")
        print("Continuing anyway...")
    
    print()
    
    # Step 2: Train
    print("Step 2: Starting training...")
    print("-" * 80)
    print("This will take 15-30 minutes on a modern GPU.")
    print("Monitor progress in real-time or check TensorBoard.")
    print()
    
    try:
        result = subprocess.run(
            [sys.executable, "train.py"],
            capture_output=False,
            text=True
        )
        if result.returncode != 0:
            print("\n✗ Training failed. Check the error messages above.")
            return
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        return
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        return
    
    print()
    
    # Step 3: Evaluate
    print("Step 3: Evaluating model...")
    print("-" * 80)
    
    try:
        result = subprocess.run(
            [sys.executable, "evaluate.py"],
            capture_output=False,
            text=True
        )
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("You can run evaluation manually later: python evaluate.py")
    
    print()
    
    # Step 4: Demo
    print("="*80)
    print("QUICK START COMPLETE!")
    print("="*80)
    print()
    print("Next steps:")
    print("  1. Try interactive chat: python inference.py")
    print("  2. Run demo mode: python inference.py --demo")
    print("  3. View training logs: tensorboard --logdir logs/")
    print()
    print("To customize:")
    print("  - Edit config.py for hyperparameters")
    print("  - Replace data in data_utils.py with your dataset")
    print("  - Adjust generation parameters in inference.py")
    print()


if __name__ == "__main__":
    main()
