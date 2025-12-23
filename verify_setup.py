"""
Utility script to verify the environment and dependencies.
Run this before training to ensure everything is set up correctly.
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (Need 3.8+)")
        return False


def check_cuda():
    """Check if CUDA is available."""
    print("\nChecking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  - CUDA version: {torch.version.cuda}")
            print(f"  - GPU: {torch.cuda.get_device_name(0)}")
            print(f"  - GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print("✗ CUDA not available (will use CPU - training will be very slow)")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False


def check_packages():
    """Check if required packages are installed."""
    print("\nChecking required packages...")
    
    required_packages = [
        'torch',
        'transformers',
        'datasets',
        'peft',
        'accelerate',
        'bitsandbytes',
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (not installed)")
            all_installed = False
    
    return all_installed


def check_directories():
    """Check if required directories exist."""
    print("\nChecking directories...")
    
    import config
    
    dirs = [
        config.DATA_DIR,
        config.MODEL_DIR,
        config.OUTPUT_DIR,
        config.LOGS_DIR,
    ]
    
    for dir_path in dirs:
        if dir_path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✓ {dir_path} (will be created)")
            dir_path.mkdir(exist_ok=True, parents=True)
    
    return True


def check_disk_space():
    """Check available disk space."""
    print("\nChecking disk space...")
    
    import shutil
    
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)
    
    if free_gb > 50:
        print(f"✓ {free_gb:.1f} GB free (OK)")
        return True
    elif free_gb > 20:
        print(f"⚠ {free_gb:.1f} GB free (may not be enough for model downloads)")
        return True
    else:
        print(f"✗ {free_gb:.1f} GB free (need at least 20 GB)")
        return False


def test_model_loading():
    """Test if we can load a small model."""
    print("\nTesting model loading capability...")
    
    try:
        from transformers import AutoTokenizer
        
        # Try loading a small tokenizer as a test
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        print("✓ Can load Hugging Face models")
        return True
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return False


def main():
    """Run all checks."""
    print("="*80)
    print("ENVIRONMENT VERIFICATION")
    print("="*80)
    
    results = {
        'Python version': check_python_version(),
        'CUDA': check_cuda(),
        'Required packages': check_packages(),
        'Directories': check_directories(),
        'Disk space': check_disk_space(),
        'Model loading': test_model_loading(),
    }
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    all_passed = True
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check}: {status}")
        if not passed:
            all_passed = False
    
    print("="*80)
    
    if all_passed:
        print("\n✓ All checks passed! Ready to train.")
        print("\nNext steps:")
        print("  1. Review config.py settings")
        print("  2. Run: python train.py")
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        print("\nTo install missing packages:")
        print("  pip install -r requirements.txt")
    
    print()


if __name__ == "__main__":
    main()
