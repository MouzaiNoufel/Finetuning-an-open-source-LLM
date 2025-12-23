# Project Structure

```
Finetuning an open-source LLM/
│
├── README.md                      # Comprehensive documentation
├── requirements.txt               # Python dependencies
├── .gitignore                    # Git ignore patterns
│
├── config.py                     # Central configuration file
│                                 # - Model settings (base model, quantization)
│                                 # - LoRA hyperparameters
│                                 # - Training configuration
│                                 # - Inference parameters
│
├── data_utils.py                 # Dataset utilities
│                                 # - Load demo Alpaca dataset
│                                 # - Load custom datasets (JSON/CSV)
│                                 # - Tokenization functions
│                                 # - Data preprocessing
│
├── model_utils.py                # Model utilities
│                                 # - Load base model with quantization
│                                 # - Configure LoRA adapters
│                                 # - Prepare model for training
│                                 # - Load trained models
│                                 # - Merge adapters with base model
│
├── train.py                      # Main training script
│                                 # - Complete training pipeline
│                                 # - Progress logging
│                                 # - Checkpoint management
│                                 # - TensorBoard integration
│
├── evaluate.py                   # Evaluation utilities
│                                 # - Calculate perplexity
│                                 # - Generate sample outputs
│                                 # - Quality metrics
│
├── inference.py                  # Inference interface
│                                 # - Interactive chat mode
│                                 # - Single instruction mode
│                                 # - Demo mode with examples
│
├── verify_setup.py               # Environment verification
│                                 # - Check Python version
│                                 # - Verify CUDA availability
│                                 # - Test package installation
│                                 # - Validate disk space
│
├── quick_start.py                # Automated quick start
│                                 # - Run full pipeline
│                                 # - Verification → Training → Evaluation
│
├── data/                         # Dataset directory
│   └── (datasets stored here)
│
├── models/                       # Downloaded base models
│   └── (Hugging Face cache)
│
├── output/                       # Training outputs
│   ├── checkpoint-*/            # Training checkpoints
│   └── lora-adapter/            # Final LoRA weights
│
└── logs/                         # Training logs
    └── (TensorBoard logs)
```

## File Descriptions

### Core Files

**config.py**
- Single source of truth for all configuration
- Easy to modify without touching code
- Includes detailed comments for each parameter
- Organized into logical sections

**data_utils.py**
- Handles all data loading and preprocessing
- Built-in demo dataset (20 instruction-response pairs)
- Support for custom JSON/CSV datasets
- Tokenization with proper padding and truncation

**model_utils.py**
- Model initialization with quantization
- LoRA configuration and integration
- Utilities for saving/loading models
- Adapter merging functionality

**train.py**
- Complete training pipeline
- Reproducibility through seeding
- Comprehensive logging
- Checkpoint management

**evaluate.py**
- Model quality assessment
- Perplexity calculation
- Sample generation evaluation
- Standalone or imported usage

**inference.py**
- Interactive chat interface
- Command-line argument support
- Configurable generation parameters
- Demo mode for quick testing

### Utility Scripts

**verify_setup.py**
- Pre-flight checks before training
- Validates environment configuration
- Tests package installation
- Checks system resources

**quick_start.py**
- Automated pipeline execution
- Runs verification → training → evaluation
- User-friendly for beginners

## Workflow

### 1. Initial Setup
```bash
pip install -r requirements.txt
python verify_setup.py
```

### 2. Training
```bash
python train.py
# or
python quick_start.py
```

### 3. Monitoring
```bash
tensorboard --logdir logs/
```

### 4. Evaluation
```bash
python evaluate.py
```

### 5. Inference
```bash
python inference.py              # Interactive mode
python inference.py --demo       # Demo mode
python inference.py "question"   # Single query
```

## Customization Points

### Change Base Model
Edit `config.py`:
```python
BASE_MODEL_NAME = "meta-llama/Llama-2-7b-hf"
```

### Use Custom Dataset
Edit `config.py`:
```python
CUSTOM_DATASET_PATH = "data/my_dataset.json"
```

Format:
```json
[
  {"instruction": "...", "response": "..."},
  ...
]
```

### Adjust Training
Edit `config.py`:
```python
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
BATCH_SIZE = 2
```

### Modify Generation
Edit `config.py`:
```python
TEMPERATURE = 0.8
MAX_NEW_TOKENS = 512
```

## Best Practices

1. **Always run verify_setup.py first** - catches issues early
2. **Monitor TensorBoard** - watch for overfitting
3. **Start small** - test with 1 epoch before full training
4. **Evaluate frequently** - check sample outputs during training
5. **Version control** - track config changes in git
6. **Save checkpoints** - training can be interrupted
7. **Document changes** - note what works for your use case

## Memory Optimization

If you encounter OOM errors:

1. Reduce batch size: `BATCH_SIZE = 1`
2. Increase gradient accumulation: `GRADIENT_ACCUMULATION_STEPS = 8`
3. Enable gradient checkpointing: `GRADIENT_CHECKPOINTING = True`
4. Reduce sequence length: `MAX_SEQ_LENGTH = 256`
5. Use 4-bit quantization: `USE_4BIT_QUANTIZATION = True`

## Production Deployment

### Option 1: Use LoRA Adapters
- Keep adapters separate from base model
- Swap adapters for different tasks
- Minimal storage overhead

### Option 2: Merge to Full Model
```python
from model_utils import merge_and_save_model
model, tokenizer = load_trained_model("output/lora-adapter")
merge_and_save_model(model, tokenizer, "output/merged-model")
```

### Option 3: API Deployment
- Wrap inference.py in FastAPI/Flask
- Add authentication and rate limiting
- Deploy to cloud (AWS, GCP, Azure)

## Troubleshooting

**Import errors**: Run `pip install -r requirements.txt`
**CUDA errors**: Update GPU drivers
**OOM errors**: Reduce batch size or sequence length
**Poor quality**: Increase epochs or adjust learning rate
**Slow training**: Enable FP16, reduce logging frequency
