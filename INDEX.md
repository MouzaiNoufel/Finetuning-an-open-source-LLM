# 📚 Project Index - Quick Navigation Guide

## 🚀 START HERE

**New to the project?**
1. Read [GETTING_STARTED.md](GETTING_STARTED.md) - 5 minute overview
2. Run `python verify_setup.py` - Check your environment
3. Run `python quick_start.py` - Automated training pipeline

**Want to understand the architecture?**
- See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Detailed architecture
- See [PROJECT_OVERVIEW.txt](PROJECT_OVERVIEW.txt) - Complete specifications

**Ready to dive in?**
- Read [README.md](README.md) - Comprehensive guide

---

## 📁 File Guide

### Configuration Files

| File | Purpose | When to Edit |
|------|---------|--------------|
| [config.py](config.py) | **All hyperparameters and settings** | Always edit this first for customization |
| [requirements.txt](requirements.txt) | Python dependencies | When adding new packages |
| [.gitignore](.gitignore) | Git ignore patterns | When excluding new file types |

### Core Training Files

| File | Purpose | Lines | Key Functions |
|------|---------|-------|---------------|
| [data_utils.py](data_utils.py) | Dataset loading & preprocessing | 315 | `prepare_dataset()`, `tokenize_function()` |
| [model_utils.py](model_utils.py) | Model loading & LoRA setup | 235 | `load_base_model()`, `prepare_model_for_training()` |
| [train.py](train.py) | Main training pipeline | 175 | `train()`, `create_training_arguments()` |
| [evaluate.py](evaluate.py) | Model evaluation | 200 | `calculate_perplexity()`, `run_evaluation()` |
| [inference.py](inference.py) | Interactive chat interface | 210 | `ChatBot.chat()`, `generate_response()` |

### Utility Scripts

| File | Purpose | Usage |
|------|---------|-------|
| [verify_setup.py](verify_setup.py) | Environment verification | `python verify_setup.py` |
| [quick_start.py](quick_start.py) | Automated pipeline | `python quick_start.py` |
| [create_custom_dataset.py](create_custom_dataset.py) | Dataset creation examples | `python create_custom_dataset.py` |

### Documentation Files

| File | Purpose | Read When |
|------|---------|-----------|
| [README.md](README.md) | Comprehensive guide | Want full documentation |
| [GETTING_STARTED.md](GETTING_STARTED.md) | Quick start guide | Starting new |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | Architecture details | Understanding structure |
| [PROJECT_OVERVIEW.txt](PROJECT_OVERVIEW.txt) | Complete specifications | Need full details |
| INDEX.md (this file) | Navigation guide | Finding specific info |

### Data Files

| File | Purpose |
|------|---------|
| [data/example_dataset.json](data/example_dataset.json) | Sample dataset format |

---

## 🎯 Common Tasks

### Task: First Time Setup
```bash
pip install -r requirements.txt
python verify_setup.py
python quick_start.py
```
**Files involved:** requirements.txt, verify_setup.py, quick_start.py

### Task: Train with Custom Data
1. Create dataset (see [create_custom_dataset.py](create_custom_dataset.py))
2. Edit [config.py](config.py): `CUSTOM_DATASET_PATH = "data/my_data.json"`
3. Run: `python train.py`

**Files to edit:** config.py, your dataset file

### Task: Adjust Training Hyperparameters
1. Edit [config.py](config.py)
2. Key settings: `NUM_EPOCHS`, `LEARNING_RATE`, `BATCH_SIZE`
3. Run: `python train.py`

**Files to edit:** config.py only

### Task: Change Base Model
1. Edit [config.py](config.py): `BASE_MODEL_NAME = "meta-llama/Llama-2-7b-hf"`
2. Run: `python train.py`

**Files to edit:** config.py only

### Task: Evaluate Trained Model
```bash
python evaluate.py
```
**Files involved:** evaluate.py, model_utils.py

### Task: Interactive Chat
```bash
python inference.py
```
**Files involved:** inference.py, model_utils.py

### Task: Adjust Generation Parameters
1. Edit [config.py](config.py)
2. Key settings: `TEMPERATURE`, `TOP_P`, `MAX_NEW_TOKENS`
3. Run: `python inference.py`

**Files to edit:** config.py only

---

## 🔍 Code Organization

### Data Pipeline
```
data_utils.py
├── load_alpaca_demo_dataset()    # Built-in demo data
├── load_custom_dataset()         # Load JSON/CSV
├── prepare_dataset()             # Main loader
├── format_instruction()          # Format single example
├── tokenize_function()           # Tokenization
└── prepare_tokenized_dataset()   # Process full dataset
```

### Model Pipeline
```
model_utils.py
├── create_bnb_config()           # Quantization setup
├── create_lora_config()          # LoRA configuration
├── load_base_model()             # Load & quantize model
├── prepare_model_for_training()  # Add LoRA adapters
├── load_trained_model()          # Load fine-tuned model
└── merge_and_save_model()        # Merge adapters
```

### Training Pipeline
```
train.py
├── set_seed()                    # Reproducibility
├── create_training_arguments()   # Training config
└── train()                       # Main training loop
```

### Evaluation Pipeline
```
evaluate.py
├── calculate_perplexity()        # Perplexity metric
├── evaluate_sample_generations() # Sample outputs
└── run_evaluation()              # Full evaluation
```

### Inference Pipeline
```
inference.py
├── ChatBot.__init__()            # Load model
├── ChatBot.generate_response()   # Single generation
├── ChatBot.chat()                # Interactive mode
├── run_single_inference()        # Non-interactive
└── demo()                        # Demo mode
```

---

## 🎓 Learning Path

### Beginner
1. Read [GETTING_STARTED.md](GETTING_STARTED.md)
2. Run `python verify_setup.py`
3. Run `python quick_start.py`
4. Experiment with `python inference.py --demo`

### Intermediate
1. Read [README.md](README.md)
2. Review [config.py](config.py) settings
3. Modify hyperparameters and retrain
4. Create custom dataset with [create_custom_dataset.py](create_custom_dataset.py)

### Advanced
1. Read [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
2. Study code in [model_utils.py](model_utils.py) and [data_utils.py](data_utils.py)
3. Modify LoRA configuration
4. Implement custom evaluation metrics
5. Deploy as API service

---

## 🔧 Configuration Quick Reference

All in [config.py](config.py):

**Model Settings:**
- `BASE_MODEL_NAME` - Which model to use
- `USE_4BIT_QUANTIZATION` - Enable/disable quantization

**LoRA Settings:**
- `LORA_R` - Rank (8, 16, 32, 64)
- `LORA_ALPHA` - Scaling factor
- `LORA_DROPOUT` - Regularization

**Training Settings:**
- `NUM_EPOCHS` - Training epochs
- `BATCH_SIZE` - Batch size per GPU
- `LEARNING_RATE` - Learning rate
- `MAX_SEQ_LENGTH` - Max sequence length

**Generation Settings:**
- `TEMPERATURE` - Creativity (0.1-1.5)
- `TOP_P` - Nucleus sampling
- `MAX_NEW_TOKENS` - Response length

---

## 📊 Output Locations

After training, find outputs here:

```
output/
├── checkpoint-100/      # Training checkpoint
├── checkpoint-200/      # Training checkpoint
└── lora-adapter/        # Final LoRA weights ← Use this for inference

logs/
└── runs/                # TensorBoard logs
```

---

## 🆘 Help & Troubleshooting

| Problem | Solution | File to Check |
|---------|----------|---------------|
| Installation issues | Run `verify_setup.py` | requirements.txt |
| Out of memory | Reduce BATCH_SIZE | config.py |
| Poor quality | Increase NUM_EPOCHS | config.py |
| Slow training | Enable USE_FP16 | config.py |
| Dataset errors | Check format | create_custom_dataset.py |
| Model not found | Check BASE_MODEL_NAME | config.py |

**Full troubleshooting guide:** See [README.md](README.md) - Troubleshooting section

---

## 📞 Quick Commands

```bash
# Setup
pip install -r requirements.txt
python verify_setup.py

# Train
python train.py
python quick_start.py

# Evaluate
python evaluate.py

# Inference
python inference.py              # Interactive
python inference.py --demo       # Demo mode
python inference.py "question"   # Single query

# Monitor
tensorboard --logdir logs/

# Create custom dataset
python create_custom_dataset.py
```

---

## 📈 Project Statistics

- **Total Files:** 16
- **Python Files:** 9
- **Documentation Files:** 5
- **Total Code Lines:** ~2,500+
- **Configuration Files:** 2
- **Data Files:** 1

**Code Quality:** Production-ready, no placeholders
**Documentation:** Comprehensive, multi-level
**Test Coverage:** Example datasets included

---

## ✅ Checklist for Success

### Before Training
- [ ] Installed all requirements (`requirements.txt`)
- [ ] Verified setup (`verify_setup.py`)
- [ ] Reviewed config (`config.py`)
- [ ] Prepared dataset (built-in or custom)

### During Training
- [ ] Monitoring TensorBoard
- [ ] Checking training loss
- [ ] Validating checkpoints exist

### After Training
- [ ] Run evaluation (`evaluate.py`)
- [ ] Test inference (`inference.py`)
- [ ] Check perplexity metric
- [ ] Review sample generations

---

**👉 Start with:** [GETTING_STARTED.md](GETTING_STARTED.md)

**Need help?** Check [README.md](README.md) troubleshooting section

**Want details?** See [PROJECT_OVERVIEW.txt](PROJECT_OVERVIEW.txt)
