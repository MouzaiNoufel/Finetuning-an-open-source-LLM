# LLM Fine-Tuning with LoRA/QLoRA

A production-ready, enterprise-grade system for fine-tuning open-source Large Language Models using parameter-efficient methods (LoRA/QLoRA). This project demonstrates professional ML engineering practices suitable for client presentations and portfolio showcases.

##  Project Overview

This project provides a **complete, production-ready framework** for fine-tuning large language models on consumer-grade GPUs. It uses state-of-the-art parameter-efficient fine-tuning (PEFT) techniques to adapt powerful base models (Mistral-7B, LLaMA-2-7B) for domain-specific tasks.

### Why This Project Matters

**For Clients:**
- **Cost-Effective**: Fine-tune 7B parameter models on a single GPU instead of requiring expensive infrastructure
- **Customizable**: Adapt general-purpose LLMs to your specific domain, brand voice, or use case
- **Fast Iteration**: LoRA enables rapid experimentation with different training strategies
- **Production-Ready**: Clean architecture, comprehensive logging, and evaluation metrics
- **Scalable**: Easy to extend to larger models or different domains

**Technical Advantages:**
- **4-bit Quantization (QLoRA)**: Reduces memory footprint by 75% with minimal quality loss
- **LoRA Adapters**: Train only 0.1% of parameters while maintaining model quality
- **Memory Efficient**: Runs on 16GB GPU (RTX 4090, A4000, etc.)
- **Reproducible**: Seeded random states and deterministic training
- **Modular Design**: Easy to swap models, datasets, or configurations

##  Architecture

```
├── config.py              # Central configuration (all hyperparameters)
├── data_utils.py          # Dataset loading and preprocessing
├── model_utils.py         # Model loading, LoRA setup, quantization
├── train.py               # Main training pipeline
├── evaluate.py            # Model evaluation (perplexity, samples)
├── inference.py           # Interactive chat interface
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore patterns
├── data/                 # Dataset directory
├── models/               # Downloaded base models
├── output/               # Training checkpoints and final model
└── logs/                 # TensorBoard logs
```

### Key Design Decisions

**1. LoRA (Low-Rank Adaptation)**
- Adds small trainable matrices to attention and FFN layers
- Freezes base model weights (no catastrophic forgetting)
- Typical configuration: rank=16, alpha=32, ~4-10M trainable parameters
- Adapters can be merged with base model for deployment

**2. 4-bit Quantization (QLoRA)**
- Uses NF4 (Normal Float 4) quantization for weights
- Double quantization for additional memory savings
- Maintains compute in FP16/BF16 for numerical stability
- Enables 7B models on 16GB VRAM

**3. Training Strategy**
- Gradient accumulation for larger effective batch sizes
- Gradient checkpointing to reduce memory
- Cosine learning rate schedule with warmup
- Mixed precision training (FP16)
- Regular evaluation and checkpointing

##  Requirements

### Hardware
- **Minimum**: 8GB GPU (RTX 4060, A4000, etc.)
- **Recommended**: 24GB GPU (RTX 4090, A5000, A6000)
- **RAM**: 32GB system memory recommended
- **Storage**: 50GB free space (for model downloads)

### Software
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- PyTorch 2.1+

##  Quick Start

### 1. Installation

```bash
# Clone or navigate to project directory
cd "Finetuning an open-source LLM"

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

All settings are in `config.py`. Key parameters:

```python
# Model selection
BASE_MODEL_NAME = "mistralai/Mistral-7B-v0.1"  # or "meta-llama/Llama-2-7b-hf"

# LoRA configuration
LORA_R = 16              # Rank (higher = more capacity)
LORA_ALPHA = 32          # Scaling factor
LORA_DROPOUT = 0.05      # Dropout for regularization

# Training
NUM_EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512

# Memory optimization
USE_4BIT_QUANTIZATION = True
GRADIENT_CHECKPOINTING = True
```

### 3. Training

```bash
# Start training with default settings
python train.py
```

**What happens during training:**
1. Downloads base model from Hugging Face (first run only)
2. Loads and preprocesses demo Alpaca-style dataset
3. Applies 4-bit quantization to base model
4. Adds LoRA adapters to target layers
5. Trains for specified epochs with evaluation
6. Saves checkpoints and final adapter weights
7. Logs metrics to TensorBoard

**Monitor training:**
```bash
# In a separate terminal
tensorboard --logdir logs/
# Open browser to http://localhost:6006
```

**Expected training time:**
- RTX 5070 ti : ~15-30 minutes (20 samples, 3 epochs)
- A6000: ~20-40 minutes
- CPU: Not recommended (very slow)

### 4. Evaluation

```bash
# Evaluate the fine-tuned model
python evaluate.py
```

This will:
- Generate sample responses for test instructions
- Calculate perplexity on validation set
- Display quality metrics

### 5. Inference

```bash
# Interactive chat mode
python inference.py

# Demo mode (predefined examples)
python inference.py --demo

# Single instruction
python inference.py "Explain how photosynthesis works"
```

##  Understanding the Results

### Metrics

**Perplexity**: Measures how well the model predicts the next token
- Lower is better
- Typical range: 5-30 for well-trained models
- <10: Excellent, 10-20: Good, 20-30: Acceptable, >30: Needs improvement

**Training Loss**: Should decrease over time
- Monitor in TensorBoard
- Plateaus indicate convergence

**Evaluation Loss**: Validates generalization
- Should track training loss
- Divergence indicates overfitting

### Sample Outputs

The model learns to:
- Follow instruction formatting
- Generate coherent, relevant responses
- Maintain consistency with training style
- Handle various question types

##  Customization

### Using Your Own Dataset

**Format**: JSON or CSV with `instruction` and `response` fields

```json
[
  {
    "instruction": "Your task description",
    "response": "Expected model output"
  },
  ...
]
```

**Configuration**:
```python
# In config.py
CUSTOM_DATASET_PATH = "path/to/your/dataset.json"
```

### Adjusting Model Behavior

**More Creative (for creative writing, brainstorming):**
```python
TEMPERATURE = 0.9
TOP_P = 0.95
DO_SAMPLE = True
```

**More Deterministic (for factual Q&A, code generation):**
```python
TEMPERATURE = 0.3
TOP_P = 0.85
DO_SAMPLE = True
```

**Longer Responses:**
```python
MAX_NEW_TOKENS = 512
MAX_SEQ_LENGTH = 1024
```

### Switching Base Models

```python
# In config.py
BASE_MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # LLaMA 2
# or
BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"  # Mistral Instruct
```

##  Educational Value

### What This Project Teaches

1. **Parameter-Efficient Fine-Tuning**: Understanding LoRA and when to use it
2. **Quantization**: Memory-efficient model deployment
3. **Production ML**: Clean code, modularity, reproducibility
4. **LLM Training**: Tokenization, attention masks, causal language modeling
5. **Evaluation**: Metrics beyond accuracy for generative models

### Key Concepts Demonstrated

- **Gradient Checkpointing**: Trade compute for memory
- **Mixed Precision**: FP16 training for speed
- **Learning Rate Scheduling**: Warmup and cosine decay
- **Data Collation**: Batching variable-length sequences
- **Model Serialization**: Saving/loading adapter weights

##  Client Presentation Points

### Value Propositions

**1. Cost Savings**
- "This approach reduces infrastructure costs by 10x compared to full fine-tuning"
- "Runs on a single GPU instead of requiring multi-GPU clusters"

**2. Speed to Market**
- "Fine-tune in hours instead of days"
- "Rapid iteration enables A/B testing different approaches"

**3. Customization**
- "Adapt to your specific domain: legal, medical, customer service, etc."
- "Match your brand voice and style guidelines"

**4. Control**
- "Full ownership of model weights and data"
- "No dependency on third-party APIs"
- "Complete data privacy and security"

**5. Scalability**
- "Start with 7B model, scale to 70B when needed"
- "Easy to retrain as new data becomes available"

### Demonstration Flow

1. **Show the problem**: Generic LLM doesn't understand domain specifics
2. **Explain the solution**: Fine-tuning with LoRA on domain data
3. **Live demo**: Before/after comparison
4. **Discuss metrics**: Quantifiable improvements
5. **Roadmap**: How to deploy and maintain

##  Advanced Topics

### Merging Adapters

Convert LoRA adapters to full model:

```python
from model_utils import load_trained_model, merge_and_save_model

adapter_path = "output/lora-adapter"
model, tokenizer = load_trained_model(adapter_path)

merge_and_save_model(model, tokenizer, "output/merged-model")
```

### Multi-GPU Training

Update `config.py`:
```python
# Automatically uses all available GPUs
DEVICE = "cuda"  # No change needed
```

Run with accelerate:
```bash
accelerate config  # First time setup
accelerate launch train.py
```

### Hyperparameter Tuning

Key parameters to experiment with:

1. **Learning Rate**: 1e-4 to 5e-4 (most important)
2. **LoRA Rank**: 8, 16, 32, 64 (higher = more capacity)
3. **Batch Size**: Constrained by memory
4. **Epochs**: 3-5 for small datasets, 1-2 for large

### Common Issues

**Out of Memory (OOM)**
- Reduce `BATCH_SIZE`
- Increase `GRADIENT_ACCUMULATION_STEPS`
- Enable `GRADIENT_CHECKPOINTING`
- Reduce `MAX_SEQ_LENGTH`

**Poor Quality**
- Increase `NUM_EPOCHS`
- Adjust `LEARNING_RATE`
- Increase `LORA_R`
- Review dataset quality

**Slow Training**
- Enable FP16: `USE_FP16 = True`
- Reduce logging frequency
- Use fewer evaluation steps

##  References

- **LoRA Paper**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **QLoRA Paper**: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- **Mistral**: [Mistral 7B](https://mistral.ai/)
- **Hugging Face**: [PEFT Documentation](https://huggingface.co/docs/peft)

##  Contributing

This project is designed for educational and commercial use. Feel free to:
- Adapt for your specific use case
- Extend with new features
- Share improvements

##  License

This project structure and code are provided as-is for educational and commercial purposes. Note that base models (Mistral, LLaMA) have their own licenses - check Hugging Face model cards for details.

##  Next Steps

1. **Train on your data**: Replace demo dataset with domain-specific examples
2. **Optimize for your use case**: Tune hyperparameters for your quality/speed requirements
3. **Deploy**: Integrate into your application via API or direct inference
4. **Monitor**: Track model performance in production
5. **Iterate**: Continuously improve with user feedback

---

**Questions?** This README covers the full lifecycle from training to deployment. For specific implementation details, see inline code comments in each module.
