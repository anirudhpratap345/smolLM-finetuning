# SmolLM2 Finance Fine-Tuning Project - Setup Complete ✅

## 📦 Project Structure Created

```
d:\LLM\Fine-Tuning SmolLM2 on Finance Data\
├── config/
│   ├── __init__.py
│   └── training_config.py           # Hyperparameter configurations
├── scripts/
│   ├── __init__.py
│   ├── data_loader.py               # Dataset loading & preprocessing
│   ├── model_setup.py               # SmolLM2 + LoRA initialization
│   ├── training_pipeline.py         # SFTTrainer wrapper
│   ├── inference.py                 # Inference & evaluation
│   └── utils.py                     # Utility functions
├── data/                            # (empty) Local datasets go here
├── models/                          # (empty) Saved adapters/models go here
├── train.py                         # Main training entry point
├── infer.py                         # Inference & evaluation entry point
├── EXAMPLES.py                      # Copy-paste code examples
├── requirements.txt                 # Dependencies
└── README.md                        # Full documentation
```

## 🎯 Key Components

### 1. **Configuration System** (`config/training_config.py`)
- Dataclass-based configurations for all settings
- Pre-configured profiles: `get_default_config()`, `get_colab_config()`, `get_local_gpu_config()`
- Easy to modify hyperparameters in one place

### 2. **Data Loading** (`scripts/data_loader.py`)
- Support for Financial PhraseBank, PIXIU, custom datasets
- Automatic formatting to SmolLM2 chat format
- Data balancing, splitting, and preprocessing

### 3. **Model Setup** (`scripts/model_setup.py`)
- Unsloth integration for 2x faster training
- Automatic LoRA adapter application
- Support for 4-bit quantization to fit on T4 GPU

### 4. **Training Pipeline** (`scripts/training_pipeline.py`)
- Production-grade SFTTrainer wrapper
- Automatic precision detection (fp16/bf16)
- Checkpointing and model saving

### 5. **Inference Engine** (`scripts/inference.py`)
- Fast inference with Unsloth optimization
- Batch generation for multiple prompts
- Evaluation utilities: sentiment classification, Q&A, latency benchmarks

### 6. **Utilities** (`scripts/utils.py`)
- Data I/O (JSON, CSV)
- Metrics calculation (accuracy, F1, confusion matrix)
- Text preprocessing (cleaning, balancing, truncation)
- File and config management

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
cd "d:\LLM\Fine-Tuning SmolLM2 on Finance Data"
pip install -r requirements.txt
pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### Step 2: Train
```bash
python train.py
```
- Downloads Financial PhraseBank automatically
- Trains on 1000 samples (configurable)
- Saves LoRA adapter to `smollm2-finance-tuned/`
- Training time: 20-90 min depending on hardware

### Step 3: Inference
```bash
python infer.py
```
- Interactive sentiment classification
- Q&A mode
- Batch processing
- Latency benchmarking

## 📊 File Purposes

| File | Purpose |
|------|---------|
| `train.py` | **Start here** - Orchestrates full training pipeline |
| `infer.py` | Load trained model and run inference |
| `EXAMPLES.py` | Copy-paste code for common tasks |
| `README.md` | Full documentation with troubleshooting |
| `requirements.txt` | All Python dependencies |

## 🔧 Customization

### Change Model
Edit `config/training_config.py`:
```python
model_name: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
# Change to different model if available on HF Hub
```

### Change Dataset
In `train.py`:
```python
# Option 1: Use different HF dataset
train_dataset, eval_dataset = load_dataset("your-dataset")

# Option 2: Load custom local data
train_dataset, eval_dataset = load_custom_dataset("data.json", formatter)
```

### Adjust Training Parameters
Edit `config/training_config.py`:
```python
per_device_batch_size = 2          # Reduce if OOM
learning_rate = 2e-4               # Range: 1e-4 to 5e-4
max_steps = 60                     # Increase for more epochs
gradient_accumulation_steps = 4    # Increase for effective larger batch
```

### Monitor Training
Enable W&B in `config/training_config.py`:
```python
report_to: str = "wandb"  # (not "none")
```

## 💡 Common Tasks

### 1. Train on custom data
```python
from scripts.data_loader import load_custom_dataset, FinanceDataProcessor

processor = FinanceDataProcessor()
train, eval = load_custom_dataset(
    "my_data.json",
    processor.format_custom_qa,
    max_samples=5000
)
# Then pass to train_smollm2()
```

### 2. Evaluate model performance
```python
from scripts.inference import FinanceEvaluator

results = evaluator.sentiment_classification(test_texts, test_labels)
print(f"Accuracy: {results['accuracy']:.2%}")
```

### 3. Save and share model
```python
# Push to Hugging Face Hub
model.push_to_hub("username/smollm2-finance")

# Or save locally
model.save_pretrained("my-model")
```

### 4. Benchmark latency
```python
results = evaluator.latency_benchmark(test_prompts)
print(f"Throughput: {results['throughput_tokens_per_sec']:.1f} tokens/sec")
```

## 📈 Expected Results

After training on 1k samples of Financial PhraseBank:

| Metric | Baseline | Post-Fine-Tune |
|--------|----------|---|
| **Sentiment Accuracy** | 70% | 80-85% |
| **Latency** | - | 50-100 tokens/sec |
| **Model Size** | 3.3GB | ~4MB (LoRA adapter) |
| **Training Time (T4)** | - | 45-90 min |

## ⚡ Performance Tips

1. **Use Unsloth**: 2x faster training + lower VRAM
2. **4-bit quantization**: Fit 1.7B on 6GB VRAM
3. **Gradient checkpointing**: Save 30-40% memory
4. **Batch size tuning**: Start with 1-2, increase if VRAM allows
5. **Variable-length texts**: Keep `packing=False` for finance data

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| **CUDA out of memory** | Reduce batch size or increase gradient_accumulation_steps |
| **Training diverges** | Lower learning rate (try 1e-4) or warmup more steps |
| **Model not found** | Run `pip install transformers` or check HF Hub |
| **Slow training** | Install Unsloth: `pip install unsloth...` |

## 📚 Next Steps

1. **Run training**: `python train.py`
2. **Evaluate results**: `python infer.py`
3. **Integrate with FinIQ**: Load adapter + inference in backend
4. **Scale up**: Train on larger datasets (5k-50k samples)
5. **Submit to leaderboard**: Open FinLLM or finance benchmarks

## 🔗 Resources

- **SmolLM2 Model Card**: https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct
- **Unsloth GitHub**: https://github.com/unslothai/unsloth
- **HF trl Docs**: https://huggingface.co/docs/trl
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **Finance Datasets**: https://huggingface.co/collections

---

**Ready to train?** Run `python train.py` to get started! 🚀
