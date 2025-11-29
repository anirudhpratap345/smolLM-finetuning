# SmolLM2 Finance Fine-Tuning Project

Production-ready pipeline for fine-tuning [SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) on finance data using LoRA and supervised fine-tuning (SFT).

## 🎯 Project Goals

- **Efficiency**: Train 1.7B parameter model on single GPU (T4/A10) in 1-4 hours
- **Accuracy**: 10-15% improvement on finance tasks (sentiment, Q&A, NER)
- **Domain Adaptation**: Reduce hallucinations with finance-specific fine-tuning
- **Latency**: 50-100 tokens/sec post-inference optimization

## 📊 Why SmolLM2 for Finance?

| Feature | Benefit |
|---------|---------|
| **1.7B Parameters** | Fits on T4 GPU; fast training (45-90 min) |
| **8k Context** | Handles long earnings reports & filings |
| **Math Reasoning** | GSM8K: 31% → Transfers to financial modeling |
| **Instruct Format** | Native chat/function-calling support |

## 🏗️ Project Structure

```
.
├── config/
│   └── training_config.py       # Hyperparameters & configurations
├── scripts/
│   ├── data_loader.py           # Dataset loading & preprocessing
│   ├── model_setup.py           # SmolLM2 + LoRA setup
│   ├── training_pipeline.py     # SFT trainer wrapper
│   └── inference.py             # Inference & evaluation
├── data/                        # Local dataset storage
├── models/                      # Saved adapters & merged models
├── train.py                     # Main training script
├── infer.py                     # Inference & evaluation script
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone and navigate to project
cd "d:\LLM\Fine-Tuning SmolLM2 on Finance Data"

# Install dependencies
pip install -r requirements.txt

# (Optional) Install Unsloth for 2x speedup
pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### 2. Train Model

```bash
# Local GPU (RTX 3060/4070)
python train.py

# For Colab, modify train.py:
# Replace: config = get_local_gpu_config()
# With:    config = get_colab_config()
```

**Training Time:**
- Free Colab (T4): 45-90 min
- RTX 3060/4070: 20-40 min
- A10 GPU: 10-20 min

### 3. Run Inference

```bash
python infer.py
```

Choose from:
1. **Interactive sentiment** - Real-time sentiment classification
2. **Interactive Q&A** - Finance Q&A
3. **Batch inference** - Process multiple texts
4. **Evaluation** - Latency benchmarks

## 📂 Configuration

Edit `config/training_config.py` to customize:

```python
# Model setup
model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
max_seq_length = 2048  # Or 8192 for long reports

# LoRA (lightweight adapters)
r = 16              # Rank (8=ultra-efficient, 64=high-quality)
lora_alpha = 16     # Scaling
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training
learning_rate = 2e-4
max_steps = 60      # ~1 epoch for 1k samples
per_device_batch_size = 2
gradient_accumulation_steps = 4

# Data
dataset_name = "financial_phrasebank"
max_samples = 1000  # Start small for iteration
```

## 📊 Datasets

Supported finance datasets on HF Hub:

| Dataset | Size | Best For |
|---------|------|----------|
| **financial_phrasebank** | 4.8k | Sentiment analysis |
| **FinGPT/PIXIU** | 10k-50k | Q&A, summarization |
| **CFM-NER** | 20k | Entity extraction |
| **Custom** | Unlimited | Domain-specific pairs |

Load custom data:

```python
from scripts.data_loader import load_custom_dataset

train_data, eval_data = load_custom_dataset(
    data_path="custom_finance.json",
    formatter=processor.format_custom_qa,
    max_samples=5000
)
```

## 🔧 Training Hyperparameters

**For Colab (Free T4):**
```python
per_device_batch_size = 1
gradient_accumulation_steps = 8
max_steps = 60
max_samples = 500
```

**For Local GPU (RTX 3060+):**
```python
per_device_batch_size = 4
gradient_accumulation_steps = 2
max_steps = 200
max_samples = 5000
```

**Tuning Tips:**
- Start with `learning_rate = 2e-4` (range: 1e-4 to 5e-4)
- Watch for loss spikes → reduce LR if diverging
- Use `gradient_checkpointing = True` to save 30-40% memory
- For variable-length finance texts: `packing = False`

## 📈 Expected Performance

| Metric | Baseline | Post-Fine-Tune |
|--------|----------|---|
| **Sentiment Accuracy** | 70% | 80-85% |
| **Q&A Relevance** | 65% | 75-80% |
| **Latency (tokens/sec)** | - | 50-100 |
| **Hallucination Rate** | 15-20% | 5-10% |

## 💾 Model Saving & Deployment

**Save LoRA Adapter (4MB):**
```python
model.save_pretrained("smollm2-finance-tuned")
tokenizer.save_pretrained("smollm2-finance-tuned")
```

**Save Merged Model (1.7B):**
```python
model.save_pretrained_merged(
    "smollm2-finance-merged",
    tokenizer,
    save_method="merged_16bit"
)
```

**Push to HF Hub:**
```bash
huggingface-cli upload <username>/smollm2-finance-tuned \
  smollm2-finance-tuned --repo-type model
```

## 🧪 Evaluation

### Sentiment Classification
```python
from scripts.inference import FinanceEvaluator

evaluator = FinanceEvaluator(inference)
results = evaluator.sentiment_classification(
    texts=test_texts,
    labels=test_labels
)
print(f"Accuracy: {results['accuracy']:.2%}")
```

### Q&A Generation
```python
answers = evaluator.qa_generation(questions)
```

### Hallucination Check
```python
results = evaluator.hallucination_check(
    prompts=test_prompts,
    expected_contexts=expected_contexts
)
print(f"Hallucination score: {results['avg_hallucination_score']:.2f}")
```

### Latency Benchmark
```python
results = evaluator.latency_benchmark(test_prompts)
print(f"Throughput: {results['throughput_tokens_per_sec']:.1f} tokens/sec")
```

## 🔗 Integration with FinIQ.ai

Add to FinIQ backend:

```python
from scripts.model_setup import SmolLM2Manager
from scripts.inference import SmolLM2Inference

# Load fine-tuned adapter
manager = SmolLM2Manager()
model, tokenizer = manager.load_model_unsloth()
model = manager.setup_lora()

# Inference
inference = SmolLM2Inference(model, tokenizer)
response = inference.chat_completion(user_query)
```

## 📚 Resources

- **SmolLM2 Docs**: [HF Model Card](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)
- **Unsloth**: [GitHub](https://github.com/unslothai/unsloth) | 2x speedup
- **trl (SFT)**: [Docs](https://huggingface.co/docs/trl) | Training library
- **LoRA Paper**: [arXiv](https://arxiv.org/abs/2106.09685)
- **Finance Datasets**: [HF Collections](https://huggingface.co/collections/financial-datasets)

## 🐛 Troubleshooting

| Issue | Fix |
|-------|-----|
| **CUDA OOM** | Reduce `per_device_batch_size`, increase `gradient_accumulation_steps` |
| **Loss diverging** | Lower learning rate (1e-4 to 5e-5) |
| **Slow training** | Install Unsloth (`pip install unsloth...`) |
| **Model not loading** | Ensure `torch`, `transformers` match versions |

## 📝 Citation

If using this project in research:

```bibtex
@misc{smollm2_finance_2024,
  title={SmolLM2 Finance Fine-Tuning Pipeline},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo}}
}
```

## 📄 License

MIT License - See LICENSE file

---

**Questions or Issues?** Open an issue or check the docs above.
