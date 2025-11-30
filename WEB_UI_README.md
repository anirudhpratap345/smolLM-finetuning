# Web UI Backend - Gradio & Streamlit

## Overview

Two interactive web interfaces for the SmolLM2 Finance Fine-Tuning project:

### 🎨 **Gradio** (`app.py`)
- Tabbed interface with 6 main sections
- Real-time status updates
- Easy deployment to Hugging Face Spaces
- Built-in sharing capabilities

### 🎯 **Streamlit** (`streamlit_app.py`)
- Clean sidebar navigation
- Session state management
- Caching for performance
- Dashboard-style metrics display

## Quick Start

### Prerequisites
```bash
pip install gradio>=4.0.0
# OR
pip install streamlit>=1.28.0
```

### Running Gradio UI
```bash
python app.py
```
Access at: **http://localhost:7860**

### Running Streamlit UI
```bash
streamlit run streamlit_app.py
```
Access at: **http://localhost:8501**

## Features Comparison

| Feature | Gradio | Streamlit |
|---------|--------|-----------|
| **Setup** | ✓ LoRA config | ✓ LoRA config |
| **Data Loading** | ✓ Multiple datasets | ✓ Multiple datasets |
| **Training** | ✓ Full control | ✓ Full control |
| **Inference** | ✓ 4 modes | ✓ 4 modes |
| **Evaluation** | ✓ Metrics display | ✓ Metrics cards |
| **Checkpoints** | ✓ Save/Load | ✓ Save/Load |
| **Hardware Info** | ✓ Top bar | ✓ Sidebar |
| **Deployment** | ✓ HF Spaces | ✓ Streamlit Cloud |
| **Performance** | Fast | Fast |
| **Learning Curve** | Easy | Very Easy |

## Detailed Usage

### 1. Model Setup
Both interfaces allow you to:
- Select model (SmolLM2-1.7B-Instruct)
- Configure LoRA parameters (R: 8-256, Alpha: 8-256)
- Initialize and verify model setup

**Hardware Recommendation:**
- R=16, Alpha=32 for most users
- R=8, Alpha=16 for memory-constrained systems
- R=32, Alpha=64 for high-end GPUs

### 2. Data Loading
Choose from:
- **Synthetic Demo**: 10-1000 generated financial samples (fast)
- **Financial PhraseBank**: Real financial sentiment data

**Recommended for demo:** Synthetic Demo (100 samples)

### 3. Training
Configure training parameters:
- **Epochs**: 1-10 (3 recommended)
- **Batch Size**: 4-64 (8 for CPU, 16-32 for GPU)
- **Learning Rate**: 1e-5 to 5e-4 (2e-4 default)
- **Warmup Steps**: 0-500 (100 default)
- **Training Samples**: 10-1000 (100 default)

**Training Time:**
- CPU (100 samples, 3 epochs): ~5-10 minutes
- GPU T4 (100 samples, 3 epochs): ~1-2 minutes
- GPU A100 (100 samples, 3 epochs): ~30 seconds

### 4. Inference Modes

#### Sentiment Analysis
- Classifies financial text as Positive/Negative/Neutral
- Example: "Stock market crashed 5%" → Negative (0.95)

#### Financial Q&A
- Generates Q&A pairs from financial text
- Example: "Interest rates are rising" → Q: Impact on bonds? A: Bond prices fall

#### Text Generation
- Generates continuations of financial text
- Temperature control for creativity (0.1-2.0)

#### Market Insights
- Analyzes market conditions
- Provides trading-relevant insights

### 5. Model Evaluation
Metrics computed:
- **Sentiment Accuracy**: Classification accuracy on test set
- **QA F1 Score**: Quality of Q&A generation
- **Generation Score**: Perplexity and coherence
- **Latency**: Inference speed in milliseconds

### 6. Checkpoint Management
- **Save**: Store trained adapters and configs
- **Load**: Resume from saved checkpoint
- Path format: `models/model-name/`

## File Structure

```
├── app.py                      # Gradio web UI
├── streamlit_app.py            # Streamlit web UI
├── GRADIO_GUIDE.md             # Gradio detailed guide
├── requirements.txt            # Python dependencies
│
├── config/
│   └── training_config.py      # Configuration dataclasses
├── scripts/
│   ├── data_loader.py          # Data processing
│   ├── model_setup.py          # Model initialization
│   ├── training_pipeline.py    # Training logic
│   ├── inference.py            # Inference engines
│   ├── hardware.py             # Hardware detection
│   └── utils.py                # Utility functions
├── models/                     # Saved checkpoints
└── data/                       # Training data
```

## API Reference

### Running Both UIs

**Terminal 1: Gradio**
```bash
python app.py
# http://localhost:7860
```

**Terminal 2: Streamlit (different port)**
```bash
streamlit run streamlit_app.py --server.port 8501
# http://localhost:8501
```

### Environment Variables

```bash
# Set device
export TORCH_DEVICE=cuda  # or cpu

# Set model cache
export HF_HOME=/path/to/cache

# Disable telemetry
export GRADIO_ANALYTICS_ENABLED=False
export STREAMLIT_LOGGER_LEVEL=warning
```

## Deployment Options

### Local Machine
```bash
# Gradio
python app.py --server_name 0.0.0.0 --server_port 7860

# Streamlit
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

### Hugging Face Spaces (Gradio)
1. Create Space on HuggingFace
2. Upload `app.py` and `requirements.txt`
3. Set environment variables
4. Deploy automatically

### Streamlit Cloud
1. Push repo to GitHub
2. Connect via Streamlit Cloud
3. Deploy automatically

### Docker
```dockerfile
FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
```

```bash
docker build -t smollm-finance .
docker run -p 7860:7860 smollm-finance
```

## Performance Optimization

### For CPU
```python
# In config
num_train_epochs: 1
per_device_train_batch_size: 4
num_samples: 10
device: cpu
```

### For GPU
```python
# In config
num_train_epochs: 3
per_device_train_batch_size: 16
num_samples: 100
device: cuda
```

### For Production
```python
# In config
num_train_epochs: 5
per_device_train_batch_size: 32
num_samples: 1000
device: cuda
use_fp16: True
```

## Troubleshooting

### Port Already in Use
```bash
# Gradio
python app.py --server_port 7861

# Streamlit
streamlit run streamlit_app.py --server.port 8502
```

### CUDA Out of Memory
1. Reduce batch size (4-8)
2. Reduce number of samples (10-50)
3. Use CPU mode
4. Use smaller LoRA rank (R=8)

### Model Loading Fails
```bash
# Clear cache
rm -rf ~/.cache/huggingface

# Reinstall dependencies
pip install --upgrade transformers peft
```

### GPU Not Detected
```bash
# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check version
python -c "import torch; print(torch.version.cuda)"
```

## Code Examples

### Custom Training Loop
```python
# In app.py, modify start_training()
def custom_training(num_epochs, data):
    trainer = FinanceSFTTrainer(STATE['model'], STATE['config'])
    results = trainer.train(data)
    return results
```

### Custom Inference Mode
```python
# In run_inference()
elif inference_mode == "Custom":
    result = STATE['inference_engine'].custom_function(text)
    return f"Result: {result}"
```

### Add New Dataset
```python
# In load_training_data()
elif dataset_choice == "Custom":
    data = load_custom_dataset("path/to/data")
    return data
```

## Performance Metrics

Typical performance on CPU/GPU:

| Task | CPU | GPU T4 | GPU A100 |
|------|-----|--------|---------|
| Model Setup | 30s | 30s | 30s |
| Data Loading (100 samples) | <1s | <1s | <1s |
| Training (3 epochs, 100 samples) | 5-10m | 1-2m | 30s |
| Inference | 2-5s | 100-200ms | 50-100ms |
| Evaluation | 30-60s | 5-10s | 2-5s |

## Contributing

To add features to the web UI:

1. Update `app.py` or `streamlit_app.py`
2. Add corresponding backend logic in `scripts/`
3. Update `requirements.txt` if needed
4. Test locally and submit PR

## Support

- 📚 [GRADIO_GUIDE.md](GRADIO_GUIDE.md) - Detailed Gradio guide
- 🔗 [GitHub](https://github.com/anirudhpratap345/smolLM-finetuning)
- 💬 Issues and discussions

## License

MIT License - See LICENSE file for details
