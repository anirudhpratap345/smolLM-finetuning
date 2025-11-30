# Gradio Web UI - Quick Start Guide

## Overview

The Gradio web UI provides an interactive interface for:
- 🔧 **Model Setup**: Configure SmolLM2 and LoRA parameters
- 📊 **Data Loading**: Load Financial PhraseBank or synthetic data
- 🎓 **Training**: Fine-tune the model with custom hyperparameters
- 🎯 **Inference**: Run multiple inference modes (Sentiment, Q&A, Generation)
- 📈 **Evaluation**: Assess model performance on finance tasks
- 💾 **Checkpoints**: Save and load trained models

## Installation

```bash
# Install Gradio and dependencies
pip install -r requirements.txt

# Or just Gradio
pip install gradio>=4.0.0
```

## Running the Web UI

### Local Machine (CPU/GPU)
```bash
python app.py
```

Access at: **http://localhost:7860**

### Colab (Google Colab)
```python
# In Colab cell
!pip install -q gradio transformers torch peft

# Clone and run
!git clone https://github.com/anirudhpratap345/smolLM-finetuning.git
%cd smolLM-finetuning
!python app.py
```

### Docker (Optional)
```bash
docker build -t smollm-finance .
docker run -p 7860:7860 smollm-finance
```

## Usage Workflow

### 1. Setup (🔧 Setup Tab)
- Click **"Initialize Model"** to setup SmolLM2
- Adjust LoRA parameters (R, Alpha) for fine-tuning efficiency
- Wait for model initialization

### 2. Load Data (📊 Data Tab)
- Choose dataset: Financial PhraseBank or Synthetic Demo
- Click **"Load Data"** to prepare training set
- See number of samples loaded

### 3. Train (🎓 Training Tab)
- Set training hyperparameters:
  - **Epochs**: 1-10 (3 recommended for demo)
  - **Batch Size**: 4-64 (8 for CPU)
  - **Learning Rate**: 1e-5 to 5e-4
  - **Warmup Steps**: 0-500
  - **Training Samples**: 10-1000
- Click **"Start Training"** to begin
- Monitor training status and loss history

### 4. Inference (🎯 Inference Tab)
Choose inference mode:
- **Sentiment Analysis**: Classify financial sentiment (Positive/Negative/Neutral)
- **Financial Q&A**: Generate answers to finance questions
- **General Text Generation**: Generate text completions
- **Market Insights**: Generate market analysis

Input your text and adjust **Temperature** (0.1-2.0):
- Low (0.1-0.5): More deterministic, focused
- Medium (0.7): Balanced
- High (1.5-2.0): More creative, random

### 5. Evaluate (📈 Evaluation Tab)
- Click **"Run Full Evaluation"** to test model on multiple metrics:
  - Sentiment Accuracy
  - Q&A F1 Score
  - Generation Quality Score
  - Inference Latency

### 6. Save/Load (💾 Checkpoints Tab)
- **Save**: Enter model name (e.g., "finance-model-v1") → Click Save
- **Load**: Enter path (e.g., "models/finance-model-v1") → Click Load

## Features

### 🎨 User Interface
- Clean, intuitive tabs for each workflow
- Real-time status updates
- Hardware information display
- System compatibility check

### ⚡ Performance
- GPU acceleration when available
- CPU-friendly defaults
- Batch processing support
- Optimized inference with batching

### 📊 Monitoring
- Training loss history visualization
- Real-time metrics display
- Model evaluation dashboard
- Checkpoint management

## Example Inputs

### Sentiment Analysis
```
Input: "Apple stock surged 5% today on strong earnings report"
Output: Sentiment: Positive | Confidence: 0.92
```

### Financial Q&A
```
Input: "What are the risks of cryptocurrency investment?"
Output: Q: What are the risks of cryptocurrency investment?
        A: Volatility, regulatory uncertainty, security risks...
```

### Market Insights
```
Input: "Tech sector showing bullish signals"
Output: Insights: The tech sector demonstrates strength with...
```

## Configuration

### Model Configuration
File: `config/training_config.py`
- Adjust model size, precision, LoRA rank
- Configure device (CPU/GPU)
- Set training parameters

### Data Configuration
File: `scripts/data_loader.py`
- Add custom datasets
- Modify preprocessing
- Implement domain-specific tokenization

### Training Configuration
File: `scripts/training_pipeline.py`
- Adjust loss functions
- Customize training loop
- Add callbacks and monitoring

## Troubleshooting

### Port Already in Use
```bash
# Run on different port
python app.py --port 7861
```

### Out of Memory (OOM)
- Reduce batch size
- Decrease number of training samples
- Use CPU-only mode

### Model Loading Fails
- Ensure model path is correct
- Check disk space
- Verify file permissions

### GPU Not Detected
- Check CUDA installation
- Verify PyTorch CUDA version
- Falls back to CPU automatically

## Advanced Usage

### Custom Datasets
```python
# In Data tab - extend dataset_choice dropdown
# Add your dataset loading logic in load_training_data()
```

### Custom Inference Modes
```python
# In Inference tab - add new inference_mode options
# Implement corresponding logic in run_inference()
```

### Export Results
```python
# Training history and metrics automatically saved
# Access via STATE['training_history']
# Export to JSON/CSV for analysis
```

## Performance Tips

1. **For CPU**: Use synthetic demo with 10-50 samples
2. **For GPU**: Increase batch size to 16-32
3. **For production**: Fine-tune with full Financial PhraseBank
4. **For speed**: Reduce epochs and warmup steps

## API Integration

To integrate trained models into other applications:

```python
from scripts.inference import SmolLM2Inference
from config.training_config import get_local_gpu_config

config = get_local_gpu_config()
inference = SmolLM2Inference(model, config)

# Run inference
result = inference.sentiment_analysis("Your text here")
```

## Support & Resources

- 📚 [README.md](README.md) - Full project documentation
- 🔗 [GitHub Repository](https://github.com/anirudhpratap345/smolLM-finetuning)
- 📖 [Gradio Documentation](https://gradio.app/docs)
- 🤗 [Hugging Face Documentation](https://huggingface.co/docs)

## License

This project is open source and available under the MIT License.
