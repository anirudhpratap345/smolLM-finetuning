"""
Gradio Web UI for SmolLM2 Finance Fine-Tuning
Interactive interface for training, inference, and evaluation
"""

import os
import sys
import json
import torch
import logging
from pathlib import Path
from typing import Tuple, List
import gradio as gr
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from config.training_config import (
    get_colab_config, get_local_gpu_config, get_cpu_config,
    FinanceSFTConfig
)
from scripts.data_loader import FinanceDataProcessor, load_financial_phrasebank
from scripts.model_setup import SmolLM2Manager
from scripts.training_pipeline import FinanceSFTTrainer
from scripts.inference import SmolLM2Inference, FinanceEvaluator
from scripts.hardware import get_recommended_config

# Global state
STATE = {
    'model': None,
    'trainer': None,
    'inference_engine': None,
    'config': None,
    'training_history': [],
    'evaluator': None
}


def get_hardware_info() -> str:
    """Get system hardware information."""
    info = []
    info.append("=== System Information ===")
    info.append(f"PyTorch Version: {torch.__version__}")
    
    if torch.cuda.is_available():
        info.append(f"GPU: {torch.cuda.get_device_name(0)}")
        info.append(f"CUDA Version: {torch.version.cuda}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        info.append(f"GPU Memory: {total_memory:.1f}GB")
    else:
        info.append("GPU: Not detected (CPU mode)")
    
    info.append(f"Recommended Config: CPU" if not torch.cuda.is_available() else "GPU")
    return "\n".join(info)


def setup_model(model_choice: str, lora_r: int, lora_alpha: int) -> Tuple[str, str]:
    """Initialize and setup SmolLM2 model."""
    try:
        logger.info(f"Setting up model: {model_choice}")
        
        # Get config based on hardware
        STATE['config'] = get_recommended_config()
        
        # Update LoRA parameters
        STATE['config'].lora.lora_r = lora_r
        STATE['config'].lora.lora_alpha = lora_alpha
        
        # Setup model
        STATE['model'] = SmolLM2Manager(STATE['config'])
        STATE['inference_engine'] = SmolLM2Inference(STATE['model'].model, STATE['config'])
        STATE['evaluator'] = FinanceEvaluator(STATE['inference_engine'], STATE['config'])
        
        message = f"""
        Model Setup Complete
        ✓ Model: {model_choice}
        ✓ LoRA R: {lora_r}
        ✓ LoRA Alpha: {lora_alpha}
        ✓ Device: {STATE['config'].training.device}
        ✓ Parameters: {sum(p.numel() for p in STATE['model'].model.parameters() if p.requires_grad):,}
        """
        
        return message.strip(), "Model ready for training!"
        
    except Exception as e:
        error_msg = f"Error setting up model: {str(e)}"
        logger.error(error_msg)
        return error_msg, ""


def load_training_data(dataset_choice: str) -> Tuple[str, int]:
    """Load training dataset."""
    try:
        logger.info(f"Loading dataset: {dataset_choice}")
        
        if dataset_choice == "Financial PhraseBank":
            data = load_financial_phrasebank()
        elif dataset_choice == "Synthetic Demo":
            processor = FinanceDataProcessor()
            data = processor.create_synthetic_finance_data(n_samples=10)
        else:
            raise ValueError(f"Unknown dataset: {dataset_choice}")
        
        num_samples = len(data) if isinstance(data, list) else len(data['train'])
        message = f"✓ Dataset loaded: {num_samples} samples"
        
        return message, num_samples
        
    except Exception as e:
        error_msg = f"Error loading dataset: {str(e)}"
        logger.error(error_msg)
        return error_msg, 0


def start_training(
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    warmup_steps: int,
    num_samples: int = 100
) -> Tuple[str, List]:
    """Start training process."""
    try:
        if STATE['model'] is None:
            return "Error: Model not initialized. Setup model first!", []
        
        logger.info("Starting training...")
        
        # Update training config
        STATE['config'].training.num_train_epochs = num_epochs
        STATE['config'].training.per_device_train_batch_size = batch_size
        STATE['config'].training.learning_rate = learning_rate
        STATE['config'].training.warmup_steps = warmup_steps
        
        # Create trainer
        STATE['trainer'] = FinanceSFTTrainer(STATE['model'], STATE['config'])
        
        # Create synthetic data for demo
        processor = FinanceDataProcessor()
        train_data = processor.create_synthetic_finance_data(n_samples=num_samples)
        
        # Train
        results = STATE['trainer'].train(train_data)
        
        # Store history
        STATE['training_history'] = results.get('history', [])
        
        message = f"""
        Training Complete!
        ✓ Epochs: {num_epochs}
        ✓ Batch Size: {batch_size}
        ✓ Learning Rate: {learning_rate}
        ✓ Final Loss: {results.get('final_loss', 'N/A')}
        """
        
        # Create training history chart data
        history = STATE['training_history']
        
        return message.strip(), history
        
    except Exception as e:
        error_msg = f"Error during training: {str(e)}"
        logger.error(error_msg)
        return error_msg, []


def run_inference(text: str, inference_mode: str, temperature: float = 0.7) -> str:
    """Run inference on input text."""
    try:
        if STATE['inference_engine'] is None:
            return "Error: Model not initialized. Setup model first!"
        
        logger.info(f"Running inference: {inference_mode}")
        
        if inference_mode == "Sentiment Analysis":
            result = STATE['inference_engine'].sentiment_analysis(text)
            return f"Sentiment: {result['sentiment']}\nConfidence: {result['score']:.2f}"
            
        elif inference_mode == "Financial Q&A":
            result = STATE['inference_engine'].generate_qa(text)
            return f"Q: {result['question']}\nA: {result['answer']}"
            
        elif inference_mode == "General Text Generation":
            result = STATE['inference_engine'].generate(text, max_tokens=100, temperature=temperature)
            return f"Generated:\n{result}"
            
        elif inference_mode == "Market Insights":
            result = STATE['inference_engine'].generate(f"Market analysis: {text}", max_tokens=150)
            return f"Insights:\n{result}"
        
        return "Unknown inference mode"
        
    except Exception as e:
        error_msg = f"Error during inference: {str(e)}"
        logger.error(error_msg)
        return error_msg


def evaluate_model() -> str:
    """Evaluate model on multiple metrics."""
    try:
        if STATE['evaluator'] is None:
            return "Error: Model not initialized. Setup model first!"
        
        logger.info("Running model evaluation...")
        
        # Run evaluation
        metrics = STATE['evaluator'].evaluate()
        
        result = f"""
        Model Evaluation Results
        ========================
        Sentiment Accuracy: {metrics.get('sentiment_accuracy', 0):.2%}
        QA F1 Score: {metrics.get('qa_f1', 0):.2f}
        Generation Score: {metrics.get('generation_score', 0):.2f}
        Inference Latency: {metrics.get('latency', 0):.2f}ms
        """
        
        return result.strip()
        
    except Exception as e:
        error_msg = f"Error during evaluation: {str(e)}"
        logger.error(error_msg)
        return error_msg


def save_model(model_name: str) -> str:
    """Save trained model."""
    try:
        if STATE['model'] is None:
            return "Error: No model to save"
        
        save_path = Path(f"models/{model_name}")
        save_path.mkdir(parents=True, exist_ok=True)
        
        STATE['model'].save_model(str(save_path))
        
        return f"✓ Model saved to {save_path}"
        
    except Exception as e:
        error_msg = f"Error saving model: {str(e)}"
        logger.error(error_msg)
        return error_msg


def load_model_checkpoint(model_path: str) -> str:
    """Load saved model."""
    try:
        if not Path(model_path).exists():
            return f"Error: Model path not found: {model_path}"
        
        STATE['model'] = SmolLM2Manager(STATE['config'])
        STATE['model'].load_model(model_path)
        STATE['inference_engine'] = SmolLM2Inference(STATE['model'].model, STATE['config'])
        
        return f"✓ Model loaded from {model_path}"
        
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        logger.error(error_msg)
        return error_msg


# ==================== GRADIO UI ====================

def create_ui():
    """Create Gradio interface with tabs."""
    
    with gr.Blocks(title="SmolLM2 Finance AI", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("# 🚀 SmolLM2 Finance Fine-Tuning Interface")
        gr.Markdown("Interactive training and inference for financial NLP models")
        
        # System Info
        with gr.Row():
            with gr.Column():
                gr.Markdown("### System Status")
                hw_info = gr.Textbox(
                    value=get_hardware_info(),
                    label="Hardware Info",
                    interactive=False,
                    lines=6
                )
        
        # ==================== SETUP TAB ====================
        with gr.Tab("🔧 Setup"):
            gr.Markdown("### Model Configuration")
            
            with gr.Row():
                model_choice = gr.Dropdown(
                    ["SmolLM2-1.7B-Instruct"],
                    value="SmolLM2-1.7B-Instruct",
                    label="Model"
                )
                lora_r = gr.Slider(8, 256, value=16, step=8, label="LoRA R")
                lora_alpha = gr.Slider(8, 256, value=32, step=8, label="LoRA Alpha")
            
            setup_btn = gr.Button("Initialize Model", variant="primary", size="lg")
            setup_status = gr.Textbox(label="Status", interactive=False)
            setup_msg = gr.Textbox(label="Message", interactive=False)
            
            setup_btn.click(
                setup_model,
                inputs=[model_choice, lora_r, lora_alpha],
                outputs=[setup_status, setup_msg]
            )
        
        # ==================== DATA TAB ====================
        with gr.Tab("📊 Data"):
            gr.Markdown("### Load Training Dataset")
            
            with gr.Row():
                dataset_choice = gr.Dropdown(
                    ["Financial PhraseBank", "Synthetic Demo"],
                    value="Synthetic Demo",
                    label="Dataset"
                )
                load_btn = gr.Button("Load Data", variant="primary")
            
            data_status = gr.Textbox(label="Status", interactive=False)
            num_samples = gr.Number(label="Samples Loaded", interactive=False)
            
            load_btn.click(
                load_training_data,
                inputs=[dataset_choice],
                outputs=[data_status, num_samples]
            )
        
        # ==================== TRAINING TAB ====================
        with gr.Tab("🎓 Training"):
            gr.Markdown("### Fine-Tune Model")
            
            with gr.Row():
                num_epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")
                batch_size = gr.Slider(4, 64, value=8, step=4, label="Batch Size")
            
            with gr.Row():
                learning_rate = gr.Number(value=2e-4, label="Learning Rate")
                warmup_steps = gr.Slider(0, 500, value=100, step=10, label="Warmup Steps")
            
            with gr.Row():
                num_train_samples = gr.Slider(10, 1000, value=100, step=10, label="Training Samples")
            
            train_btn = gr.Button("Start Training", variant="primary", size="lg")
            train_status = gr.Textbox(label="Training Status", interactive=False, lines=5)
            
            # Training history (placeholder)
            train_history = gr.Json(label="Training History")
            
            train_btn.click(
                start_training,
                inputs=[num_epochs, batch_size, learning_rate, warmup_steps, num_train_samples],
                outputs=[train_status, train_history]
            )
        
        # ==================== INFERENCE TAB ====================
        with gr.Tab("🎯 Inference"):
            gr.Markdown("### Run Inference")
            
            with gr.Row():
                inference_mode = gr.Radio(
                    ["Sentiment Analysis", "Financial Q&A", "General Text Generation", "Market Insights"],
                    value="Sentiment Analysis",
                    label="Inference Mode"
                )
                temperature = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature")
            
            input_text = gr.Textbox(
                label="Input Text",
                placeholder="Enter financial text...",
                lines=4
            )
            
            infer_btn = gr.Button("Run Inference", variant="primary", size="lg")
            output_text = gr.Textbox(
                label="Output",
                interactive=False,
                lines=6
            )
            
            infer_btn.click(
                run_inference,
                inputs=[input_text, inference_mode, temperature],
                outputs=[output_text]
            )
        
        # ==================== EVALUATION TAB ====================
        with gr.Tab("📈 Evaluation"):
            gr.Markdown("### Model Evaluation")
            
            eval_btn = gr.Button("Run Full Evaluation", variant="primary", size="lg")
            eval_results = gr.Textbox(
                label="Evaluation Results",
                interactive=False,
                lines=6
            )
            
            eval_btn.click(
                evaluate_model,
                outputs=[eval_results]
            )
        
        # ==================== SAVE/LOAD TAB ====================
        with gr.Tab("💾 Checkpoints"):
            gr.Markdown("### Save and Load Models")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Save Model")
                    save_name = gr.Textbox(
                        label="Model Name",
                        placeholder="e.g., finance-model-v1"
                    )
                    save_btn = gr.Button("Save", variant="primary")
                    save_status = gr.Textbox(label="Status", interactive=False)
                    
                    save_btn.click(
                        save_model,
                        inputs=[save_name],
                        outputs=[save_status]
                    )
                
                with gr.Column():
                    gr.Markdown("#### Load Model")
                    load_path = gr.Textbox(
                        label="Model Path",
                        placeholder="e.g., models/finance-model-v1"
                    )
                    load_btn = gr.Button("Load", variant="primary")
                    load_status = gr.Textbox(label="Status", interactive=False)
                    
                    load_btn.click(
                        load_model_checkpoint,
                        inputs=[load_path],
                        outputs=[load_status]
                    )
        
        # Footer
        gr.Markdown("""
        ---
        **SmolLM2 Finance Fine-Tuning** | Powered by Hugging Face Transformers & PEFT
        
        📚 [Documentation](README.md) | 💻 [GitHub](https://github.com/anirudhpratap345/smolLM-finetuning)
        """)
    
    return demo


if __name__ == "__main__":
    print("Starting Gradio Web UI for SmolLM2 Finance Fine-Tuning...")
    print("Access at: http://localhost:7860")
    
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
