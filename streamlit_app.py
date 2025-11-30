"""
Streamlit Web UI for SmolLM2 Finance Fine-Tuning
Alternative interactive interface with caching and session state
"""

import os
import sys
import json
import torch
import logging
import streamlit as st
from pathlib import Path
from typing import Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from config.training_config import get_recommended_config
from scripts.data_loader import FinanceDataProcessor, load_financial_phrasebank
from scripts.model_setup import SmolLM2Manager
from scripts.training_pipeline import FinanceSFTTrainer
from scripts.inference import SmolLM2Inference, FinanceEvaluator
from scripts.hardware import get_recommended_config

# Page configuration
st.set_page_config(
    page_title="SmolLM2 Finance AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== SESSION STATE ====================
if 'model' not in st.session_state:
    st.session_state.model = None
if 'config' not in st.session_state:
    st.session_state.config = None
if 'inference_engine' not in st.session_state:
    st.session_state.inference_engine = None
if 'trainer' not in st.session_state:
    st.session_state.trainer = None
if 'evaluator' not in st.session_state:
    st.session_state.evaluator = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = []

# ==================== UI COMPONENTS ====================

def display_header():
    """Display main header and system info."""
    st.title("🚀 SmolLM2 Finance Fine-Tuning")
    st.markdown("Interactive training and inference for financial NLP models")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("PyTorch Version", torch.__version__)
    
    with col2:
        if torch.cuda.is_available():
            st.metric("GPU", f"{torch.cuda.get_device_name(0)}")
        else:
            st.metric("Compute", "CPU")
    
    with col3:
        if torch.cuda.is_available():
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            st.metric("GPU Memory", f"{memory:.1f}GB")
        else:
            st.metric("Status", "Ready")


def setup_section():
    """Model setup section."""
    st.header("🔧 Model Setup")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model_choice = st.selectbox(
            "Model",
            ["SmolLM2-1.7B-Instruct"]
        )
    
    with col2:
        lora_r = st.slider("LoRA R", 8, 256, 16, step=8)
    
    with col3:
        lora_alpha = st.slider("LoRA Alpha", 8, 256, 32, step=8)
    
    if st.button("Initialize Model", key="setup_btn", use_container_width=True):
        with st.spinner("Setting up model..."):
            try:
                st.session_state.config = get_recommended_config()
                st.session_state.config.lora.lora_r = lora_r
                st.session_state.config.lora.lora_alpha = lora_alpha
                
                st.session_state.model = SmolLM2Manager(st.session_state.config)
                st.session_state.inference_engine = SmolLM2Inference(
                    st.session_state.model.model,
                    st.session_state.config
                )
                st.session_state.evaluator = FinanceEvaluator(
                    st.session_state.inference_engine,
                    st.session_state.config
                )
                
                st.success("✓ Model initialized successfully!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Model**: {model_choice}")
                with col2:
                    st.info(f"**Params**: {sum(p.numel() for p in st.session_state.model.model.parameters() if p.requires_grad):,}")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")


def data_section():
    """Data loading section."""
    st.header("📊 Data Loading")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        dataset_choice = st.selectbox(
            "Dataset",
            ["Synthetic Demo", "Financial PhraseBank"]
        )
    
    with col2:
        num_samples = st.number_input("Samples", min_value=10, max_value=1000, value=100, step=10)
    
    if st.button("Load Data", key="load_btn", use_container_width=True):
        with st.spinner("Loading dataset..."):
            try:
                if dataset_choice == "Financial PhraseBank":
                    data = load_financial_phrasebank()
                else:
                    processor = FinanceDataProcessor()
                    data = processor.create_synthetic_finance_data(n_samples=num_samples)
                
                num_loaded = len(data) if isinstance(data, list) else len(data['train'])
                st.success(f"✓ Loaded {num_loaded} samples")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")


def training_section():
    """Training section."""
    st.header("🎓 Fine-Tuning")
    
    if st.session_state.model is None:
        st.warning("⚠️ Model not initialized. Go to Setup tab first.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_epochs = st.slider("Epochs", 1, 10, 3)
        learning_rate = st.number_input("Learning Rate", 1e-5, 5e-4, 2e-4, format="%.2e")
    
    with col2:
        batch_size = st.slider("Batch Size", 4, 64, 8, step=4)
        warmup_steps = st.slider("Warmup Steps", 0, 500, 100, step=10)
    
    num_train_samples = st.slider("Training Samples", 10, 1000, 100, step=10)
    
    if st.button("Start Training", key="train_btn", use_container_width=True, type="primary"):
        with st.spinner("Training model..."):
            try:
                # Update config
                st.session_state.config.training.num_train_epochs = num_epochs
                st.session_state.config.training.per_device_train_batch_size = batch_size
                st.session_state.config.training.learning_rate = learning_rate
                st.session_state.config.training.warmup_steps = warmup_steps
                
                # Create trainer
                st.session_state.trainer = FinanceSFTTrainer(
                    st.session_state.model,
                    st.session_state.config
                )
                
                # Create synthetic data
                processor = FinanceDataProcessor()
                train_data = processor.create_synthetic_finance_data(n_samples=num_train_samples)
                
                # Train
                results = st.session_state.trainer.train(train_data)
                st.session_state.training_history = results.get('history', [])
                
                st.success("✓ Training complete!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Final Loss", f"{results.get('final_loss', 'N/A'):.4f}")
                with col2:
                    st.metric("Epochs", num_epochs)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")


def inference_section():
    """Inference section."""
    st.header("🎯 Inference")
    
    if st.session_state.inference_engine is None:
        st.warning("⚠️ Model not initialized. Go to Setup tab first.")
        return
    
    mode = st.radio(
        "Inference Mode",
        ["Sentiment Analysis", "Financial Q&A", "Text Generation", "Market Insights"],
        horizontal=True
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        input_text = st.text_area("Input Text", height=100, placeholder="Enter financial text...")
    
    with col2:
        temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
    
    if st.button("Run Inference", key="infer_btn", use_container_width=True, type="primary"):
        if not input_text.strip():
            st.warning("Enter some text first!")
            return
        
        with st.spinner("Running inference..."):
            try:
                if mode == "Sentiment Analysis":
                    result = st.session_state.inference_engine.sentiment_analysis(input_text)
                    st.write(f"**Sentiment**: {result.get('sentiment', 'N/A')}")
                    st.progress(result.get('score', 0.5))
                
                elif mode == "Financial Q&A":
                    result = st.session_state.inference_engine.generate_qa(input_text)
                    st.write(f"**Q**: {result.get('question', 'N/A')}")
                    st.write(f"**A**: {result.get('answer', 'N/A')}")
                
                else:
                    result = st.session_state.inference_engine.generate(
                        f"{mode}: {input_text}",
                        max_tokens=200,
                        temperature=temperature
                    )
                    st.write(result)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")


def evaluation_section():
    """Evaluation section."""
    st.header("📈 Model Evaluation")
    
    if st.session_state.evaluator is None:
        st.warning("⚠️ Model not initialized. Go to Setup tab first.")
        return
    
    if st.button("Run Evaluation", key="eval_btn", use_container_width=True, type="primary"):
        with st.spinner("Running evaluation..."):
            try:
                metrics = st.session_state.evaluator.evaluate()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Sentiment Acc.", f"{metrics.get('sentiment_accuracy', 0):.1%}")
                
                with col2:
                    st.metric("QA F1 Score", f"{metrics.get('qa_f1', 0):.2f}")
                
                with col3:
                    st.metric("Gen. Score", f"{metrics.get('generation_score', 0):.2f}")
                
                with col4:
                    st.metric("Latency", f"{metrics.get('latency', 0):.1f}ms")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")


def checkpoint_section():
    """Model checkpointing section."""
    st.header("💾 Model Checkpoints")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Save Model")
        model_name = st.text_input("Model Name", placeholder="finance-model-v1")
        if st.button("Save", key="save_btn", use_container_width=True):
            if st.session_state.model is None:
                st.error("No model to save")
            elif not model_name.strip():
                st.error("Enter model name")
            else:
                with st.spinner("Saving..."):
                    try:
                        save_path = Path(f"models/{model_name}")
                        save_path.mkdir(parents=True, exist_ok=True)
                        st.session_state.model.save_model(str(save_path))
                        st.success(f"✓ Saved to {save_path}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    with col2:
        st.subheader("Load Model")
        model_path = st.text_input("Model Path", placeholder="models/finance-model-v1")
        if st.button("Load", key="load_checkpoint_btn", use_container_width=True):
            if not model_path.strip():
                st.error("Enter model path")
            else:
                with st.spinner("Loading..."):
                    try:
                        if st.session_state.config is None:
                            st.error("Setup model config first")
                        else:
                            st.session_state.model = SmolLM2Manager(st.session_state.config)
                            st.session_state.model.load_model(model_path)
                            st.session_state.inference_engine = SmolLM2Inference(
                                st.session_state.model.model,
                                st.session_state.config
                            )
                            st.success(f"✓ Loaded from {model_path}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")


# ==================== MAIN APP ====================

def main():
    """Main Streamlit app."""
    
    display_header()
    
    # Sidebar navigation
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio(
        "Select Section",
        ["Setup", "Data", "Training", "Inference", "Evaluation", "Checkpoints"]
    )
    
    # Hardware info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Info")
    st.sidebar.write(f"**PyTorch**: {torch.__version__}")
    st.sidebar.write(f"**GPU**: {'Available' if torch.cuda.is_available() else 'Not available'}")
    
    # Main content
    st.markdown("---")
    
    if page == "Setup":
        setup_section()
    elif page == "Data":
        data_section()
    elif page == "Training":
        training_section()
    elif page == "Inference":
        inference_section()
    elif page == "Evaluation":
        evaluation_section()
    elif page == "Checkpoints":
        checkpoint_section()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **SmolLM2 Finance Fine-Tuning** | Built with ❤️ using Streamlit
    
    [📚 Documentation](GRADIO_GUIDE.md) | [💻 GitHub](https://github.com/anirudhpratap345/smolLM-finetuning)
    """)


if __name__ == "__main__":
    main()
