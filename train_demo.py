"""
Demo training script - Minimal working example for SmolLM2 fine-tuning.
Uses tiny synthetic dataset to validate the pipeline without GPU/large downloads.
"""

import logging
import sys
import torch
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from config.training_config import get_cpu_config
from scripts.hardware import print_hardware_info, get_recommended_config
from scripts.model_setup import SmolLM2Manager


def create_tiny_dataset(num_samples: int = 10):
    """Create tiny synthetic dataset for quick testing."""
    from datasets import Dataset
    
    texts = [
        "<|im_start|>user\nClassify sentiment: NVIDIA stock surged 15% after strong earnings.<|im_end|>\n<|im_start|>assistant\nPositive<|im_end|>\n",
        "<|im_start|>user\nClassify sentiment: Tesla fell 8% due to supply chain concerns.<|im_end|>\n<|im_start|>assistant\nNegative<|im_end|>\n",
        "<|im_start|>user\nClassify sentiment: Market is stable with mixed signals.<|im_end|>\n<|im_start|>assistant\nNeutral<|im_end|>\n",
        "<|im_start|>user\nClassify sentiment: Apple exceeded revenue expectations.<|im_end|>\n<|im_start|>assistant\nPositive<|im_end|>\n",
        "<|im_start|>user\nClassify sentiment: Tech sector faces headwinds.<|im_end|>\n<|im_start|>assistant\nNegative<|im_end|>\n",
    ]
    
    # Repeat to get desired number of samples
    texts = (texts * ((num_samples // len(texts)) + 1))[:num_samples]
    
    data = Dataset.from_dict({"text": texts})
    
    # Split
    split = data.train_test_split(test_size=0.2, seed=42)
    return split['train'], split['test']


def main_demo():
    """Demo training pipeline with minimal resources."""
    
    logger.info("="*80)
    logger.info("SmolLM2 Finance Fine-Tuning - DEMO MODE")
    logger.info("="*80)
    
    # Print hardware
    print_hardware_info()
    
    # Step 1: Create tiny dataset
    logger.info("\n[Step 1/3] Creating synthetic dataset...")
    train_data, eval_data = create_tiny_dataset(num_samples=10)
    logger.info(f"Created: Train={len(train_data)}, Eval={len(eval_data)}")
    logger.info(f"Sample: {train_data[0]['text'][:100]}...")
    
    # Step 2: Load model (no training, just verify it loads)
    logger.info("\n[Step 2/3] Loading SmolLM2 model...")
    try:
        manager = SmolLM2Manager("HuggingFaceTB/SmolLM2-1.7B-Instruct")
        logger.info("Attempting model load with CPU support...")
        
        # Try standard load (no 4-bit on CPU)
        manager.load_model_standard(max_seq_length=2048, load_in_4bit=False)
        model, tokenizer = manager.get_model_and_tokenizer()
        
        logger.info(f"Model loaded successfully!")
        logger.info(f"Model dtype: {model.dtype}")
        logger.info(f"Tokenizer vocab: {len(tokenizer)}")
        
        # Apply LoRA
        logger.info("\nApplying LoRA adapters...")
        manager.setup_lora(r=8, lora_alpha=16)  # Smaller for CPU
        logger.info("LoRA adapters applied!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("This is expected on CPU without sufficient memory.")
        logger.info("For actual training, use a GPU:")
        logger.info("  - Google Colab: https://colab.research.google.com")
        logger.info("  - Install CUDA: https://developer.nvidia.com/cuda-downloads")
        
        return None
    
    # Step 3: Verify inference capability
    logger.info("\n[Step 3/3] Testing inference...")
    try:
        from scripts.inference import SmolLM2Inference
        
        inference = SmolLM2Inference(model, tokenizer, temperature=0.1)
        
        test_prompt = "<|im_start|>user\nClassify sentiment: NVIDIA stock soared 20%<|im_end|>\n<|im_start|>assistant\n"
        response = inference.generate(test_prompt, max_new_tokens=10)
        
        logger.info(f"Test prompt: NVIDIA stock soared 20%")
        logger.info(f"Response: {response}")
        
    except Exception as e:
        logger.warning(f"Inference test failed: {e}")
    
    logger.info("\n" + "="*80)
    logger.info("DEMO COMPLETE!")
    logger.info("="*80)
    logger.info("\nFor FULL training with actual dataset:")
    logger.info("  1. Use Google Colab (free GPU): https://colab.research.google.com")
    logger.info("  2. Or install CUDA locally: https://developer.nvidia.com/cuda-downloads")
    logger.info("  3. Then run: python train.py")


if __name__ == "__main__":
    try:
        main_demo()
    except KeyboardInterrupt:
        logger.info("\nDemo interrupted by user.")
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        sys.exit(1)
