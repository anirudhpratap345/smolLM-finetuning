"""
Main training script for SmolLM2 fine-tuning on finance data.
Orchestrates data loading, model setup, training, and inference.
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

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

# Check GPU availability
if torch.cuda.is_available():
    logger.info(f"[OK] GPU detected: {torch.cuda.get_device_name(0)}")
    logger.info(f"     CUDA Version: {torch.version.cuda}")
    logger.info(f"     GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
else:
    logger.warning("[WARNING] No GPU detected. Training will be very slow on CPU.")

from config.training_config import get_local_gpu_config
from scripts.data_loader import load_financial_phrasebank
from scripts.model_setup import setup_smollm2
from scripts.training_pipeline import train_smollm2
from scripts.inference import SmolLM2Inference, FinanceEvaluator
from scripts.hardware import print_hardware_info, get_recommended_config


def main():
    """Main training pipeline."""
    
    logger.info("="*80)
    logger.info("SmolLM2 Finance Fine-Tuning Pipeline")
    logger.info("="*80)
    
    # Print hardware info
    print_hardware_info()
    
    # 1. Load configuration
    logger.info("\n[Step 1/5] Loading configuration...")
    config = get_recommended_config()  # Auto-detect best config
    logger.info(f"Config loaded: {config.model.model_name}")
    
    # 2. Load dataset
    logger.info("\n[Step 2/5] Loading dataset...")
    train_dataset, eval_dataset = load_financial_phrasebank(
        max_samples=config.data.max_samples,
        test_size=config.data.test_size,
        seed=config.data.random_seed
    )
    logger.info(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")
    
    # 3. Setup model with LoRA
    logger.info("\n[Step 3/5] Setting up SmolLM2 with LoRA...")
    model, tokenizer = setup_smollm2(
        model_name=config.model.model_name,
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=config.model.load_in_4bit,
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha
    )
    logger.info("Model ready for training")
    
    # 4. Train
    logger.info("\n[Step 4/5] Training...")
    result, output_dir = train_smollm2(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=config.training.output_dir,
        per_device_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        max_steps=config.training.max_steps,
        warmup_steps=config.training.warmup_steps,
        report_to=config.training.report_to,
        save_merged=False
    )
    logger.info(f"Training complete. Model saved to {output_dir}")
    
    # 5. Quick inference test
    logger.info("\n[Step 5/5] Testing inference...")
    inference = SmolLM2Inference(model, tokenizer)
    
    test_prompt = "Classify sentiment: NVIDIA shares rose 12% on AI chip demand."
    response = inference.chat_completion(test_prompt)
    logger.info(f"Test prompt: {test_prompt}")
    logger.info(f"Response: {response}")
    
    logger.info("\n" + "="*80)
    logger.info("Pipeline complete!")
    logger.info("="*80)
    
    return output_dir


if __name__ == "__main__":
    try:
        output_dir = main()
        print(f"\n✅ Training successful. Model saved to: {output_dir}")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
