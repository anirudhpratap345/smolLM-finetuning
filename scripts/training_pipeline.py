"""
Training pipeline for SmolLM2 fine-tuning on finance data.
Uses trl's SFTTrainer for production-grade supervised fine-tuning.
"""

import logging
import torch
from typing import Optional, Tuple
from pathlib import Path

from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from transformers import TrainingArguments

logger = logging.getLogger(__name__)


class FinanceSFTTrainer:
    """Wrapper for SFTTrainer with finance-specific defaults."""
    
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        output_dir: str = "smollm2-finance-tuned",
        per_device_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        max_steps: int = 60,
        warmup_steps: int = 5,
        report_to: str = "none",
        use_bf16: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model with LoRA adapters
            tokenizer: Tokenizer
            train_dataset: Training data
            eval_dataset: Evaluation data (optional)
            output_dir: Output directory
            per_device_batch_size: Batch size per GPU
            gradient_accumulation_steps: Gradient accumulation
            learning_rate: Learning rate
            max_steps: Training steps
            warmup_steps: Warmup steps
            report_to: Logging backend ('none', 'wandb')
            use_bf16: Use bfloat16 precision
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        self.trainer = None
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Determine precision
        fp16 = not use_bf16 and not torch.cuda.is_bf16_supported()
        bf16 = use_bf16 and torch.cuda.is_bf16_supported()
        
        # Disable mixed precision on CPU
        if not torch.cuda.is_available():
            fp16 = False
            bf16 = False
            logger.warning("Mixed precision disabled on CPU")
        
        logger.info(f"Training precision - FP16: {fp16}, BF16: {bf16}")
        
        # SFT Config
        self.sft_config = SFTConfig(
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            max_steps=max_steps,
            warmup_steps=warmup_steps,
            max_grad_norm=1.0,
            weight_decay=0.01,
            
            # Optimizer
            optim="adamw_8bit",
            
            # Precision
            fp16=fp16,
            bf16=bf16,
            
            # Scheduling
            lr_scheduler_type="linear",
            
            # Logging & checkpointing
            logging_steps=1,
            save_steps=max(1, max_steps // 2),
            eval_steps=max(1, max_steps // 2),
            save_total_limit=3,
            
            # Output
            output_dir=output_dir,
            report_to=report_to,
            logging_dir=f"{output_dir}/logs",
            
            # Data
            packing=False,  # Variable-length finance texts
            dataset_num_proc=2,
            
            # Misc
            seed=3407,
        )
        
        logger.info(f"SFT Config:\n{self.sft_config}")
    
    def train(self) -> dict:
        """
        Run training.
        
        Returns:
            Training results
        """
        logger.info("Initializing SFTTrainer...")
        
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            dataset_text_field="text",
            args=self.sft_config,
        )
        
        logger.info("Starting training...")
        result = self.trainer.train()
        
        logger.info(f"Training complete. Results:\n{result}")
        
        return result
    
    def save_model(self, save_merged: bool = False):
        """
        Save trained model (adapter or merged).
        
        Args:
            save_merged: Save merged model (base + adapter) or just adapter
        """
        if save_merged:
            logger.info(f"Saving merged model to {self.output_dir}/merged")
            try:
                self.model.save_pretrained_merged(
                    f"{self.output_dir}/merged",
                    self.tokenizer,
                    save_method="merged_16bit"
                )
            except Exception as e:
                logger.warning(f"Merged save failed: {e}. Saving adapter only.")
                self.model.save_pretrained(self.output_dir)
                self.tokenizer.save_pretrained(self.output_dir)
        else:
            logger.info(f"Saving LoRA adapter to {self.output_dir}")
            self.model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)


def train_smollm2(
    model,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    output_dir: str = "smollm2-finance-tuned",
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    max_steps: int = 60,
    warmup_steps: int = 5,
    report_to: str = "none",
    save_merged: bool = False
) -> Tuple[dict, str]:
    """
    Convenience function to train SmolLM2.
    
    Args:
        model: Model with LoRA adapters
        tokenizer: Tokenizer
        train_dataset: Training data
        eval_dataset: Evaluation data
        output_dir: Output directory
        per_device_batch_size: Batch size
        gradient_accumulation_steps: Gradient accumulation
        learning_rate: Learning rate
        max_steps: Training steps
        warmup_steps: Warmup steps
        report_to: Logging backend
        save_merged: Save merged model
    
    Returns:
        (training_results, output_dir)
    """
    trainer = FinanceSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        report_to=report_to
    )
    
    result = trainer.train()
    trainer.save_model(save_merged=save_merged)
    
    return result, output_dir


if __name__ == "__main__":
    print("Training module imported successfully. Use in main training scripts.")
