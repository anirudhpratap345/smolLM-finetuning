"""
Training configuration for SmolLM2 fine-tuning on finance data.
Centralizes all hyperparameters, paths, and model settings.
"""

import torch
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model and quantization settings."""
    model_name: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    max_seq_length: int = 2048
    dtype: Optional[str] = None  # Auto (bfloat16 if supported, else fp16)
    load_in_4bit: bool = torch.cuda.is_available()  # Only if GPU available
    use_gradient_checkpointing: bool = True


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""
    r: int = 16  # Rank (8 for ultra-efficiency, 64 for quality)
    lora_alpha: int = 16  # Scaling factor
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_dropout: float = 0.05
    bias: str = "none"
    use_rslora: bool = False
    random_state: int = 3407


@dataclass
class TrainingConfig:
    """SFT training hyperparameters."""
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_steps: int = 60  # ~1 epoch for 1k samples; scale to 200-300 for full
    warmup_steps: int = 5
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    optim: str = "adamw_8bit"
    seed: int = 3407
    
    # Scheduling & precision
    lr_scheduler_type: str = "linear"
    fp16: bool = False  # Set based on GPU capability
    bf16: bool = True   # Preferred if supported
    
    # Logging & checkpointing
    logging_steps: int = 1
    save_steps: int = 50
    eval_steps: int = 50
    save_total_limit: int = 3
    
    # Output & tracking
    output_dir: str = "smollm2-finance-tuned"
    report_to: str = "none"  # "wandb" for W&B tracking
    
    # Data
    packing: bool = False  # Variable-length finance texts
    dataset_num_proc: int = 2


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset_name: str = "financial_phrasebank"  # HF dataset identifier
    dataset_config: str = "sentences_allagree"
    train_split: str = "train"
    max_samples: Optional[int] = 1000  # Subset for fast iteration
    test_size: float = 0.1
    random_seed: int = 42
    
    # Custom dataset path (if using local data)
    local_data_path: Optional[str] = None


@dataclass
class InferenceConfig:
    """Inference and evaluation settings."""
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 100
    do_sample: bool = True
    repetition_penalty: float = 1.1


# Complete configuration
@dataclass
class FinanceSFTConfig:
    """Complete fine-tuning configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


def get_default_config() -> FinanceSFTConfig:
    """Returns default configuration."""
    return FinanceSFTConfig()


def get_colab_config() -> FinanceSFTConfig:
    """Optimized configuration for free Colab (T4 GPU)."""
    config = FinanceSFTConfig()
    config.training.per_device_train_batch_size = 1
    config.training.gradient_accumulation_steps = 8
    config.training.max_steps = 60
    config.data.max_samples = 500
    return config


def get_local_gpu_config() -> FinanceSFTConfig:
    """Optimized for local GPU (RTX 3060/4070)."""
    config = FinanceSFTConfig()
    config.training.per_device_train_batch_size = 4
    config.training.gradient_accumulation_steps = 2
    config.training.max_steps = 200
    config.data.max_samples = 5000
    return config


def get_cpu_config() -> FinanceSFTConfig:
    """Configuration for CPU-only training (very slow, use for testing only)."""
    config = FinanceSFTConfig()
    config.model.load_in_4bit = False  # 4-bit not supported on CPU
    config.training.per_device_train_batch_size = 1
    config.training.gradient_accumulation_steps = 4
    config.training.max_steps = 10  # Very limited
    config.data.max_samples = 100  # Very small
    config.training.fp16 = False
    config.training.bf16 = False
    return config
