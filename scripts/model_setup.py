"""
Model setup and LoRA configuration for SmolLM2.
Uses Unsloth for 2x faster training and reduced VRAM (with fallback to standard transformers).
"""

import logging
import torch
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Try to import Unsloth (optional, with fallback)
FastLanguageModel = None
UNSLOTH_AVAILABLE = False

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
    logger.info("✓ Unsloth available - will use optimized loading")
except (ImportError, RuntimeError) as e:
    logger.warning(f"Unsloth not available ({type(e).__name__}). Using standard transformers (slower).")
    logger.info("  To use Unsloth, install: pip install -q 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'")
    FastLanguageModel = None

from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM


class SmolLM2Manager:
    """Manage SmolLM2 model loading, LoRA setup, and quantization."""
    
    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"):
        """
        Initialize model manager.
        
        Args:
            model_name: Model identifier on HF Hub
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
    
    def load_model_unsloth(
        self,
        max_seq_length: int = 2048,
        dtype: str = None,
        load_in_4bit: bool = True
    ) -> Tuple:
        """
        Load model with Unsloth optimization (or fallback to standard if unavailable).
        
        Args:
            max_seq_length: Context window (2048 or 8192)
            dtype: Data type (None for auto: bfloat16 if supported, else fp16)
            load_in_4bit: Quantize to 4-bit
        
        Returns:
            (model, tokenizer)
        """
        if UNSLOTH_AVAILABLE:
            logger.info(f"Loading {self.model_name} with Unsloth (4-bit: {load_in_4bit})")
            
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=max_seq_length,
                dtype=dtype,  # Auto
                load_in_4bit=load_in_4bit,
            )
            
            logger.info(f"Model loaded via Unsloth. Parameters: {self._count_params(self.model):,}")
        else:
            logger.info(f"Loading {self.model_name} with standard transformers (Unsloth unavailable)")
            self.load_model_standard(max_seq_length, load_in_4bit)
        
        return self.model, self.tokenizer
    
    def load_model_standard(
        self,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True
    ) -> Tuple:
        """
        Load model with standard HF transformers (fallback if Unsloth unavailable).
        
        Args:
            max_seq_length: Context window
            load_in_4bit: Use 4-bit quantization
        
        Returns:
            (model, tokenizer)
        """
        logger.info(f"Loading {self.model_name} with standard transformers")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        kwargs = {"torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32}
        if load_in_4bit:
            kwargs["load_in_4bit"] = True
            kwargs["device_map"] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        
        logger.info(f"Model loaded. Parameters: {self._count_params(self.model):,}")
        return self.model, self.tokenizer
    
    @staticmethod
    def _count_params(model) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def setup_lora(
        self,
        r: int = 16,
        lora_alpha: int = 16,
        target_modules: list = None,
        lora_dropout: float = 0.05,
        bias: str = "none",
        use_gradient_checkpointing: bool = True
    ):
        """
        Apply LoRA to model.
        
        Args:
            r: LoRA rank (8=ultra-efficient, 16=balanced, 64=high-quality)
            lora_alpha: Scaling factor
            target_modules: Modules to apply LoRA to
            lora_dropout: Dropout for LoRA
            bias: Bias type
            use_gradient_checkpointing: Memory optimization
        """
        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        
        logger.info(f"Applying LoRA (r={r}, alpha={lora_alpha})")
        
        # Try Unsloth first
        if UNSLOTH_AVAILABLE:
            try:
                self.model = FastLanguageModel.get_peft_model(
                    self.model,
                    r=r,
                    target_modules=target_modules,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias=bias,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    random_state=3407,
                )
                logger.info("LoRA applied via Unsloth")
            except Exception as e:
                logger.warning(f"Unsloth LoRA failed: {e}. Falling back to standard PEFT.")
                self._setup_lora_standard(
                    r, lora_alpha, target_modules, lora_dropout, bias
                )
        else:
            self._setup_lora_standard(
                r, lora_alpha, target_modules, lora_dropout, bias
            )
        
        # Print LoRA adapter size
        adapter_size = self._count_params(self.model)
        logger.info(f"LoRA adapters: {adapter_size:,} trainable params (~{adapter_size / 1e6:.2f}MB)")
        
        return self.model
    
    def _setup_lora_standard(
        self,
        r: int,
        lora_alpha: int,
        target_modules: list,
        lora_dropout: float,
        bias: str
    ):
        """Apply LoRA via standard PEFT library."""
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
    
    def prepare_for_inference(self):
        """Prepare model for fast inference (optional optimization)."""
        if UNSLOTH_AVAILABLE:
            try:
                self.model = FastLanguageModel.for_inference(self.model)
                logger.info("Model optimized for inference with Unsloth")
            except Exception as e:
                logger.warning(f"Inference optimization failed: {e}")
        else:
            logger.debug("Unsloth not available, skipping inference optimization")
        
        return self.model
    
    def save_adapter(self, output_dir: str):
        """Save LoRA adapter (not full model)."""
        logger.info(f"Saving LoRA adapter to {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
    
    def save_merged(self, output_dir: str, save_method: str = "merged_16bit"):
        """
        Save merged model (base + adapter).
        
        Args:
            output_dir: Output directory
            save_method: 'merged_16bit' or 'merged_4bit'
        """
        logger.info(f"Saving merged model to {output_dir} ({save_method})")
        self.model.save_pretrained_merged(output_dir, self.tokenizer, save_method=save_method)
    
    def get_model_and_tokenizer(self):
        """Return current model and tokenizer."""
        return self.model, self.tokenizer


def setup_smollm2(
    model_name: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    r: int = 16,
    lora_alpha: int = 16,
    use_unsloth: bool = True
) -> Tuple:
    """
    Convenience function to setup SmolLM2 with LoRA.
    
    Args:
        model_name: Model identifier
        max_seq_length: Context window
        load_in_4bit: Quantization
        r: LoRA rank
        lora_alpha: LoRA alpha
        use_unsloth: Use Unsloth optimizations
    
    Returns:
        (model, tokenizer)
    """
    manager = SmolLM2Manager(model_name)
    
    if use_unsloth:
        manager.load_model_unsloth(max_seq_length, load_in_4bit=load_in_4bit)
    else:
        manager.load_model_standard(max_seq_length, load_in_4bit=load_in_4bit)
    
    manager.setup_lora(r=r, lora_alpha=lora_alpha)
    
    return manager.get_model_and_tokenizer()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing SmolLM2 setup...")
    model, tokenizer = setup_smollm2(max_seq_length=2048)
    print(f"Model: {type(model)}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
