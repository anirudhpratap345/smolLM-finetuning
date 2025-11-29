"""
Hardware detection and configuration helpers.
"""

import logging
import torch
import platform

logger = logging.getLogger(__name__)


def detect_hardware():
    """Detect available hardware and return configuration info."""
    info = {
        "os": platform.system(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if info["cuda_available"]:
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        info["device"] = "cuda"
    else:
        info["gpu_name"] = None
        info["cuda_version"] = None
        info["gpu_memory_gb"] = None
        info["device"] = "cpu"
    
    return info


def get_recommended_config():
    """Return the recommended config based on detected hardware."""
    from config.training_config import get_local_gpu_config, get_colab_config, get_cpu_config
    
    info = detect_hardware()
    
    if not info["cuda_available"]:
        logger.warning("No GPU detected. Using CPU configuration (very slow!).")
        logger.warning("For best results, use a GPU:")
        logger.warning("  - Colab: https://colab.research.google.com")
        logger.warning("  - Local: Install CUDA from https://developer.nvidia.com/cuda-downloads")
        return get_cpu_config()
    
    # GPU detected
    vram_gb = info["gpu_memory_gb"]
    
    if vram_gb >= 24:
        logger.info(f"High-end GPU detected ({vram_gb:.1f}GB). Using optimized config.")
        return get_local_gpu_config()
    elif vram_gb >= 16:
        logger.info(f"Mid-range GPU detected ({vram_gb:.1f}GB). Using standard config.")
        return get_local_gpu_config()
    elif vram_gb >= 8:
        logger.info(f"Entry-level GPU detected ({vram_gb:.1f}GB). Using conservative config.")
        config = get_local_gpu_config()
        config.training.per_device_train_batch_size = 1
        config.training.gradient_accumulation_steps = 4
        return config
    else:
        logger.warning(f"Limited VRAM ({vram_gb:.1f}GB). Using minimal config for Colab T4.")
        return get_colab_config()


def print_hardware_info():
    """Print detected hardware information."""
    info = detect_hardware()
    
    print("\n" + "="*80)
    print("HARDWARE INFORMATION")
    print("="*80)
    print(f"OS: {info['os']}")
    print(f"Python: {info['python_version']}")
    print(f"PyTorch: {info['torch_version']}")
    
    if info["cuda_available"]:
        print(f"GPU: {info['gpu_name']}")
        print(f"CUDA: {info['cuda_version']}")
        print(f"VRAM: {info['gpu_memory_gb']:.1f}GB")
        print(f"Device: {info['device'].upper()} [OK]")
    else:
        print("GPU: Not available")
        print("Device: CPU (WARNING: Very slow)")
        print("\nTo enable GPU:")
        print("  1. Install CUDA: https://developer.nvidia.com/cuda-downloads")
        print("  2. Install PyTorch with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("  3. Or use Colab: https://colab.research.google.com")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print_hardware_info()
    
    config = get_recommended_config()
    print(f"Recommended config loaded:")
    print(f"  Batch size: {config.training.per_device_train_batch_size}")
    print(f"  Max samples: {config.data.max_samples}")
    print(f"  Max steps: {config.training.max_steps}")
