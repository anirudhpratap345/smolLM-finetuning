"""
GPU Installation Helper Script
Detects hardware and provides installation instructions for CUDA/PyTorch
"""

import sys
import platform
import subprocess


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)


def check_system_info():
    """Check system information."""
    print_header("SYSTEM INFORMATION")
    
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"Architecture: {platform.machine()}")
    
    # Check GPU
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"\n✓ GPU DETECTED:")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA: {torch.version.cuda}")
            print(f"  cuDNN: {torch.version.cudnn}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            return True
        else:
            print(f"\n✗ No GPU detected (CPU-only PyTorch)")
            return False
    except ImportError:
        print(f"\n✗ PyTorch not installed")
        return False


def install_cuda_windows():
    """Install CUDA on Windows."""
    print_header("INSTALLING CUDA FOR WINDOWS")
    
    instructions = """
1. Download CUDA Toolkit 11.8:
   https://developer.nvidia.com/cuda-11-8-0-download-archive
   
2. Select:
   - OS: Windows
   - Architecture: x86_64
   - Version: 11 (Windows 11)
   - Installer Type: exe (local)

3. Run the installer and follow the setup wizard

4. Add CUDA to PATH (if not done automatically):
   Windows Settings → Environment Variables
   Add to PATH:
   - C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin
   - C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\libnvvp

5. Verify installation:
   nvcc --version

6. Then reinstall PyTorch with CUDA:
   pip uninstall torch torchvision torchaudio -y
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
"""
    print(instructions)


def install_cuda_linux():
    """Install CUDA on Linux."""
    print_header("INSTALLING CUDA FOR LINUX")
    
    instructions = """
1. Download CUDA Toolkit 11.8:
   https://developer.nvidia.com/cuda-11-8-0-download-archive
   
2. Select:
   - OS: Linux
   - Architecture: x86_64
   - Distribution: (your distro, e.g., Ubuntu 22.04)
   - Installer Type: runfile (local)

3. Run installer:
   sudo sh cuda_11.8.0_linux.run

4. Add to ~/.bashrc:
   export PATH=/usr/local/cuda-11.8/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

5. Apply changes:
   source ~/.bashrc

6. Verify:
   nvcc --version

7. Reinstall PyTorch with CUDA:
   pip uninstall torch torchvision torchaudio -y
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
"""
    print(instructions)


def install_pytorch_cpu():
    """Install PyTorch for CPU."""
    print_header("INSTALLING PYTORCH FOR CPU")
    
    print("PyTorch (CPU-only) installation:")
    print("\npip install torch torchvision torchaudio")
    print("\nNote: Training will be VERY slow on CPU. Recommend GPU or Colab.")


def install_pytorch_gpu():
    """Install PyTorch for GPU."""
    print_header("INSTALLING PYTORCH FOR GPU (CUDA 11.8)")
    
    print("PyTorch (GPU) installation:")
    print("\npip uninstall torch torchvision torchaudio -y")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("\nOr with conda:")
    print("conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia")


def recommend_options():
    """Print recommended options based on system."""
    print_header("RECOMMENDED OPTIONS FOR YOU")
    
    system = platform.system()
    
    print("\n1. EASIEST: Google Colab (Free GPU)")
    print("   - No installation needed")
    print("   - Free T4 GPU (45-90 min training)")
    print("   - Visit: https://colab.research.google.com")
    print("   - Upload project and run cells")
    
    print("\n2. INSTALL LOCALLY: Your GPU Hardware")
    if system == "Windows":
        print("   - Download CUDA 11.8 for Windows")
        print("   - URL: https://developer.nvidia.com/cuda-11-8-0-download-archive")
        print("   - Then reinstall PyTorch with CUDA support")
    elif system == "Darwin":
        print("   - macOS GPU support is limited")
        print("   - Recommend Colab or Linux machine")
    else:
        print("   - Download CUDA 11.8 for Linux")
        print("   - URL: https://developer.nvidia.com/cuda-11-8-0-download-archive")
        print("   - Then reinstall PyTorch with CUDA support")
    
    print("\n3. CPU-ONLY: Current Setup (Slow)")
    print("   - Use python train_demo.py for testing")
    print("   - Full training will take HOURS")


def main():
    """Main entry point."""
    print("\n" + "="*80)
    print("  SmolLM2 Finance - GPU Setup Helper")
    print("="*80)
    
    # Check current setup
    has_gpu = check_system_info()
    
    if has_gpu:
        print("\n✓ Your system is ready for GPU training!")
        print("  Run: python train.py")
    else:
        print("\n✗ No GPU detected. Choose an option:")
        recommend_options()
        
        system = platform.system()
        
        print("\n" + "-"*80)
        choice = input("\nWhat would you like to do?\n1. Get CUDA installation instructions\n2. Get PyTorch installation\n3. View Colab link\n4. Exit\n\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            if system == "Windows":
                install_cuda_windows()
            elif system == "Darwin":
                print("macOS GPU support is limited. Recommend Colab or Linux.")
            else:
                install_cuda_linux()
        elif choice == "2":
            install_pytorch_gpu()
        elif choice == "3":
            print("\nOpen: https://colab.research.google.com")
            print("Then see QUICKSTART.md for Colab instructions")
        elif choice == "4":
            print("Exiting.")
        else:
            print("Invalid choice.")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExited.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
