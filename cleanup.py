"""
Cleanup script - Uninstall all SmolLM2 project dependencies
"""

import subprocess
import sys


def uninstall_packages():
    """Uninstall all project dependencies."""
    
    packages_to_uninstall = [
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "peft",
        "datasets",
        "accelerate",
        "bitsandbytes",
        "trl",
        "scikit-learn",
        "evaluate",
        "wandb",
        "unsloth",
    ]
    
    print("="*80)
    print("UNINSTALLING ALL PROJECT DEPENDENCIES")
    print("="*80)
    print(f"\nPackages to uninstall: {len(packages_to_uninstall)}")
    
    for pkg in packages_to_uninstall:
        print(f"\nUninstalling {pkg}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", pkg],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"  ✓ {pkg} uninstalled")
        else:
            print(f"  ✗ {pkg} not found or error")
    
    print("\n" + "="*80)
    print("CLEANUP COMPLETE")
    print("="*80)
    print("\nAll project dependencies have been uninstalled.")
    print("Your project files remain in place.")
    print("\nTo reinstall, run:")
    print("  pip install -r requirements.txt")


if __name__ == "__main__":
    try:
        uninstall_packages()
    except KeyboardInterrupt:
        print("\n\nCleanup cancelled by user.")
    except Exception as e:
        print(f"Error during cleanup: {e}")
        sys.exit(1)
