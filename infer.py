"""
Inference script for testing fine-tuned SmolLM2 model.
Load trained adapter and run inference on finance tasks.
"""

import logging
import sys
import torch
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from scripts.model_setup import SmolLM2Manager
from scripts.inference import SmolLM2Inference, FinanceEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_trained_model(
    adapter_dir: str = "smollm2-finance-tuned",
    base_model: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    load_merged: bool = False
):
    """
    Load trained model with LoRA adapter.
    
    Args:
        adapter_dir: Directory with saved adapter
        base_model: Base model name (ignored if merged model)
        load_merged: Load merged model instead of adapter
    
    Returns:
        (model, tokenizer)
    """
    logger.info(f"Loading model from {adapter_dir}...")
    
    if load_merged:
        # Load merged model
        try:
            from peft import AutoPeftModelForCausalLM
            model = AutoPeftModelForCausalLM.from_pretrained(
                adapter_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            logger.info("Loaded merged model")
        except Exception as e:
            logger.warning(f"Merged model load failed: {e}. Trying standard load...")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(adapter_dir)
    else:
        # Load base model + adapter
        manager = SmolLM2Manager(base_model)
        manager.load_model_unsloth(load_in_4bit=True)
        
        # Load adapter
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(manager.model, adapter_dir)
            logger.info("Loaded base model with LoRA adapter")
        except Exception as e:
            logger.error(f"Failed to load adapter: {e}")
            raise
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    
    logger.info("Model and tokenizer loaded successfully")
    return model, tokenizer


def interactive_inference(
    model,
    tokenizer,
    task: str = "sentiment"
):
    """
    Interactive inference loop.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        task: Task type ('sentiment', 'qa')
    """
    inference = SmolLM2Inference(model, tokenizer)
    
    logger.info(f"\nStarting {task} inference. Type 'quit' to exit.")
    print("-" * 80)
    
    while True:
        if task == "sentiment":
            user_input = input("\nEnter financial text for sentiment: ").strip()
            if user_input.lower() == "quit":
                break
            
            response = inference.chat_completion(
                f"Classify sentiment: {user_input}",
                temperature=0.1,
                max_new_tokens=20
            )
            print(f"Sentiment: {response}\n")
        
        elif task == "qa":
            user_input = input("\nAsk a finance question: ").strip()
            if user_input.lower() == "quit":
                break
            
            response = inference.chat_completion(
                user_input,
                temperature=0.7,
                max_new_tokens=200
            )
            print(f"Answer: {response}\n")


def batch_inference_example(model, tokenizer):
    """Example: Batch inference on multiple texts."""
    
    logger.info("\nRunning batch inference example...")
    
    inference = SmolLM2Inference(model, tokenizer)
    
    test_texts = [
        "Apple stock surged 15% after strong earnings report.",
        "Tesla plummeted 8% due to supply chain concerns.",
        "Market is stable with mixed economic signals."
    ]
    
    prompts = [f"<|im_start|>user\nClassify sentiment: {text}<|im_end|>\n<|im_start|>assistant\n" for text in test_texts]
    
    predictions = inference.batch_generate(prompts, temperature=0.1, max_new_tokens=20)
    
    print("\nBatch Results:")
    print("-" * 80)
    for text, pred in zip(test_texts, predictions):
        print(f"Text: {text}")
        print(f"Sentiment: {pred}")
        print()


def evaluate_example(model, tokenizer):
    """Example: Run evaluation metrics."""
    
    logger.info("\nRunning evaluation example...")
    
    inference = SmolLM2Inference(model, tokenizer)
    evaluator = FinanceEvaluator(inference)
    
    # Latency benchmark
    test_prompts = [
        "Analyze NVIDIA Q4 revenue drivers.",
        "What are the risks in the tech sector?",
        "Summarize the latest market trends."
    ]
    
    results = evaluator.latency_benchmark(test_prompts, num_runs=2)
    
    print("\nLatency Benchmark:")
    print("-" * 80)
    print(f"Average latency: {results['avg_latency_s']:.3f}s")
    print(f"Throughput: {results['throughput_tokens_per_sec']:.1f} tokens/sec")


def main():
    """Main inference script."""
    
    logger.info("="*80)
    logger.info("SmolLM2 Finance Model - Inference & Evaluation")
    logger.info("="*80)
    
    # Load trained model
    model, tokenizer = load_trained_model(
        adapter_dir="smollm2-finance-tuned",
        load_merged=False  # Set to True if model was saved merged
    )
    
    # Choose mode
    print("\n" + "="*80)
    print("Inference Modes:")
    print("  1. Interactive sentiment classification")
    print("  2. Interactive Q&A")
    print("  3. Batch inference example")
    print("  4. Evaluate (latency benchmark)")
    print("  5. Exit")
    print("="*80)
    
    while True:
        choice = input("\nSelect mode (1-5): ").strip()
        
        if choice == "1":
            interactive_inference(model, tokenizer, task="sentiment")
        elif choice == "2":
            interactive_inference(model, tokenizer, task="qa")
        elif choice == "3":
            batch_inference_example(model, tokenizer)
        elif choice == "4":
            evaluate_example(model, tokenizer)
        elif choice == "5":
            logger.info("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
