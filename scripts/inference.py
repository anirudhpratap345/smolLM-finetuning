"""
Inference and evaluation utilities for fine-tuned SmolLM2.
Includes generation, sentiment classification, and benchmark evaluation.
"""

import logging
import torch
from typing import List, Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm

# Optional Unsloth optimization
UNSLOTH_AVAILABLE = False
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except (ImportError, RuntimeError):
    FastLanguageModel = None

logger = logging.getLogger(__name__)


class SmolLM2Inference:
    """Inference utilities for fine-tuned SmolLM2."""
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 100,
        repetition_penalty: float = 1.1
    ):
        """
        Initialize inference engine.
        
        Args:
            model: Fine-tuned model
            tokenizer: Tokenizer
            device: Device to run on
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_new_tokens: Max generation length
            repetition_penalty: Repetition penalty
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        
        # Prepare for inference (Unsloth optimization if available)
        if UNSLOTH_AVAILABLE:
            try:
                self.model = FastLanguageModel.for_inference(model)
                logger.info("Model optimized for inference with Unsloth")
            except Exception as e:
                logger.debug(f"Unsloth inference optimization failed: {e}")
        else:
            logger.debug("Unsloth not available, using standard inference")
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_new_tokens: Override max tokens
            top_p: Override top_p
        
        Returns:
            Generated text
        """
        temperature = temperature or self.temperature
        max_new_tokens = max_new_tokens or self.max_new_tokens
        top_p = top_p or self.top_p
        
        # Tokenize
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=self.repetition_penalty,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new generated part
        generated_only = generated_text[len(prompt):].strip()
        
        return generated_only
    
    def chat_completion(
        self,
        user_message: str,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None
    ) -> str:
        """
        Chat completion (instruction following).
        
        Args:
            user_message: User query
            temperature: Override temperature
            max_new_tokens: Override max tokens
        
        Returns:
            Assistant response
        """
        prompt = f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        
        response = self.generate(prompt, temperature=temperature, max_new_tokens=max_new_tokens)
        
        # Clean up response (remove assistant marker if present)
        response = response.replace("<|im_end|>", "").strip()
        
        return response
    
    def batch_generate(
        self,
        prompts: List[str],
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None
    ) -> List[str]:
        """
        Generate for multiple prompts (batched).
        
        Args:
            prompts: List of prompts
            temperature: Override temperature
            max_new_tokens: Override max tokens
        
        Returns:
            List of generated texts
        """
        results = []
        for prompt in tqdm(prompts, desc="Generating"):
            result = self.generate(prompt, temperature=temperature, max_new_tokens=max_new_tokens)
            results.append(result)
        
        return results


class FinanceEvaluator:
    """Evaluation utilities for finance tasks."""
    
    def __init__(self, inference_engine: SmolLM2Inference):
        """Initialize evaluator."""
        self.inference = inference_engine
    
    def sentiment_classification(
        self,
        texts: List[str],
        labels: Optional[List[str]] = None
    ) -> Dict:
        """
        Evaluate sentiment classification.
        
        Args:
            texts: Input texts
            labels: Ground truth labels (optional for accuracy)
        
        Returns:
            Results dict with predictions and metrics
        """
        predictions = []
        
        for text in tqdm(texts, desc="Sentiment classification"):
            prompt = f"<|im_start|>user\nClassify sentiment: {text}<|im_end|>\n<|im_start|>assistant\n"
            response = self.inference.generate(prompt, temperature=0.1, max_new_tokens=20)
            predictions.append(response.strip().lower())
        
        results = {"predictions": predictions}
        
        # Calculate accuracy if labels provided
        if labels:
            labels_lower = [l.lower() for l in labels]
            accuracy = sum(1 for p, l in zip(predictions, labels_lower) if p.startswith(l)) / len(labels)
            results["accuracy"] = accuracy
            logger.info(f"Sentiment classification accuracy: {accuracy:.2%}")
        
        return results
    
    def qa_generation(
        self,
        questions: List[str]
    ) -> List[str]:
        """
        Generate Q&A responses.
        
        Args:
            questions: List of questions
        
        Returns:
            List of answers
        """
        answers = []
        
        for question in tqdm(questions, desc="Q&A generation"):
            answer = self.inference.chat_completion(question, max_new_tokens=150)
            answers.append(answer)
        
        return answers
    
    def hallucination_check(
        self,
        prompts: List[str],
        expected_contexts: List[str]
    ) -> Dict:
        """
        Check model for hallucinations by comparing output to expected context.
        
        Args:
            prompts: Input prompts
            expected_contexts: Expected knowledge contexts
        
        Returns:
            Hallucination metrics
        """
        from difflib import SequenceMatcher
        
        hallucination_scores = []
        
        for prompt, context in tqdm(
            zip(prompts, expected_contexts),
            desc="Hallucination check",
            total=len(prompts)
        ):
            response = self.inference.chat_completion(prompt, max_new_tokens=100)
            
            # Simple overlap check (not perfect, but a proxy)
            ratio = SequenceMatcher(None, response.lower(), context.lower()).ratio()
            hallucination_score = 1.0 - ratio
            hallucination_scores.append(hallucination_score)
        
        return {
            "avg_hallucination_score": np.mean(hallucination_scores),
            "hallucination_scores": hallucination_scores
        }
    
    def latency_benchmark(
        self,
        prompts: List[str],
        num_runs: int = 3
    ) -> Dict:
        """
        Benchmark inference latency and throughput.
        
        Args:
            prompts: Prompts to benchmark
            num_runs: Runs per prompt
        
        Returns:
            Latency metrics
        """
        import time
        
        latencies = []
        token_counts = []
        
        for prompt in tqdm(prompts[:5], desc="Latency benchmark"):
            times = []
            
            for _ in range(num_runs):
                start = time.time()
                response = self.inference.generate(prompt, max_new_tokens=50)
                elapsed = time.time() - start
                times.append(elapsed)
                
                # Count tokens
                tokens = len(self.inference.tokenizer.encode(response))
                token_counts.append(tokens)
            
            latencies.extend(times)
        
        avg_latency = np.mean(latencies)
        throughput = np.mean(token_counts) / avg_latency if avg_latency > 0 else 0
        
        return {
            "avg_latency_s": avg_latency,
            "throughput_tokens_per_sec": throughput,
            "latencies": latencies,
        }


def evaluate_finance_model(
    model,
    tokenizer,
    test_data: Dict,
    task: str = "sentiment"
) -> Dict:
    """
    Comprehensive evaluation of finance model.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        test_data: Test data dict with 'texts', 'labels' keys
        task: Task type ('sentiment', 'qa', 'hallucination', 'latency')
    
    Returns:
        Evaluation results
    """
    inference = SmolLM2Inference(model, tokenizer)
    evaluator = FinanceEvaluator(inference)
    
    if task == "sentiment":
        return evaluator.sentiment_classification(
            test_data.get("texts", []),
            test_data.get("labels")
        )
    elif task == "qa":
        return {"answers": evaluator.qa_generation(test_data.get("questions", []))}
    elif task == "hallucination":
        return evaluator.hallucination_check(
            test_data.get("prompts", []),
            test_data.get("contexts", [])
        )
    elif task == "latency":
        return evaluator.latency_benchmark(test_data.get("prompts", []))
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Inference module imported successfully. Use in evaluation scripts.")
