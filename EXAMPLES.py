"""
Quick start guide - Copy-paste examples for common tasks.
"""

# ============================================================================
# 1. BASIC TRAINING (From scratch)
# ============================================================================

from config.training_config import get_local_gpu_config
from scripts.data_loader import load_financial_phrasebank
from scripts.model_setup import setup_smollm2
from scripts.training_pipeline import train_smollm2

# Load config (use get_colab_config() for Colab)
config = get_local_gpu_config()

# Load dataset
train_data, eval_data = load_financial_phrasebank(max_samples=1000)

# Setup model with LoRA
model, tokenizer = setup_smollm2(
    max_seq_length=2048,
    r=16,
    lora_alpha=16
)

# Train
result, output_dir = train_smollm2(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_data,
    eval_dataset=eval_data,
    output_dir="my-smollm2",
    max_steps=100
)

# Save
model.save_pretrained("my-smollm2")
tokenizer.save_pretrained("my-smollm2")


# ============================================================================
# 2. LOAD CUSTOM DATASET
# ============================================================================

from scripts.data_loader import load_custom_dataset, FinanceDataProcessor

# Option A: Load from HF Hub
train, eval = load_custom_dataset(
    data_path="sentiment-financial",  # HF dataset ID
    formatter=FinanceDataProcessor().format_custom_qa
)

# Option B: Load from local file
processor = FinanceDataProcessor("local_dataset")
dataset = processor.load_dataset()
formatted = processor.format_dataset(processor.format_custom_qa)
train, eval = processor.prepare_splits(formatted, max_samples=5000)

# Option C: Create custom pairs
from scripts.utils import DataUtils

texts = ["Apple stock rose 15%", "Tesla fell 8%"]
labels = ["Positive", "Negative"]
pairs = DataUtils.create_instruction_pairs(texts, labels)


# ============================================================================
# 3. SENTIMENT CLASSIFICATION
# ============================================================================

from scripts.inference import SmolLM2Inference

inference = SmolLM2Inference(model, tokenizer)

# Single prompt
sentiment = inference.chat_completion(
    "Classify sentiment: NVIDIA earnings beat expectations"
)
print(f"Sentiment: {sentiment}")

# Batch
texts = [
    "Apple stock surged 15%",
    "Tesla plummeted 8%",
    "Market is neutral"
]
prompts = [f"Classify sentiment: {t}" for t in texts]
results = inference.batch_generate(prompts, temperature=0.1, max_new_tokens=20)


# ============================================================================
# 4. Q&A GENERATION
# ============================================================================

questions = [
    "What are the risks in tech stocks?",
    "Analyze NVIDIA Q4 revenue drivers",
    "Summarize market trends"
]

for q in questions:
    answer = inference.chat_completion(q, max_new_tokens=150)
    print(f"Q: {q}\nA: {answer}\n")


# ============================================================================
# 5. EVALUATION METRICS
# ============================================================================

from scripts.inference import FinanceEvaluator
from scripts.utils import MetricsUtils

evaluator = FinanceEvaluator(inference)

# Sentiment accuracy
results = evaluator.sentiment_classification(
    test_texts=["NVIDIA up 12%", "Tesla down 5%"],
    labels=["Positive", "Negative"]
)
print(f"Accuracy: {results.get('accuracy', 'N/A')}")

# Custom metrics
predictions = ["Positive", "Negative", "Positive"]
labels = ["Positive", "Positive", "Positive"]

accuracy = MetricsUtils.accuracy(predictions, labels)
f1 = MetricsUtils.f1_score(predictions, labels)
cm = MetricsUtils.confusion_matrix(predictions, labels)

print(f"Accuracy: {accuracy:.2%}")
print(f"F1 Score: {f1:.3f}")
print(f"Confusion Matrix: {cm}")


# ============================================================================
# 6. LATENCY BENCHMARK
# ============================================================================

test_prompts = [
    "Analyze NVIDIA earnings",
    "What is the stock market trend?",
    "Summarize earnings report"
]

results = evaluator.latency_benchmark(test_prompts, num_runs=3)

print(f"Latency: {results['avg_latency_s']:.3f}s")
print(f"Throughput: {results['throughput_tokens_per_sec']:.1f} tokens/sec")


# ============================================================================
# 7. LOAD & INFER FROM SAVED ADAPTER
# ============================================================================

from scripts.model_setup import SmolLM2Manager
from peft import PeftModel

# Load base model
manager = SmolLM2Manager()
base_model, tokenizer = manager.load_model_unsloth()

# Load adapter
model = PeftModel.from_pretrained(base_model, "my-smollm2")

# Infer
inference = SmolLM2Inference(model, tokenizer)
response = inference.chat_completion("Classify sentiment: NVIDIA up 12%")


# ============================================================================
# 8. DATA BALANCING & PREPROCESSING
# ============================================================================

from scripts.utils import TextUtils, DataUtils

# Balance classes
texts = ["pos text", "neg text", "pos text", "pos text"]
labels = ["positive", "negative", "positive", "positive"]

balanced_texts, balanced_labels = TextUtils.balance_classes(
    texts, labels,
    target_samples_per_class=2
)

# Clean text
dirty_text = "NVIDIA  stock   rose   15%"
clean = TextUtils.clean_text(dirty_text)

# Save results
results = {"predictions": ["Positive", "Negative"], "accuracy": 0.85}
DataUtils.save_json(results, "results.json")


# ============================================================================
# 9. WANDB INTEGRATION (Optional)
# ============================================================================

# In training_config.py, change:
# report_to: str = "wandb"

# Then login before training:
# wandb login

# View dashboard at wandb.ai


# ============================================================================
# 10. COLAB QUICK START
# ============================================================================

# Cell 1: Install
# !pip install -q torch transformers peft accelerate bitsandbytes trl datasets
# !pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Cell 2: Clone and setup
# !git clone <your-repo>
# %cd <your-repo>
# from config.training_config import get_colab_config

# Cell 3: Train
# config = get_colab_config()
# train_data, eval_data = load_financial_phrasebank(max_samples=config.data.max_samples)
# model, tokenizer = setup_smollm2()
# result, output_dir = train_smollm2(model, tokenizer, train_data, eval_data)

# Cell 4: Push to HF Hub
# from huggingface_hub import notebook_login
# notebook_login()
# model.push_to_hub("<username>/smollm2-finance")
