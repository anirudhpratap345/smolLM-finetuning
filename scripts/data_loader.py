"""
Data loading and preprocessing for finance datasets.
Supports Financial PhraseBank, PIXIU, synthetic data, and custom formats.
"""

import logging
from typing import Dict, List, Optional, Callable
import datasets
from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)


class FinanceDataProcessor:
    """Handle loading, formatting, and preprocessing finance datasets."""
    
    def __init__(self, dataset_name: str, config_name: Optional[str] = None):
        """
        Initialize data processor.
        
        Args:
            dataset_name: HF dataset identifier (e.g., 'financial_phrasebank')
            config_name: Dataset config/subset (e.g., 'sentences_allagree')
        """
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.dataset = None
    
    def load_dataset(self) -> DatasetDict:
        """Load dataset from Hugging Face Hub."""
        logger.info(f"Loading dataset: {self.dataset_name} (config: {self.config_name})")
        
        if self.config_name:
            self.dataset = load_dataset(self.dataset_name, self.config_name)
        else:
            self.dataset = load_dataset(self.dataset_name)
        
        logger.info(f"Dataset loaded: {self.dataset}")
        return self.dataset
    
    def format_financial_phrasebank(self, example: Dict) -> Dict:
        """
        Format Financial PhraseBank data for instruction tuning.
        
        Input: {'sentence': str, 'label': int (0=negative, 1=neutral, 2=positive)}
        Output: {'text': chat format}
        """
        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        label_text = label_map.get(example.get('label'), "Unknown")
        
        text = (
            f"<|im_start|>user\n"
            f"Classify the sentiment of this financial sentence:\n{example['sentence']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"{label_text}<|im_end|>\n"
        )
        
        return {"text": text}
    
    def format_pixiu_qa(self, example: Dict) -> Dict:
        """
        Format PIXIU/FinGPT Q&A data.
        
        Input: {'instruction': str, 'input': str, 'output': str}
        Output: {'text': chat format}
        """
        input_text = example.get('input', '')
        prompt = f"{example['instruction']}\n{input_text}".strip()
        
        text = (
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{example['output']}<|im_end|>\n"
        )
        
        return {"text": text}
    
    def format_custom_qa(self, example: Dict) -> Dict:
        """
        Format custom Q&A data (generic instruction/output format).
        
        Input: {'instruction': str, 'output': str}
        Output: {'text': chat format}
        """
        text = (
            f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
            f"<|im_start|>assistant\n{example['output']}<|im_end|>\n"
        )
        
        return {"text": text}
    
    def format_dataset(self, formatter: Callable) -> DatasetDict:
        """
        Apply formatting function to dataset.
        
        Args:
            formatter: Function that takes example dict and returns formatted dict with 'text' field
        
        Returns:
            Formatted dataset
        """
        if self.dataset is None:
            self.load_dataset()
        
        # Handle both DatasetDict and dict
        if isinstance(self.dataset, dict):
            # Convert dict of datasets to proper format
            formatted = {}
            for split_name, split_data in self.dataset.items():
                if hasattr(split_data, 'map'):
                    formatted[split_name] = split_data.map(
                        formatter,
                        num_proc=2,
                        desc=f"Formatting {split_name}"
                    )
                else:
                    formatted[split_name] = split_data
            formatted = formatted
        else:
            # Standard DatasetDict
            formatted = self.dataset.map(
                formatter,
                num_proc=2,
                desc="Formatting dataset"
            )
        
        logger.info(f"Dataset formatted: {formatted}")
        return formatted
    
    def prepare_splits(
        self,
        dataset: DatasetDict,
        split_name: str = "train",
        test_size: float = 0.1,
        seed: int = 42,
        max_samples: Optional[int] = None
    ) -> tuple:
        """
        Prepare train/test splits.
        
        Args:
            dataset: Formatted dataset
            split_name: Original split to use (e.g., 'train')
            test_size: Fraction for test set
            seed: Random seed for reproducibility
            max_samples: Max samples to use (for fast iteration)
        
        Returns:
            (train_dataset, eval_dataset)
        """
        # Get base split
        if isinstance(dataset, dict):
            if split_name in dataset:
                data = dataset[split_name]
            else:
                # Get first available split
                data = next(iter(dataset.values()))
        elif split_name in dataset:
            data = dataset[split_name]
        else:
            data = dataset['train']
        
        # Limit samples if specified
        if max_samples and len(data) > max_samples:
            data = data.shuffle(seed=seed).select(range(max_samples))
            logger.info(f"Limited to {max_samples} samples")
        else:
            data = data.shuffle(seed=seed)
        
        # Split into train/eval
        split = data.train_test_split(test_size=test_size, seed=seed)
        
        logger.info(f"Train: {len(split['train'])} | Eval: {len(split['test'])}")
        
        return split['train'], split['test']


def load_financial_phrasebank(
    max_samples: Optional[int] = 1000,
    test_size: float = 0.1,
    seed: int = 42
) -> tuple:
    """
    Load and format Financial PhraseBank for sentiment classification.
    
    Args:
        max_samples: Subset size for iteration speed
        test_size: Eval split fraction
        seed: Random seed
    
    Returns:
        (train_dataset, eval_dataset)
    """
    try:
        # Try to load from HF (may fail due to dataset script restrictions)
        processor = FinanceDataProcessor("financial_phrasebank", "sentences_allagree")
        dataset = processor.load_dataset()
        formatted = processor.format_dataset(processor.format_financial_phrasebank)
    except Exception as e:
        logger.warning(f"Failed to load financial_phrasebank: {e}")
        logger.info("Creating synthetic financial sentiment dataset instead...")
        
        # Create synthetic dataset if loading fails
        from datasets import Dataset as HFDataset
        
        synthetic_data = [
            {"sentence": "Apple stock surged 15% after strong earnings report.", "label": 2},
            {"sentence": "Tesla plummeted 8% due to supply chain concerns.", "label": 0},
            {"sentence": "Market is stable with mixed economic signals.", "label": 1},
            {"sentence": "NVIDIA exceeded revenue expectations by 20%.", "label": 2},
            {"sentence": "Banking sector faces regulatory headwinds.", "label": 0},
            {"sentence": "Tech stocks show moderate growth this quarter.", "label": 1},
            {"sentence": "Energy prices rally on geopolitical tensions.", "label": 2},
            {"sentence": "Consumer spending slows amid inflation concerns.", "label": 0},
            {"sentence": "Cloud computing demand remains robust.", "label": 2},
            {"sentence": "Retail sector shows mixed performance.", "label": 1},
        ]
        
        # Repeat to get more samples
        repeat_count = max(1, (max_samples or 100) // len(synthetic_data))
        synthetic_data = synthetic_data * repeat_count
        
        dataset = {"train": HFDataset.from_dict({
            "sentence": [d["sentence"] for d in synthetic_data],
            "label": [d["label"] for d in synthetic_data]
        })}
        
        processor = FinanceDataProcessor("synthetic")
        processor.dataset = dataset
        formatted = processor.format_dataset(processor.format_financial_phrasebank)
    
    train_data, eval_data = processor.prepare_splits(
        formatted,
        split_name="train",
        test_size=test_size,
        seed=seed,
        max_samples=max_samples
    )
    
    return train_data, eval_data


def load_custom_dataset(
    data_path: str,
    formatter: Callable,
    max_samples: Optional[int] = None,
    test_size: float = 0.1,
    seed: int = 42
) -> tuple:
    """
    Load custom dataset from local file or HF Hub.
    
    Args:
        data_path: Path to .csv, .json, or HF dataset ID
        formatter: Formatting function
        max_samples: Subset size
        test_size: Eval split fraction
        seed: Random seed
    
    Returns:
        (train_dataset, eval_dataset)
    """
    # Load from path
    if data_path.endswith('.csv'):
        dataset = load_dataset('csv', data_files=data_path)
    elif data_path.endswith('.json'):
        dataset = load_dataset('json', data_files=data_path)
    else:
        # Assume HF Hub dataset
        dataset = load_dataset(data_path)
    
    processor = FinanceDataProcessor(data_path)
    processor.dataset = dataset
    formatted = processor.format_dataset(formatter)
    
    train_data, eval_data = processor.prepare_splits(
        formatted,
        test_size=test_size,
        seed=seed,
        max_samples=max_samples
    )
    
    return train_data, eval_data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test: Load Financial PhraseBank
    print("Testing Financial PhraseBank loader...")
    train, eval_set = load_financial_phrasebank(max_samples=100)
    print(f"Train samples: {len(train)}")
    print(f"Sample: {train[0]['text'][:200]}")
