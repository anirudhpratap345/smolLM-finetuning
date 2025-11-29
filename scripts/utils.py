"""
Utility functions for the SmolLM2 finance fine-tuning project.
Includes helpers for data processing, metrics, and common tasks.
"""

import logging
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


class DataUtils:
    """Utilities for data handling."""
    
    @staticmethod
    def load_json(filepath: Union[str, Path]) -> Union[List, Dict]:
        """Load JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def save_json(data: Union[List, Dict], filepath: Union[str, Path], indent: int = 2):
        """Save to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent)
        logger.info(f"Saved to {filepath}")
    
    @staticmethod
    def load_csv(filepath: Union[str, Path]) -> List[Dict]:
        """Load CSV file into list of dicts."""
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)
    
    @staticmethod
    def save_csv(data: List[Dict], filepath: Union[str, Path]):
        """Save list of dicts to CSV."""
        if not data:
            logger.warning("No data to save")
            return
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        keys = data[0].keys()
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)
        
        logger.info(f"Saved {len(data)} rows to {filepath}")
    
    @staticmethod
    def create_instruction_pairs(
        texts: List[str],
        labels: List[str],
        instruction_template: str = "Classify sentiment: {text}"
    ) -> List[Dict]:
        """
        Create instruction-output pairs for SFT.
        
        Args:
            texts: Input texts
            labels: Output labels
            instruction_template: Template for instructions
        
        Returns:
            List of {"instruction": ..., "output": ...} dicts
        """
        pairs = []
        for text, label in zip(texts, labels):
            pairs.append({
                "instruction": instruction_template.format(text=text),
                "output": str(label)
            })
        return pairs


class MetricsUtils:
    """Utilities for metrics calculation."""
    
    @staticmethod
    def accuracy(predictions: List[str], labels: List[str]) -> float:
        """Calculate accuracy."""
        if not predictions:
            return 0.0
        correct = sum(1 for p, l in zip(predictions, labels) if p == l)
        return correct / len(predictions)
    
    @staticmethod
    def f1_score(predictions: List[str], labels: List[str]) -> float:
        """Calculate macro F1 score."""
        from sklearn.metrics import f1_score as sklearn_f1
        return sklearn_f1(labels, predictions, average='macro', zero_division=0)
    
    @staticmethod
    def confusion_matrix(predictions: List[str], labels: List[str]) -> Dict:
        """Calculate confusion matrix."""
        from sklearn.metrics import confusion_matrix as sklearn_cm
        
        classes = sorted(set(labels + predictions))
        cm = sklearn_cm(labels, predictions, labels=classes)
        
        return {
            "classes": classes,
            "matrix": cm.tolist()
        }
    
    @staticmethod
    def precision_recall(predictions: List[str], labels: List[str]) -> Dict:
        """Calculate precision and recall per class."""
        from sklearn.metrics import precision_recall_fscore_support
        
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        classes = sorted(set(labels))
        
        return {
            "classes": classes,
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "support": support.tolist()
        }


class FileUtils:
    """Utilities for file operations."""
    
    @staticmethod
    def ensure_dir(dirpath: Union[str, Path]):
        """Create directory if it doesn't exist."""
        Path(dirpath).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def list_files(dirpath: Union[str, Path], pattern: str = "*.txt") -> List[Path]:
        """List files in directory matching pattern."""
        return list(Path(dirpath).glob(pattern))
    
    @staticmethod
    def get_file_size(filepath: Union[str, Path]) -> int:
        """Get file size in bytes."""
        return Path(filepath).stat().st_size
    
    @staticmethod
    def format_size(size_bytes: int) -> str:
        """Format bytes to human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}TB"


class TextUtils:
    """Utilities for text processing."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Basic text cleaning."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove common artifacts
        text = text.replace('\n', ' ').replace('\r', '')
        return text
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 512) -> str:
        """Truncate text to max length."""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."
    
    @staticmethod
    def balance_classes(
        texts: List[str],
        labels: List[str],
        target_samples_per_class: Optional[int] = None
    ) -> tuple:
        """
        Balance dataset by undersampling majority classes.
        
        Args:
            texts: Input texts
            labels: Labels
            target_samples_per_class: Target samples per class (None = min class size)
        
        Returns:
            (balanced_texts, balanced_labels)
        """
        from sklearn.utils import class_weight
        
        # Group by class
        class_groups = {}
        for text, label in zip(texts, labels):
            if label not in class_groups:
                class_groups[label] = []
            class_groups[label].append(text)
        
        # Determine target
        if target_samples_per_class is None:
            target_samples_per_class = min(len(v) for v in class_groups.values())
        
        # Balance
        balanced_texts = []
        balanced_labels = []
        
        for label, group_texts in class_groups.items():
            selected = np.random.choice(
                group_texts,
                size=min(target_samples_per_class, len(group_texts)),
                replace=False
            )
            balanced_texts.extend(selected)
            balanced_labels.extend([label] * len(selected))
        
        logger.info(f"Balanced dataset: {len(balanced_texts)} samples, {len(class_groups)} classes")
        
        return balanced_texts, balanced_labels


class ConfigUtils:
    """Utilities for configuration management."""
    
    @staticmethod
    def save_config(config: Any, filepath: Union[str, Path]):
        """Save config as JSON."""
        from dataclasses import asdict
        
        if hasattr(config, '__dataclass_fields__'):
            config_dict = asdict(config)
        else:
            config_dict = vars(config)
        
        DataUtils.save_json(config_dict, filepath)
    
    @staticmethod
    def load_config_dict(filepath: Union[str, Path]) -> Dict:
        """Load config from JSON."""
        return DataUtils.load_json(filepath)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional file to write logs to
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level))
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, log_level))
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler (optional)
    if log_file:
        FileUtils.ensure_dir(Path(log_file).parent)
        fh = logging.FileHandler(log_file)
        fh.setLevel(getattr(logging, log_level))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger


if __name__ == "__main__":
    setup_logging()
    
    # Test utilities
    print("Testing DataUtils...")
    test_data = {"key": "value", "nested": [1, 2, 3]}
    DataUtils.save_json(test_data, "test.json")
    loaded = DataUtils.load_json("test.json")
    print(f"Saved and loaded: {loaded}")
    
    print("\nTesting TextUtils...")
    text = "This  is   a   test."
    cleaned = TextUtils.clean_text(text)
    print(f"Cleaned: {cleaned}")
    
    print("\nUtilities working!")
