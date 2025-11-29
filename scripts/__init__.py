"""
SmolLM2 Finance Fine-Tuning Scripts
"""

from .data_loader import (
    FinanceDataProcessor,
    load_financial_phrasebank,
    load_custom_dataset
)

from .model_setup import (
    SmolLM2Manager,
    setup_smollm2
)

from .training_pipeline import (
    FinanceSFTTrainer,
    train_smollm2
)

from .inference import (
    SmolLM2Inference,
    FinanceEvaluator,
    evaluate_finance_model
)

from .utils import (
    DataUtils,
    MetricsUtils,
    FileUtils,
    TextUtils,
    ConfigUtils,
    setup_logging
)

__version__ = "0.1.0"
__all__ = [
    "FinanceDataProcessor",
    "load_financial_phrasebank",
    "load_custom_dataset",
    "SmolLM2Manager",
    "setup_smollm2",
    "FinanceSFTTrainer",
    "train_smollm2",
    "SmolLM2Inference",
    "FinanceEvaluator",
    "evaluate_finance_model",
    "DataUtils",
    "MetricsUtils",
    "FileUtils",
    "TextUtils",
    "ConfigUtils",
    "setup_logging",
]
