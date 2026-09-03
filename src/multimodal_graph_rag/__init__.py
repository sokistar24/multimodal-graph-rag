"""Evidence-attributed evaluation of graph-augmented and multimodal RAG."""

from .clients import ModelClient, PricingSnapshot, load_pricing
from .config import ExperimentConfig, load_experiment_config
from .errors import (
    CacheError,
    ConfigurationError,
    InvalidModelOutputError,
    MissingEvidenceError,
    MultimodalGraphRagError,
    PricingError,
    ProviderError,
)
from .schemas import (
    EvidenceBundle,
    EvidenceItem,
    QuestionRecord,
    RunRecord,
    load_questions,
    write_questions,
)

__all__ = [
    "CacheError",
    "ConfigurationError",
    "EvidenceBundle",
    "EvidenceItem",
    "ExperimentConfig",
    "InvalidModelOutputError",
    "MissingEvidenceError",
    "ModelClient",
    "MultimodalGraphRagError",
    "PricingError",
    "PricingSnapshot",
    "ProviderError",
    "QuestionRecord",
    "RunRecord",
    "load_experiment_config",
    "load_pricing",
    "load_questions",
    "write_questions",
]

__version__ = "0.3.0"
