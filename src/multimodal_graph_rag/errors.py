"""Exception types raised by the package.

Every failure that would otherwise be hidden behind a placeholder value (a
zero-scored answer, an empty triple list, a silently rebuilt cache, a guessed
price) is raised as one of these so that a run stops explicitly.
"""


class MultimodalGraphRagError(Exception):
    """Base class for all package errors."""


class ConfigurationError(MultimodalGraphRagError):
    """A configuration file, path, or argument is missing or inconsistent."""


class ProviderError(MultimodalGraphRagError):
    """A model provider call failed after the configured retry budget."""


class InvalidModelOutputError(MultimodalGraphRagError):
    """A model returned output that does not satisfy the requested contract."""


class PricingError(MultimodalGraphRagError):
    """No frozen price is available for a model that was charged."""


class CacheError(MultimodalGraphRagError):
    """A cache file exists but cannot be read or does not match its metadata."""


class MissingEvidenceError(MultimodalGraphRagError):
    """Evidence that a system must supply to the generator does not exist."""
