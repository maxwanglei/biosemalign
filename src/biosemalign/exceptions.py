"""Exception hierarchy.

All errors derive from :class:`BioSemAlignError` so a calling application can
distinguish "the framework refused" from "the framework crashed".
"""

from __future__ import annotations

__all__ = [
    "AdjudicationFailure",
    "BioSemAlignError",
    "CacheCorruption",
    "CacheMiss",
    "ConfigurationError",
    "ProfileError",
    "ProviderError",
    "ResourceNotPermitted",
    "UnsupportedInput",
]


class BioSemAlignError(Exception):
    """Base class for every error raised by BioSemAlign."""


class ConfigurationError(BioSemAlignError):
    """Settings or environment are not usable."""


class ProfileError(ConfigurationError):
    """A task profile is missing, malformed, or internally inconsistent."""


class ResourceNotPermitted(BioSemAlignError):
    """A resource was requested that the registry forbids in this environment.

    Raised, for example, when a profile asks for UMLS with no licence key
    configured, or when a restricted-data deployment has disabled network
    providers entirely (§20.2).
    """


class ProviderError(BioSemAlignError):
    """An external knowledge provider failed or returned something unusable."""


class CacheMiss(ProviderError):
    """A required response was absent from a network-forbidden cache mode.

    This is deliberately an error rather than a silent network call: it is what
    makes "this test quietly hit the live API" impossible (§20.3).
    """


class CacheCorruption(ProviderError):
    """A cache entry exists but failed its identity or integrity checks.

    Corruption is deliberately distinct from a miss.  Treating a truncated or
    tampered entry as absent would let a supposedly frozen run replace it with
    live data and silently change the evidence used by the analysis.
    """


class UnsupportedInput(BioSemAlignError):
    """The input cannot be represented as a Semantic Alignment Unit."""


class AdjudicationFailure(BioSemAlignError):
    """The adjudicator produced nothing usable after its retries."""
