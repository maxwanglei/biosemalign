"""Adjudicator selection.

The profile names a backend; this resolves it. ``vllm`` is declared and
deliberately unimplemented in v0.1 so that a profile requesting it fails with
an explanation rather than silently falling back to different behaviour — a
silent downgrade would corrupt an evaluation comparing the two.
"""

from __future__ import annotations

from biosemalign.adjudication.backend import Adjudicator
from biosemalign.adjudication.deterministic import DeterministicAdjudicator
from biosemalign.adjudication.mock_backend import MockAdjudicator
from biosemalign.exceptions import ConfigurationError
from biosemalign.profiles.schema import TaskProfile

__all__ = ["build_adjudicator"]


def build_adjudicator(profile: TaskProfile) -> Adjudicator:
    """Construct the adjudicator the profile asks for."""
    backend = profile.adjudication.backend.strip().casefold()

    if backend == "deterministic":
        return DeterministicAdjudicator(profile)
    if backend == "mock":
        return MockAdjudicator()
    if backend == "vllm":
        raise ConfigurationError(
            "the vLLM adjudicator arrives in v0.2. Set adjudication.backend to "
            "'deterministic' to run the terminology-only pipeline."
        )
    raise ConfigurationError(
        f"unknown adjudication backend {profile.adjudication.backend!r}; "
        "expected 'deterministic', 'mock', or 'vllm'"
    )
