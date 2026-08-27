"""Terminology providers. Every provider satisfies the same protocol (§13.1)."""

from biosemalign.knowledge.providers.base import KnowledgeProvider
from biosemalign.knowledge.providers.rxnorm import RxNavProvider

__all__ = ["KnowledgeProvider", "RxNavProvider"]
