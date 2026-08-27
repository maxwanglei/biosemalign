"""Small shared helpers with no dependencies on the rest of the package."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from datetime import UTC, datetime
from difflib import SequenceMatcher
from typing import Any

__all__ = [
    "lexical_similarity",
    "normalize_text",
    "stable_digest",
    "stable_id",
    "tokenize",
    "utc_now",
]

_WHITESPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9]+")


def utc_now() -> datetime:
    """Timezone-aware current time. Never use naive datetimes in provenance."""
    return datetime.now(UTC)


def stable_digest(payload: Any, *, length: int = 64, key: str | bytes | None = None) -> str:
    """Deterministic SHA-256 or HMAC-SHA-256 digest of a JSON payload.

    Keys are sorted so that logically identical structures produce identical
    digests across runs and across machines, which is what makes the provider
    cache reproducible (§17.2).
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    if key is None:
        digest = hashlib.sha256(encoded).hexdigest()
    else:
        secret = key.encode("utf-8") if isinstance(key, str) else key
        digest = hmac.new(secret, encoded, hashlib.sha256).hexdigest()
    return digest[:length]


def stable_id(
    prefix: str,
    payload: Any,
    *,
    length: int = 12,
    key: str | bytes | None = None,
) -> str:
    """A short, deterministic, human-recognizable identifier.

    Used for SAU identifiers so that reprocessing the same input reproduces the
    same identifier and outputs can be diffed between runs.
    """
    return f"{prefix}-{stable_digest(payload, length=length, key=key).upper()}"


def normalize_text(value: str) -> str:
    """Case-fold and collapse whitespace for lexical comparison.

    Deliberately conservative: it does not strip punctuation or expand
    abbreviations, because those transformations lose information that the
    terminology-native matchers handle better.
    """
    return _WHITESPACE.sub(" ", value.strip()).casefold()


def tokenize(value: str) -> list[str]:
    """Alphanumeric tokens of two or more characters, case-folded.

    Single characters are dropped so that filler words ("a" in "NOT A DRUG")
    do not inflate overlap denominators, while dosage tokens such as "50" and
    "mg" are kept — those carry most of the signal in a reported product name.
    """
    return [t for t in _TOKEN.findall(value.casefold()) if len(t) >= 2]


def lexical_similarity(source: str, target: str) -> float:
    """How much a reported string and a concept label have in common, in [0, 1].

    Two measures, combined by taking the better of them, because they catch
    different things:

    * character-sequence ratio catches misspellings — "amoxicilin" against
      "amoxicillin" scores about 0.95 despite sharing no whole token;
    * token overlap catches verbose terminology labels — "TOPROL XL 50 MG"
      against "24 HR metoprolol succinate 50 MG Extended Release Oral Tablet
      [Toprol]" scores 0.75 on shared tokens while its character ratio is poor.

    A string with genuinely nothing in common scores near zero on both, which
    is what makes this usable as a floor for automatic acceptance.
    """
    a, b = normalize_text(source), normalize_text(target)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    sequence_ratio = SequenceMatcher(None, a, b).ratio()

    source_tokens = set(tokenize(a))
    overlap = len(source_tokens & set(tokenize(b))) / len(source_tokens) if source_tokens else 0.0

    return max(sequence_ratio, overlap)
