"""Runtime settings, sourced from the environment.

Credentials are read from environment variables or a local ``.env`` only. They
are never accepted from task-profile YAML (§20.1) — profiles are checked into
version control and shared between projects, which is exactly the wrong place
for a UMLS key.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from biosemalign.enums import EgressDataClass, ResourceMode

__all__ = ["Settings", "get_settings", "reset_settings_cache"]


class Settings(BaseSettings):
    """Environment-driven configuration, prefixed ``BIOSEMALIGN_``."""

    model_config = SettingsConfigDict(
        env_prefix="BIOSEMALIGN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Resource access policy (§20.3) ---
    mode: ResourceMode = Field(
        default=ResourceMode.FROZEN,
        description=(
            "ONLINE refreshes the cache, CACHE_PREFER uses live services only on a miss, "
            "and FROZEN/OFFLINE are strictly cache-only. FROZEN additionally requires "
            "release-bound cache identities."
        ),
    )
    cache_dir: Path = Field(
        default=Path(".biosemalign_cache"),
        description="Provider response cache. Also serves as the offline fixture store.",
    )
    deployment_environment: (
        Literal["local", "institutional_server", "hpc", "restricted_enclave"] | None
    ) = Field(
        default=None,
        description=(
            "Explicit deployment authorization. Required before ONLINE or CACHE_PREFER "
            "can use external services; cache-only replay may leave it unset."
        ),
    )
    permitted_egress_classes: frozenset[EgressDataClass] = Field(
        default_factory=lambda: frozenset({EgressDataClass.PUBLIC_TERMINOLOGY}),
        description=(
            "Data classes permitted to leave the deployment in live modes. Lexical source "
            "values require an explicit sensitive_text opt-in."
        ),
    )
    cache_retention_days: int | None = Field(
        default=None,
        ge=1,
        description="Optional maximum cache-entry age; None retains immutable snapshots.",
    )
    output_retention_policy_days: int | None = Field(
        default=30,
        ge=1,
        description=(
            "Declared site policy metadata for generated outputs; deletion is deliberately "
            "left to the study's explicit archival/cleanup workflow."
        ),
    )
    cache_dir_mode: int = Field(default=0o700, ge=0, le=0o777)
    cache_file_mode: int = Field(default=0o600, ge=0, le=0o777)
    output_dir_mode: int = Field(default=0o700, ge=0, le=0o777)
    output_file_mode: int = Field(default=0o600, ge=0, le=0o777)
    cache_store_raw_queries: bool = Field(
        default=False,
        description="Persist provider query text in cache metadata; disabled for privacy by default.",
    )
    cache_allow_legacy_offline_entries: bool = Field(
        default=True,
        description=(
            "Allow pre-0.2 committed cassettes only in OFFLINE mode. FROZEN never accepts "
            "legacy entries without endpoint/release identity."
        ),
    )
    identifier_hmac_key: SecretStr | None = Field(
        default=None,
        description=(
            "Secret key for deployment-scoped, non-linkable identifier digests. Never emit "
            "the key itself; use identifier_hmac_key_fingerprint for provenance."
        ),
    )

    # --- LLM backend (§18.8). Unset in v0.1; the deterministic adjudicator needs none. ---
    llm_base_url: str | None = None
    llm_api_key: SecretStr | None = None
    llm_model: str | None = None

    # --- Licensed resources ---
    umls_api_key: SecretStr | None = Field(
        default=None, description="Requires an authorized UMLS account (§6.5)."
    )

    # --- HTTP behaviour ---
    rxnav_base_url: str = "https://rxnav.nlm.nih.gov/REST"
    rxnav_terminology_release: str | None = Field(
        default=None,
        description=(
            "Expected RxNorm release. Required in FROZEN mode; ONLINE validates it when set."
        ),
    )
    http_timeout_seconds: float = Field(default=30.0, gt=0)
    http_max_attempts: int = Field(default=4, ge=1)
    rxnav_requests_per_second: float = Field(
        default=15.0,
        ge=0.0,
        description="RxNav documents a ~20 req/s per-IP cap; stay under it by default.",
    )

    # --- Profiles ---
    profile_path: Path | None = Field(
        default=None, description="Extra directory to search for task profiles."
    )

    # --- Logging ---
    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"

    @field_validator(
        "cache_dir_mode",
        "cache_file_mode",
        "output_dir_mode",
        "output_file_mode",
        mode="before",
    )
    @classmethod
    def _parse_permission_mode(cls, value: object) -> object:
        """Accept conventional environment values such as ``0700`` and ``0o600``."""
        if not isinstance(value, str):
            return value
        text = value.strip().lower()
        if text.startswith("0o"):
            return int(text[2:], 8)
        if len(text) in {3, 4} and text and all(character in "01234567" for character in text):
            return int(text, 8)
        return int(text, 10)

    @field_validator("mode", mode="before")
    @classmethod
    def _parse_resource_mode(cls, value: object) -> object:
        """Make friendly string input deterministic while retaining enum validation."""
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @field_validator("deployment_environment", mode="before")
    @classmethod
    def _parse_deployment_environment(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("rxnav_base_url")
    @classmethod
    def _validate_rxnav_base_url(cls, value: str) -> str:
        endpoint = value.strip().rstrip("/")
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("rxnav_base_url must be an absolute HTTP(S) endpoint")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("rxnav_base_url must not contain credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("rxnav_base_url must not contain a query string or fragment")
        return endpoint

    @field_validator("identifier_hmac_key")
    @classmethod
    def _validate_identifier_hmac_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and len(value.get_secret_value().encode("utf-8")) < 32:
            raise ValueError("identifier_hmac_key must contain at least 32 bytes")
        return value

    @property
    def offline(self) -> bool:
        """True when the network must not be touched under any circumstance."""
        return self.mode in {ResourceMode.FROZEN, ResourceMode.OFFLINE}

    @property
    def network_enabled(self) -> bool:
        """True when the selected mode may issue an external request."""
        return self.mode in {ResourceMode.ONLINE, ResourceMode.CACHE_PREFER}

    @property
    def identifier_hmac_key_fingerprint(self) -> str | None:
        """Non-secret identifier for the configured key, safe for provenance."""
        if self.identifier_hmac_key is None:
            return None
        digest = hashlib.sha256(
            self.identifier_hmac_key.get_secret_value().encode("utf-8")
        ).hexdigest()
        return digest[:16]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, read once."""
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cached settings. Used by tests that manipulate the environment."""
    get_settings.cache_clear()
