"""Deployment settings and logging behavior, without external services."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import structlog
from pydantic import SecretStr, ValidationError

from biosemalign import logging as app_logging
from biosemalign.enums import EgressDataClass, ResourceMode
from biosemalign.settings import Settings, get_settings, reset_settings_cache


@pytest.mark.parametrize(
    ("value", "expected"),
    [("0700", 0o700), ("0o600", 0o600), ("755", 0o755), ("493", 493), (448, 448)],
)
def test_permission_settings_accept_octal_or_decimal(value: object, expected: int) -> None:
    settings = Settings(_env_file=None, cache_dir_mode=value)
    assert settings.cache_dir_mode == expected


@pytest.mark.parametrize("value", ["nonsense", "0o888", -1, 0o1000])
def test_permission_settings_reject_invalid_values(value: object) -> None:
    with pytest.raises((ValidationError, ValueError)):
        Settings(_env_file=None, output_file_mode=value)


@pytest.mark.parametrize(
    ("mode", "offline"),
    [(" frozen ", True), ("offline", True), ("online", False), ("cache_prefer", False)],
)
def test_access_mode_is_explicit_and_case_insensitive(mode: str, offline: bool) -> None:
    settings = Settings(_env_file=None, mode=mode, deployment_environment=" LOCAL ")
    assert settings.offline is offline
    assert settings.network_enabled is not offline
    assert settings.deployment_environment == "local"


def test_defaults_do_not_authorize_sensitive_text_egress() -> None:
    settings = Settings(_env_file=None)
    assert settings.mode is ResourceMode.FROZEN
    assert settings.deployment_environment is None
    assert settings.permitted_egress_classes == {EgressDataClass.PUBLIC_TERMINOLOGY}


@pytest.mark.parametrize(
    "endpoint",
    [
        "relative",
        "ftp://example.test",
        "https://user:secret@example.test",
        "https://example.test?a=b",
        "https://example.test/#fragment",
    ],
)
def test_endpoint_validation_rejects_ambiguous_or_credentialed_urls(endpoint: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, rxnav_base_url=endpoint)


def test_endpoint_trailing_slash_is_canonicalized() -> None:
    settings = Settings(_env_file=None, rxnav_base_url=" https://example.test/REST/ ")
    assert settings.rxnav_base_url == "https://example.test/REST"


def test_secret_key_is_validated_and_only_fingerprint_is_public() -> None:
    with pytest.raises(ValidationError, match="32 bytes"):
        Settings(_env_file=None, identifier_hmac_key="short")
    first = Settings(_env_file=None, identifier_hmac_key=SecretStr("a" * 32))
    same = Settings(_env_file=None, identifier_hmac_key="a" * 32)
    other = Settings(_env_file=None, identifier_hmac_key="b" * 32)
    assert first.identifier_hmac_key_fingerprint == same.identifier_hmac_key_fingerprint
    assert first.identifier_hmac_key_fingerprint != other.identifier_hmac_key_fingerprint
    assert len(first.identifier_hmac_key_fingerprint or "") == 16
    assert "a" * 32 not in first.model_dump_json()
    assert Settings(_env_file=None).identifier_hmac_key_fingerprint is None


def test_environment_and_settings_cache_are_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BIOSEMALIGN_MODE", "offline")
    monkeypatch.setenv(
        "BIOSEMALIGN_PERMITTED_EGRESS_CLASSES", '["public_terminology", "sensitive_text"]'
    )
    monkeypatch.setenv("BIOSEMALIGN_CACHE_FILE_MODE", "0600")
    reset_settings_cache()
    first = get_settings()
    assert first.mode is ResourceMode.OFFLINE
    assert first.cache_file_mode == 0o600
    assert EgressDataClass.SENSITIVE_TEXT in first.permitted_egress_classes
    monkeypatch.setenv("BIOSEMALIGN_MODE", "online")
    assert get_settings() is first
    reset_settings_cache()
    assert get_settings().mode is ResourceMode.ONLINE


@pytest.mark.parametrize("fmt", ["json", "console"])
def test_logging_configures_once_and_allows_explicit_reconfiguration(
    fmt: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    previous = structlog.get_config()
    monkeypatch.setattr(app_logging, "_configured", False)
    try:
        app_logging.configure_logging(level="UNKNOWN", fmt=fmt)
        processors = structlog.get_config()["processors"]
        app_logging.configure_logging()
        assert structlog.get_config()["processors"] is processors
        app_logging.get_logger("m0-settings-test").info("mapping_recorded", record_id="opaque-1")
        captured = capsys.readouterr().err
        if fmt == "json":
            payload = json.loads(captured)
            assert payload["event"] == "mapping_recorded"
            assert payload["record_id"] == "opaque-1"
        else:
            assert "mapping_recorded" in captured
        app_logging.configure_logging(level="ERROR", fmt="json")
        app_logging.get_logger("m0-filter-test").info("filtered")
        assert capsys.readouterr().err == ""
    finally:
        structlog.configure(**previous)
