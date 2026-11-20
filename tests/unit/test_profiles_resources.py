"""Configuration must fail explicitly before it can change mapping policy."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from biosemalign.enums import MentionMode, RelationshipType
from biosemalign.exceptions import ConfigurationError, ProfileError, ResourceNotPermitted
from biosemalign.profiles.loader import available_profiles, load_profile, load_profile_file
from biosemalign.profiles.schema import (
    AdjudicationConfig,
    InputConfig,
    KnowledgeConfig,
    RetrievalConfig,
    RoutingConfig,
    TaskProfile,
    ValidationConfig,
)
from biosemalign.resources.registry import (
    DeploymentEnvironment,
    ResourceRegistry,
    ResourceSpec,
    get_registry,
)
from biosemalign.settings import reset_settings_cache


def test_packaged_profile_and_predicate_policy():
    profile = load_profile("faers_drug_normalization")
    assert profile.task.desired_granularity == "ingredient"
    assert profile.input.mention_mode is MentionMode.USER_SPECIFIED
    assert profile.knowledge.permits("has_ingredient")
    assert not profile.knowledge.permits("may_treat")
    assert not profile.knowledge.permits("unlisted")
    assert profile.permits_relation(RelationshipType.EXACT)
    assert not profile.permits_relation(RelationshipType.RELATED_NOT_EQUIVALENT)
    assert KnowledgeConfig().permits("custom")
    assert not KnowledgeConfig(prohibited_relationships=["custom"]).permits("custom")
    profile.adjudication.relation_types = []
    assert profile.permits_relation(RelationshipType.EXACT)


@pytest.mark.parametrize(
    ("model", "values", "message"),
    [
        (InputConfig, {"mapping_key_context": ["route", "route"]}, "duplicate"),
        (InputConfig, {"mapping_key_context": ["record_id"]}, "occurrence-only"),
        (InputConfig, {"mapping_key_context": ["typo"]}, "unknown canonical"),
        (InputConfig, {"automatic_mention_detection": True}, "not implemented"),
        (
            KnowledgeConfig,
            {"allowed_relationships": ["x"], "prohibited_relationships": ["x"]},
            "both",
        ),
        (RetrievalConfig, {"lexical_retrieval": True}, "not implemented"),
        (RetrievalConfig, {"embedding_retrieval": True}, "not implemented"),
        (AdjudicationConfig, {"candidate_constrained": False}, "bounded"),
        (ValidationConfig, {"prohibit_unretrieved_identifiers": False}, "bounded"),
        (RoutingConfig, {"rejection_lexical_similarity": 0.9}, "must not exceed"),
        (RetrievalConfig, {"maximum_candidates": 0}, "greater than or equal"),
    ],
)
def test_invalid_policy_is_rejected(model, values, message):
    with pytest.raises(ValidationError, match=message):
        model(**values)


def test_profile_target_must_be_retrievable():
    payload = load_profile("faers_drug_normalization").model_dump()
    payload["task"]["target_terminology"] = "OtherVocabulary"
    with pytest.raises(ValidationError, match="not among"):
        TaskProfile.model_validate(payload)


def test_profile_discovery_precedence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    local = tmp_path / "configs" / "profiles"
    local.mkdir(parents=True)
    configured = tmp_path / "profiles"
    configured.mkdir()
    payload = load_profile("faers_drug_normalization").model_dump(mode="json")
    for directory, version in [(local, "local"), (configured, "configured")]:
        payload["profile_version"] = version
        (directory / "faers_drug_normalization.yml").write_text(yaml.safe_dump(payload))
    monkeypatch.setenv("BIOSEMALIGN_PROFILE_PATH", str(configured))
    reset_settings_cache()
    assert load_profile("faers_drug_normalization").profile_version == "configured"
    assert load_profile(local / "faers_drug_normalization.yml").profile_version == "local"
    assert available_profiles() == ["faers_drug_normalization"]
    with pytest.raises(ProfileError, match="unknown profile"):
        load_profile("does_not_exist")
    reset_settings_cache()


@pytest.mark.parametrize(
    "contents,message",
    [
        ("[invalid", "not valid YAML"),
        ("- list", "mapping"),
        ("profile_name: incomplete", "not a valid task profile"),
    ],
)
def test_bad_profile_file(tmp_path, contents, message):
    path = tmp_path / "bad.yaml"
    path.write_text(contents)
    with pytest.raises(ProfileError, match=message):
        load_profile_file(path)


def test_missing_profile_file(tmp_path):
    with pytest.raises(ProfileError, match="not found"):
        load_profile_file(tmp_path / "missing.yaml")


def test_resource_gates_and_case_insensitive_lookup(monkeypatch):
    get_registry.cache_clear()
    registry = get_registry()
    rxnorm = registry.require("rxNORM")
    assert rxnorm.provider == "rxnav"
    with pytest.raises(ResourceNotPermitted, match="not permitted"):
        registry.require("RxNorm", environment=DeploymentEnvironment.RESTRICTED_ENCLAVE)
    with pytest.raises(ResourceNotPermitted, match="not implemented"):
        registry.require("UMLS")
    with pytest.raises(ResourceNotPermitted, match="not in the registry"):
        registry.get("Unknown")
    licensed = ResourceSpec(
        name="licensed",
        provider="local",
        access_method="local",
        license_required=True,
        credential_env_var="BIOSEMALIGN_UMLS_API_KEY",
    )
    monkeypatch.delenv("BIOSEMALIGN_UMLS_API_KEY", raising=False)
    reset_settings_cache()
    with pytest.raises(ResourceNotPermitted, match="licence credential"):
        licensed.check_usable(environment=DeploymentEnvironment.LOCAL)
    monkeypatch.setenv("BIOSEMALIGN_UMLS_API_KEY", "local-test-secret")
    reset_settings_cache()
    licensed.check_usable(environment=DeploymentEnvironment.HPC)
    ResourceRegistry(resources=[licensed]).require("LICENSED")
    reset_settings_cache()


@pytest.mark.parametrize("contents", ["[bad", "resources: [not-an-object]"])
def test_invalid_registry_override(tmp_path, monkeypatch, contents):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "resources.yaml").write_text(contents)
    get_registry.cache_clear()
    with pytest.raises(ConfigurationError, match="not a valid resource registry"):
        get_registry()
    get_registry.cache_clear()


def test_valid_registry_override(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = Path("configs/resources.yaml")
    path.parent.mkdir()
    path.write_text("resources: [{name: Local, provider: local, access_method: file}]")
    get_registry.cache_clear()
    assert get_registry().require("local").access_method == "file"
    get_registry.cache_clear()
