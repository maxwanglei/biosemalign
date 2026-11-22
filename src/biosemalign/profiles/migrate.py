"""Convert Schema 1 task profiles to Schema 2 (§5.8, §7)."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import yaml

from biosemalign.enums import DecisionType
from biosemalign.schemas.sau import ContextualAttributes

__all__ = ["is_schema_1_profile", "migrate_profile_data", "migrate_profile_yaml"]

_OCCURRENCE_ONLY = frozenset({"record_id", "evidence_block_id", "neighboring_fields", "extensions"})
_ATTRIBUTE_FIELDS = frozenset(ContextualAttributes.model_fields) - _OCCURRENCE_ONLY

_SCHEMA_1_TOP_KEYS = frozenset({"retrieval", "adjudication", "validation"})
_SCHEMA_1_TASK_KEYS = frozenset(
    {"target_terminology", "desired_granularity", "target_spec", "source_terminologies"}
)
_SCHEMA_1_INPUT_KEYS = frozenset({"required_context", "optional_context", "mapping_key_context"})
_SCHEMA_1_KNOWLEDGE_KEYS = frozenset(
    {"primary_resources", "secondary_resources", "blinded_context_fields"}
)

_CLEAN_RENAMES = {"extract_strength": "drug.extract_strength"}
_CHECK_RENAMES = {"drug_semantics": "drug.semantics"}


def is_schema_1_profile(raw: Mapping[str, Any]) -> bool:
    """Return True when ``raw`` uses Schema 1 layout or explicitly sets ``profile_schema: 1``."""
    schema_version = raw.get("profile_schema")
    if schema_version == 1:
        return True
    if schema_version is not None:
        return False
    if any(k in raw for k in _SCHEMA_1_TOP_KEYS):
        return True
    if isinstance(raw.get("workflow"), dict):
        return True
    task = raw.get("task")
    if isinstance(task, Mapping) and any(k in task for k in _SCHEMA_1_TASK_KEYS):
        return True
    target = raw.get("target")
    if isinstance(target, Mapping) and "granularity" in target:
        return True
    input_cfg = raw.get("input")
    if isinstance(input_cfg, Mapping) and any(k in input_cfg for k in _SCHEMA_1_INPUT_KEYS):
        return True
    knowledge = raw.get("knowledge")
    return isinstance(knowledge, Mapping) and any(k in knowledge for k in _SCHEMA_1_KNOWLEDGE_KEYS)


def migrate_profile_data(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a Schema 1 profile mapping into a Schema 2 mapping.

    Raises ``ValueError`` if the Schema 1 mapping requested removed or
    unsupported modes (such as ``retrieval.lexical_retrieval: true`` or
    ``adjudication.candidate_constrained: false``) that have no safe automatic
    translation.
    """
    data = deepcopy(dict(raw))
    if not is_schema_1_profile(data):
        return data

    data.pop("profile_schema", None)
    task_in = dict(data.pop("task", None) or {})
    target_in = dict(data.pop("target", None) or {})
    input_in = dict(data.pop("input", None) or {})
    knowledge_in = dict(data.pop("knowledge", None) or {})
    retrieval_in = dict(data.pop("retrieval", None) or {})
    adjudication_in = dict(data.pop("adjudication", None) or {})
    validation_in = dict(data.pop("validation", None) or {})
    workflow_in = data.pop("workflow", None)
    routing_in = data.pop("routing", None)

    _check_removed_flags(retrieval_in, adjudication_in, validation_in)

    target_out = _build_target(task_in, target_in, validation_in)
    task_out = _build_task(task_in, adjudication_in)
    context_out, input_out = _build_context_and_input(input_in, knowledge_in)
    knowledge_out = _build_knowledge(knowledge_in)
    limits_out = _build_limits(retrieval_in)
    tiers_out = _build_tiers(workflow_in, retrieval_in, adjudication_in)
    checks_out = _build_checks(validation_in)

    migrated: dict[str, Any] = {
        "profile_schema": 2,
        "profile_name": data["profile_name"],
        "profile_version": data["profile_version"],
    }
    if data.get("description") is not None:
        migrated["description"] = data["description"]
    domain = data.get("domain") or _infer_domain(task_out, target_out)
    if domain is not None:
        migrated["domain"] = domain
    migrated["workflow"] = workflow_in if isinstance(workflow_in, str) else "mapping"
    migrated["task"] = task_out
    migrated["target"] = target_out
    if input_out:
        migrated["input"] = input_out
    if context_out["fields"]:
        migrated["context"] = context_out
    if knowledge_out:
        migrated["knowledge"] = knowledge_out
    migrated["limits"] = limits_out
    if tiers_out is not None:
        migrated["tiers"] = tiers_out
    migrated["checks"] = checks_out
    if routing_in is not None:
        migrated["routing"] = routing_in
    return migrated


def migrate_profile_yaml(text: str) -> str:
    """Parse YAML text for a Schema 1 profile and dump it as Schema 2 YAML."""
    raw = yaml.safe_load(text)
    if not isinstance(raw, Mapping):
        raise ValueError("profile YAML must contain a mapping at the top level")
    migrated = migrate_profile_data(raw)
    return yaml.safe_dump(migrated, sort_keys=False)


def _check_removed_flags(
    retrieval_in: Mapping[str, Any],
    adjudication_in: Mapping[str, Any],
    validation_in: Mapping[str, Any],
) -> None:
    if retrieval_in.get("lexical_retrieval"):
        raise ValueError(
            "retrieval.lexical_retrieval was removed in Schema 2; configure a 'lexical' matcher"
        )
    if retrieval_in.get("embedding_retrieval"):
        raise ValueError(
            "retrieval.embedding_retrieval was removed in Schema 2; configure a 'dense' matcher"
        )
    if adjudication_in.get("candidate_constrained") is False:
        raise ValueError("adjudication.candidate_constrained=false is not supported in Schema 2")
    if validation_in.get("prohibit_unretrieved_identifiers") is False:
        raise ValueError(
            "validation.prohibit_unretrieved_identifiers=false is not supported in Schema 2 (G4)"
        )


def _infer_domain(task_out: Mapping[str, Any], target_out: Mapping[str, Any]) -> str | None:
    if task_out.get("operation") == "drug_normalization":
        return "drug"
    if str(target_out.get("spec") or "").casefold() == "rxnorm":
        return "drug"
    if str(target_out.get("terminology") or "").casefold() == "rxnorm":
        return "drug"
    return None


def _build_target(
    task_in: dict[str, Any], target_in: dict[str, Any], validation_in: dict[str, Any]
) -> dict[str, Any]:
    terminology = target_in.pop("terminology", None) or task_in.pop("target_terminology", None)
    spec = target_in.pop("spec", None) or task_in.pop("target_spec", None)
    level = (
        target_in.pop("level", None)
        or target_in.pop("granularity", None)
        or task_in.pop("desired_granularity", None)
    )
    sources = target_in.pop("source_terminologies", None)
    if sources is None:
        sources = task_in.pop("source_terminologies", [])
    else:
        task_in.pop("source_terminologies", None)

    out: dict[str, Any] = {}
    if terminology is not None:
        out["terminology"] = terminology
    if spec is not None:
        out["spec"] = spec
    if level is not None:
        out["level"] = level
    if sources:
        out["source_terminologies"] = list(sources)
    if "require_active_concept" in validation_in:
        out["require_active_concept"] = bool(validation_in.pop("require_active_concept"))
    return out


def _build_task(task_in: dict[str, Any], adjudication_in: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(task_in)
    if "relation_types" not in out and "relation_types" in adjudication_in:
        out["relation_types"] = list(adjudication_in["relation_types"])
    return out


def _build_context_and_input(
    input_in: dict[str, Any], knowledge_in: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    required = list(input_in.pop("required_context", []) or [])
    optional = list(input_in.pop("optional_context", []) or [])
    key_fields = list(input_in.pop("mapping_key_context", []) or [])
    blinded = list(knowledge_in.pop("blinded_context_fields", []) or [])

    seen: list[str] = []
    for name in [*required, *optional, *key_fields, *blinded]:
        if name not in seen:
            seen.append(name)

    fields: list[dict[str, Any]] = []
    for name in seen:
        is_attr = name in _ATTRIBUTE_FIELDS
        is_req = name in required
        is_opt = name in optional
        is_key = name in key_fields
        is_blind = name in blinded

        use: list[str] = []
        if is_blind:
            if is_key:
                use = ["key"]
        elif is_key and (is_req or is_opt):
            use = ["key", "resolve", "check"]
        elif is_key:
            use = ["key"]
        elif is_req or is_opt:
            use = ["resolve", "check"] if is_attr else []

        entry: dict[str, Any] = {
            "name": name,
            "kind": "attribute" if is_attr else "context",
            "use": use,
        }
        if is_req:
            entry["required"] = True
        if is_blind:
            entry["blinded"] = True
        fields.append(entry)

    return {"fields": fields}, input_in


def _build_knowledge(knowledge_in: dict[str, Any]) -> dict[str, Any]:
    primary = list(knowledge_in.pop("primary_resources", []) or [])
    secondary = list(knowledge_in.pop("secondary_resources", []) or [])
    resources = list(knowledge_in.pop("resources", []) or [])
    if not resources and (primary or secondary):
        resources = list(dict.fromkeys([*primary, *secondary]))
    out: dict[str, Any] = {}
    if resources:
        out["resources"] = resources
    for key in ("allowed_relationships", "prohibited_relationships"):
        if key in knowledge_in:
            out[key] = list(knowledge_in[key])
    return out


def _build_limits(retrieval_in: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "candidates": retrieval_in.get("maximum_candidates", 10),
        "members": 2000,
    }
    if "progressive_enrichment_batch_size" in retrieval_in:
        out["enrichment_batch_size"] = retrieval_in["progressive_enrichment_batch_size"]
    return out


def _as_op(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {"op": value, "options": {}}
    if isinstance(value, Mapping):
        return {"op": str(value["op"]), "options": dict(value.get("options") or {})}
    raise ValueError(f"expected operator config or name, got {type(value).__name__}")


def _compact_op(op: str, options: dict[str, Any]) -> str | dict[str, Any]:
    return {"op": op, "options": options} if options else op


def _build_checks(validation_in: dict[str, Any]) -> list[str | dict[str, Any]]:
    rules = validation_in.pop("rules", None)
    preserve = validation_in.pop("preserve_combination_ingredients", None)
    flag_salt = validation_in.pop("flag_salt_active_moiety_conflation", None)
    if rules is None:
        rules = [{"op": "drug_semantics"}]

    checks: list[str | dict[str, Any]] = []
    for raw_rule in rules:
        parsed = _as_op(raw_rule)
        op_name = _CHECK_RENAMES.get(parsed["op"], parsed["op"])
        options = dict(parsed["options"])
        if op_name == "drug.semantics":
            if preserve is not None and "preserve_combination_ingredients" not in options:
                options["preserve_combination_ingredients"] = bool(preserve)
            if flag_salt is not None and "flag_salt_active_moiety_conflation" not in options:
                options["flag_salt_active_moiety_conflation"] = bool(flag_salt)
        checks.append(_compact_op(op_name, options))
    return checks


def _build_tiers(
    workflow_in: Any,
    retrieval_in: Mapping[str, Any],
    adjudication_in: Mapping[str, Any],
) -> list[dict[str, Any]] | None:
    if not isinstance(workflow_in, Mapping):
        return [_default_tier(retrieval_in, adjudication_in)]
    raw_tiers = workflow_in.get("tiers") or []
    migrated_tiers: list[dict[str, Any]] = []
    for raw_tier in raw_tiers:
        migrated_tiers.extend(_convert_tier(dict(raw_tier), retrieval_in, adjudication_in))
    return migrated_tiers


def _default_rxnav_match(retrieval_in: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "op": "drug.rxnav",
        "options": {
            "exact_lookup": bool(retrieval_in.get("exact_lookup", True)),
            "approximate_match": bool(retrieval_in.get("terminology_approximate_match", True)),
            "include_historical": bool(retrieval_in.get("include_historical_candidates", True)),
        },
    }


def _default_resolve(adjudication_in: Mapping[str, Any]) -> str | dict[str, Any]:
    backend = str(adjudication_in.get("backend", "deterministic")).strip().casefold()
    if backend == "deterministic":
        return "rules"
    if backend == "vllm":
        options: dict[str, Any] = {}
        if "temperature" in adjudication_in:
            options["temperature"] = adjudication_in["temperature"]
        if "llm_max_candidates" in adjudication_in:
            options["max_candidates"] = adjudication_in["llm_max_candidates"]
        if adjudication_in.get("prompt_version") is not None:
            options["prompt_version"] = adjudication_in["prompt_version"]
        return _compact_op("llm", options)
    if backend == "mock":
        return "mock"
    raise ValueError(f"unknown Schema 1 adjudication backend {backend!r}")


def _default_tier(
    retrieval_in: Mapping[str, Any], adjudication_in: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "name": "default",
        "match": [_default_rxnav_match(retrieval_in)],
        "link": {"op": "standard", "options": {"merge": "priority", "concordance": False}},
        "resolve": _default_resolve(adjudication_in),
    }


def _convert_tier(
    tier: dict[str, Any],
    retrieval_in: Mapping[str, Any],
    adjudication_in: Mapping[str, Any],
) -> list[dict[str, Any]]:
    clean_ops = [
        _compact_op(_CLEAN_RENAMES.get(op["op"], op["op"]), op["options"])
        for op in (_as_op(x) for x in tier.get("transforms", []))
    ]
    raw_match = tier.get("retrievers") if "retrievers" in tier else tier.get("coding")
    if raw_match is None:
        raw_match = ["rxnav"]
    if not isinstance(raw_match, list):
        raw_match = [raw_match]
    parsed_match = [_as_op(m) for m in raw_match]

    raw_link = tier.get("fuser") if "fuser" in tier else tier.get("linking")
    link_cfg = _convert_link(raw_link)

    raw_resolve = tier.get("proposer") if "proposer" in tier else tier.get("resolution")
    resolve_cfg, pass_on, pass_on_review = _convert_resolve(raw_resolve, adjudication_in)

    # Appendix B: a single ``lexicon`` matcher with multiple methods (the old default)
    # becomes an ``exact`` tier that passes on to a ``lexical`` tier.
    if len(parsed_match) == 1 and parsed_match[0]["op"] == "lexicon":
        lex_opts = dict(parsed_match[0]["options"])
        methods = list(lex_opts.pop("methods", ["exact", "trigram"]))
        if len(methods) > 1:
            return _split_lexicon_methods(
                name=str(tier["name"]),
                clean=clean_ops,
                methods=methods,
                lex_opts=lex_opts,
                link_cfg=link_cfg,
                resolve_cfg=resolve_cfg,
                pass_on=pass_on,
                pass_on_review=pass_on_review,
            )

    match_ops = [_convert_single_match(m, retrieval_in) for m in parsed_match]
    out: dict[str, Any] = {
        "name": tier["name"],
        "match": match_ops,
        "link": link_cfg,
        "resolve": resolve_cfg,
    }
    if clean_ops:
        out["clean"] = clean_ops
    if pass_on:
        out["pass_on"] = pass_on
    if pass_on_review:
        out["pass_on_review_request"] = True
    return [out]


def _split_lexicon_methods(
    *,
    name: str,
    clean: list[str | dict[str, Any]],
    methods: list[str],
    lex_opts: dict[str, Any],
    link_cfg: str | dict[str, Any],
    resolve_cfg: str | dict[str, Any],
    pass_on: list[str],
    pass_on_review: bool,
) -> list[dict[str, Any]]:
    split: list[dict[str, Any]] = []
    for index, method in enumerate(methods):
        is_last = index == len(methods) - 1
        if method == "exact":
            match_op = _compact_op("exact", dict(lex_opts))
        else:
            match_op = _compact_op("lexical", {**lex_opts, "method": method})
        tier_entry: dict[str, Any] = {
            "name": name if is_last else f"{name}_{method}",
            "match": [match_op],
            "link": link_cfg,
            "resolve": resolve_cfg,
        }
        if clean:
            tier_entry["clean"] = clean
        if is_last:
            if pass_on:
                tier_entry["pass_on"] = pass_on
            if pass_on_review:
                tier_entry["pass_on_review_request"] = True
        else:
            tier_entry["pass_on"] = list(
                dict.fromkeys([*pass_on, DecisionType.NO_VALID_MAPPING.value])
            )
            if pass_on_review:
                tier_entry["pass_on_review_request"] = True
        split.append(tier_entry)
    return split


def _convert_single_match(
    op: dict[str, Any], retrieval_in: Mapping[str, Any]
) -> str | dict[str, Any]:
    name = op["op"]
    opts = dict(op["options"])
    if name in {"rxnav", "drug.rxnav"}:
        merged = {
            "exact_lookup": bool(retrieval_in.get("exact_lookup", True)),
            "approximate_match": bool(retrieval_in.get("terminology_approximate_match", True)),
            "include_historical": bool(retrieval_in.get("include_historical_candidates", True)),
            **opts,
        }
        return _compact_op("drug.rxnav", merged)
    if name == "lexicon":
        methods = list(opts.pop("methods", ["exact"]))
        method = methods[0]
        if method == "exact":
            return _compact_op("exact", opts)
        return _compact_op("lexical", {**opts, "method": method})
    return _compact_op(name, opts)


def _convert_link(raw_link: Any) -> str | dict[str, Any]:
    if raw_link is None:
        return "standard"
    parsed = _as_op(raw_link)
    name = parsed["op"]
    opts = dict(parsed["options"])
    if name == "knowledge_package":
        return _compact_op("standard", opts)
    if name == "reciprocal_rank_fusion":
        return _compact_op("standard", {"merge": "rrf", **opts})
    if name in {"cross_vocab_concordance", "concordance"}:
        return _compact_op("standard", {"concordance": True, **opts})
    return _compact_op(name, opts)


def _convert_resolve(
    raw_resolve: Any, adjudication_in: Mapping[str, Any]
) -> tuple[str | dict[str, Any], list[str], bool]:
    if raw_resolve is None:
        return _default_resolve(adjudication_in), [], False
    parsed = _as_op(raw_resolve)
    name = parsed["op"]
    opts = dict(parsed["options"])
    pass_on = [str(x) for x in opts.pop("pass_on", [])]
    pass_on_review = bool(opts.pop("pass_on_review_request", False))
    if name in {"concordant_exact", "concordant"}:
        return _compact_op("concordant", opts), pass_on, pass_on_review
    if name == "adjudicator":
        if opts:
            # Explicit options or default backend mapping
            base = _default_resolve(adjudication_in)
            if isinstance(base, dict):
                merged = {**base["options"], **opts}
                return _compact_op(base["op"], merged), pass_on, pass_on_review
            return _compact_op(base, opts), pass_on, pass_on_review
        return _default_resolve(adjudication_in), pass_on, pass_on_review
    return _compact_op(name, opts), pass_on, pass_on_review
