# Changelog

## Unreleased — WP7: extension kit and documents

Core schema `0.4.0`. Artifacts serialized under `0.2.0` and `0.3.0` still verify unchanged.

### Added

- Public contract test kit (`biosemalign.testing.contracts` re-exported from `biosemalign.testing`): `check_cleaner_contract`, `check_matcher_contract`, `check_expander_contract`, `check_linker_contract`, `check_resolver_contract`, `check_selector_contract`, `check_check_contract`, `check_target_spec_contract`, `check_domain_pack_contract`, and fixture builders `make_contract_sau`, `make_contract_candidate`, `make_contract_package`, plus verification in `tests/unit/test_extension_kit.py`.
- Section 6 extension guides under `docs/guides/` covering target specs, local controlled lists, all module operators, model-backed operators and backends, prompts, checks/policy packs, context fields, terminology providers, domain packs, and workflows.
- Installable example third-party plugin distribution under `examples/pk_pack/` (`biosemalign-pk-pack`) advertising entry-point group `biosemalign.plugins` (`pk = "biosemalign_pk_pack:register"`) with synthetic `ToyPK` target spec, cleaner, check, and profile, tested in CI.

### Changed

- Auto-loaded entry-point plugins (`ensure_plugins_loaded()`) in `biosemalign.modules.registry` (`available`, `build`) and `biosemalign.modules.model` (`available_llm_backends`, `build_llm_backend`).
- Removed `BioSemAlign.route()` and `BioSemAlign._provenance()` so no alignment decision can be minted outside the kernel (`cli.py:inspect` now runs `BioSemAlign.align_artifact()`).

## Unreleased — WP6: the phenotyping workflow

Core schema `0.4.0`. Artifacts serialized under `0.2.0` and `0.3.0` still verify unchanged.

### Added

- Phenotyping data contracts (`ConceptSetScope`, `ConceptSetMember`, `ConceptSetProposal`, `ConceptSetDecision`, `ConceptSetArtifact`, `MemberReviewAction`, `ConceptSetReview`, `ConceptSet`) in `biosemalign.schemas.phenotyping` and registered in `ROOT_CONTRACTS`.
- Expand operators `hierarchy` (`HierarchyExpander`, downward `subsumption` + `granularity.via` BFS with optional ancestors/siblings) and `lexical_neighbors` (`LexicalNeighborsExpander`, BM25/trigram neighbors), plus selector `hierarchy` (`HierarchySelector`) registered under `biosemalign.modules.select`.
- Phenotyping workflow (`PhenotypingWorkflow`, `phenotyping_workflow_from_profile`, `apply_concept_set_review`, `export_concept_set`) in `biosemalign.workflows.phenotyping` enforcing kernel guarantees G2, G4, G5, G7, and G8 (human review required before export; reviewer-added codes verified in terminology and marked `added_by_reviewer=True`).
- `JudgeMembersReply`, `MemberJudgment`, and `ScriptedModelSelector` in `biosemalign.testing` enforcing M1 (integer-indexed JSON reply forbidding terminology codes), M2 (surfacing model/hierarchy disagreement), M5, M6, and M9.
- `BioSemAlign.propose_concept_set()`, `BioSemAlign.run()` phenotyping dispatch, interactive `BioSemAlign.expand_code()`, CLI `biosemalign phenotype {propose, review, export, verify}`, and `verify_artifact` support for `ConceptSetArtifact` and `ConceptSet`.

### Changed

- Updated `Selector.select` protocol signature in `biosemalign.modules.base` to accept `ExpansionSet`, and widened `TierConfig.expand` in `biosemalign.profiles.schema` to accept a list of expand operators alongside a single operator.

## Unreleased — WP5: the coding workflow

Core schema `0.4.0`. Artifacts serialized under `0.2.0` and `0.3.0` still verify unchanged.

### Added

- Coding workflow contracts (`CodedConcept`, `CodingTierAttempt`, `CodingArtifact`) registered in `ROOT_CONTRACTS` and verified by `verify_artifact` / `verify_jsonl` / `biosemalign verify`.
- Per-SAU coding cascade (`CodingTier`, `CodingCascade`, `CodingWorkflow`, `coding_workflow_from_profile`) in `biosemalign.workflows.coding` executing `clean? -> match (-> link, optional)` with G1/G2/M4/M5/M6 guards and model call provenance.
- `BioSemAlign.run()` dispatching on `profile.workflow`, `BioSemAlign.code_artifact()`, `BioSemAlign.code_all_artifacts()`, and local non-model interactive `BioSemAlign.code_text()`.
- `--profile` / `-p` option on `biosemalign code`, plus `cleaners` and `profile` options on `code_terms(...)`.

### Changed

- Upgraded DuckDB bulk coding (`CODER_VERSION = "0.2.0"`) with a second cleaning pass for terms uncoded after the exact pass, recording `coding_pass`, `cleaned_by`, and `cleaned_query` on `codes.parquet`, `terms.parquet`, `CodingReport`, and `coding_manifest.json`.
- Made `TierConfig.link` optional (`OperatorConfig | None`) in Schema 2 profiles so coding tiers can skip annotation linking (`link: null`).

## Unreleased — WP8: model readiness

Core schema `0.4.0`. Artifacts serialized under `0.2.0` and `0.3.0` still verify unchanged.

### Added

- `uses_model: bool` across all `biosemalign.modules.base` operator protocols (`Operator`, `Cleaner`, `Matcher`, `Expander`, `Linker`, `Resolver`, `Selector`, `Check`), recorded on `OperatorProvenance.uses_model`.
- Model-contribution marks in data contracts (`Mention.retrieval_text_from_model`, `RetrievalEvidence.from_model_suggestion`, `ExpansionMember.from_model_suggestion`, alongside existing `Authority.LLM_INTERNAL_KNOWLEDGE`) and typed `ModelCallRecord` / `Provenance.model_calls`.
- Single-doorway `ModelService`, `PromptTemplate`, `load_prompt` (with domain overrides and worked-examples digest), `render_declared_context`, `render_numbered_candidates`, and LLM backend registry (`biosemalign.modules.model`).
- Externalized `choose_candidate.txt` prompt templates: core domain-neutral template in `biosemalign.modules.prompts` and drug-pack wording in `biosemalign.domains.drug.prompts`.
- `RoutingConfig.allow_model_auto_abstain` (default `False`), `ResearchRunManifest.{model_calls, model_cache_hits}`, `EvaluationReport.model_agreement` (`ModelAgreementMetrics`), and offline scripted test-kit operators (`ScriptedLLMBackend`, `ScriptedModelCleaner`, `ScriptedModelMatcher`, `ScriptedModelExpander`, `ScriptedModelLinker`, `ScriptedModelCheck`) in `biosemalign.testing`.

### Changed

- Refactored `LeashedLLMResolver` (`llm@0.3.0`, `uses_model=True`) to route through `ModelService`, load domain/core `choose_candidate.txt`, replace untyped `extensions["llm"]` with typed `ModelCallRecord`, and route model-declined terms (`candidate=0`) to `HUMAN_REVIEW` by default (M3; `test_declining_every_candidate_abstains` updated to verify both M3 default review routing and `allow_model_auto_abstain=True` opt-in).
- Updated `DeterministicResolver` (`rules@0.5.0`) to ignore `LLM_INTERNAL_KNOWLEDGE` statements and judge `EXACT` against the original mention when lexical evidence came solely from model suggestions (M4).
- Bumped `ROUTING_POLICY_VERSION` to `0.3.0` enforcing G10 (`from_model_suggestion` candidates never auto-accepted), and updated kernel/validator to enforce M4–M6 and M2 across all module stages (`clean`, `match`, `expand`, `link`, `resolve`, `check`).

## Unreleased — WP4: keep the most specific code

Core schema `0.4.0`. Artifacts serialized under `0.2.0` and `0.3.0` still verify unchanged.

### Added

- `LexiconStore.incoming_relationships` and `LexiconProvider.candidates_for_codes` for reverse hierarchy/refinement traversal inside vocabulary families.
- `TargetSpec.refinement` (`RefinementRule`) and RxNorm spec `1.1` rules for refining coarse `BN`/`IN`/`PIN` anchors down to `SBD`/`SCD`/`SCDC` via `strength` and `formulation`.
- `ExpansionStep`, `ExpansionMember`, `ExpansionSet`, and `RefinementOutcome` (`biosemalign.schemas.expansion`, `biosemalign.schemas.codes`), plus optional `refinement` on `KnowledgePackage`, `AlignmentProposal`, and `AlignmentDecision`.
- `RefineBySlotsExpander` (`refine_by_slots`) in `biosemalign.modules.expand` and kernel `expand` execution with post-expand G2 relationship policy enforcement.

### Changed

- Always derive `KnowledgePackage.links` (`derive_link_set` never returns `None`), including direct candidate links when concordance was not assessed; `0.4.0` artifacts require non-null `package.links`.
- Switched `DeterministicResolver` (`rules@0.4.0`), `ConcordantResolver`, and `LeashedLLMResolver` to read pre-derived `package.links` and honor `unique` / `none` / `multiple` `package.refinement` outcomes in selections, fields, and decision rationale.
- Updated `CHANNEL_PRIORITY` so `ONTOLOGY_NEIGHBORHOOD` ranks ahead of `NORMALIZED_LOOKUP`.

## Unreleased — WP3: context that is used

Core schema `0.4.0`. Artifacts serialized under `0.2.0` and `0.3.0` still verify unchanged.

### Added

- Purpose-scoped `context_view` and kernel `isolated_view` (`biosemalign.input.context`, `biosemalign.kernel`), physically isolating resolvers/selectors to declared `use` fields and excluding `neighboring_fields` and blinded fields.
- `MentionFidelity` (`verbatim`, `normalized_match`, `not_found`, `unknown`), `Mention.extractor` / `Mention.fidelity`, `canonical_preserved_attributes`, and `SourceType.EXTRACTED_TERMS` with `ExtractedTermsAdapter` and `MENTION_NOT_IN_SOURCE` check (`InputConfig.mention_not_in_source_severity`, defaulting to `warning`).
- `TaskProfile.effective_context_fields()`, `effective_mapping_key_fields()` (derived from attributes and non-empty `use`), and `effective_blinded_fields()`.
- Context & multi-table adapter authoring guide in `docs/guides/context-and-adapters.md`.

### Changed

- Removed FAERS-specific column aliases from core `biosemalign.input.context` in favor of domain-pack and profile `context.fields` declarations (`domains/drug/fields.yaml`, `faers_drug_normalization.yaml`).
- Switched `SAUBuilder`, `assess_readiness`, `DeterministicResolver`, `ConcordantResolver`, `LeashedLLMResolver`, `ExactMatcher`/`LexicalMatcher`/`DenseMatcher`, `PolicyCheck`, and domain semantic checks to read context exclusively via `context_view` / `canonical_preserved_attributes`.

## Unreleased — WP2: domain-free core

Core schema `0.4.0`. Artifacts serialized under `0.2.0` and `0.3.0` still verify unchanged.

### Added

- Dynamic domain/plugin loader (`biosemalign.plugins`: `BUILTIN_DOMAINS`, `load_domain`, `unload_domain`, `without_domain`, `ensure_plugins_loaded`) and separated domain packs under `biosemalign.domains.drug` and `biosemalign.domains.hierarchy`.
- `KnowledgeStatement.object_class` as the typed replacement for `extensions["target_tty"]`, keeping `extensions["target_tty"]` fallback when verifying legacy artifacts.
- `TargetSpec` support for identity levels (`via: []` / `level=None`), multi-hop `max_hops`, `subsumption` predicates, and the `ancestor` (`SPECIFICITY`) semantic-loss rule.
- Provider registry (`biosemalign.knowledge.providers.registry`), lexicon-tool registry, spec/profile-package/domain-field registries, and routing hooks (`register_difficult_pattern_hook`, `register_routing_rule_hook`).

### Changed

- Moved RxNav/drug specs, profiles, `ExtractStrengthCleaner` (`drug.extract_strength`), `RxNavMatcher` (`drug.rxnav`), `ValidateDrugSemantics` (`drug.semantics`), salt routing hooks, `derive_links`, and `import_omop` from core modules into `biosemalign.domains.drug`.
- Made core defaults domain-free (`TierConfig.match` defaults to `["exact"]`, `TaskProfile.checks` defaults to `[]`, `DeterministicResolver` bumped to `0.3.0` emitting `rules@0.3.0`, `KnowledgeRouter` uses the provider registry and `memo_pins`), with core imports of `domains/` asserted empty in `tests/unit/test_layers.py`.

## Unreleased — WP1: one vocabulary (profile schema 2)

Core schema `0.3.0`. Artifacts serialized under `0.2.0` and `0.3.0` still verify unchanged.

### Changed

- Unified the module and stage vocabularies to `clean`, `match`, `expand`, `link`, `resolve`,
  `select`, and `check` across `biosemalign.modules`, `biosemalign.kernel`,
  `biosemalign.workflows`, `biosemalign.api`, and provenance `OperatorProvenance.kind`.
  Removed legacy `ops/`, `adjudication/`, `coding/`, `linking/`, `resolution/`, and `pipelines/`
  packages and their aliases.
- Replaced `knowledge_package`, `reciprocal_rank_fusion`, and `cross_vocab_concordance` /
  `concordance` linkers with one `standard` linker (`merge: priority | rrf`, `k`, `concordance`,
  `equivalence_keys`, `conflict_vocabularies`), and replaced `concordant_exact` with one
  `concordant` resolver using target-neutral pass reasons (`no_target_link`, `non_target_code`).
- Moved `pass_on` and `pass_on_review_request` escalation from resolvers to `Tier` / `TierConfig`.
- Introduced profile Schema 2 (`profile_schema: 2`) with `domain`, `workflow`, `target`, `context`,
  `limits`, `tiers`, and `checks`; updated the packaged `faers_drug_normalization` profile to
  `0.2.0`. Schema 1 profiles convert in memory with a `DeprecationWarning` and via
  `biosemalign profiles migrate`.

## Unreleased — v0.2 development

Core schema `0.3.0`. Records written under `0.2.0` still validate; new fields are optional.

### Added

- Composable workflow operators (`biosemalign.ops`): transform, retrieve, fuse, propose, and
  validate protocols, a named-operator registry, and built-ins wrapping the v0.1 stages
  (`rxnav`, `knowledge_package`, `adjudicator`, `drug_semantics`) plus the `unicode_normalize`
  and `extract_strength` transforms and an `AdapterTransformOp` for external cleaners.
- Workflow combinators (`Tier`, `Cascade`, `ParallelUnion`) with startup verification, and an
  optional `workflow` section in task profiles. Omitting it runs the single-tier v0.1 pipeline.
- The alignment kernel: the only path that validates, routes, and mints decisions. It re-filters
  prohibited relationships after any fuser, isolates proposers from canonical inputs, withholds
  `knowledge.blinded_context_fields` from proposers, and rejects transforms that rewrite source
  text or asserted attributes. An exhausted cascade abstains and routes to human review.
- Decision provenance records the workflow (`workflow`, with a stable `workflow_id`) and the
  per-decision `resolution` trace (`resolved_by_tier`, pass-through reasons, withheld context).
  Artifact validation rejects traces inconsistent with the workflow or the proposal.
- `Mention.retrieval_text`: the query transforms produce. EXACT claims are still judged
  against the source mention.
- `validation.rules` profile setting; the terminology checks remain mandatory.
- `verify_artifact()` and the `biosemalign verify` command for consumer-side re-validation.

- Local lexicon bundles: an engine-defined contract (`concepts`, `synonyms`, `relationships`,
  `xrefs`), a strict builder producing a read-only DuckDB index pinned by SHA-256, and an OMOP
  importer for Athena downloads (`biosemalign lexicon build | info | import-omop`). Exact,
  word-BM25, and character-trigram retrieval; extensions are never installed or loaded.
- `LexiconProvider` implements the provider protocol, so validation and enrichment work unchanged
  whether RxNav or a bundle serves a vocabulary. A bundle's vocabularies take precedence over
  remote providers; the `LocalLexicon` registry entry permits restricted enclaves.
  `BIOSEMALIGN_LEXICON_PATH` and `BIOSEMALIGN_LEXICON_SHA256` (required in FROZEN mode) configure
  it, and resource provenance records the bundle digest as `content_sha256`.
- Cross-vocabulary concordance: the `lexicon` retriever, `cross_vocab_concordance` fuser, and
  `concordant_exact` proposer. Equivalence uses permitted keys only; a group with two codes from
  one vocabulary is a conflict; agreement counts independent sources. Decisions carry
  `concordance_state` and kernel-derived `matched_codes`, which artifact validation re-derives.
- Crosswalk columns `concordance_state`, `matched_codes`, and `resolved_by_tier`.
- `lexicon` extra (`duckdb>=1.1`, `pyarrow>=16.0`); DuckDB 1.5.6 is pinned in `constraints.txt`
  because 2.0.0 ships no Python 3.13 or 3.14 wheels.

- The leashed LLM adjudicator (`adjudication.backend: vllm`, or the `llm` proposer): an
  OpenAI-compatible client with JSON-schema constrained decoding whose reply can name only a
  retrieved candidate's number. The deterministic rules project the chosen identity, and
  disagreement with terminology evidence routes to review. Egress is enforced at transport, and
  responses are cached under the pinned model revision for exact FROZEN replay.
- Declarative policy packs (`validation.rules: [{op: policy, ...}]`): distinction-preservation,
  context-compatibility, and category-tag rules, validated on load and versioned by digest.
- Dense retrieval: `biosemalign lexicon embed`, embedding indexes bound to bundle and model digests,
  the `dense` retriever, a dependency-free hashing baseline encoder, and the
  `reciprocal_rank_fusion` fuser.
- Evaluation by resolving tier, cumulative coverage and precision by tier prefix, and per-stratum
  metrics with a disparity summary (`GoldMapping.stratum`) for differential-resolution audits.
- `biosemalign calibrate` and `routing.calibration`: Learn-then-Test thresholds with exact binomial
  tests and a Bonferroni correction, per tier where there are enough cases, gating automatic
  acceptance with a finite-sample error guarantee. `evaluate --calibration` refuses overlapping
  cases.
- The persistent crosswalk memo (`align-batch --memo`, `BioSemAlign.attach_memo`): exact reuse of
  proposals and evidence across runs, with validation and routing re-run and `provenance.memo`
  recording the run that first resolved the input.
- Run manifests record the workflow, per-tier attempts, resolutions, and time, and memo hits.

- Bulk coding (`biosemalign code`, `biosemalign.coding`): every exact dictionary code for every
  term, in one set-based pass over distinct normalized terms. Every code is kept — all
  vocabularies, linked to the target or not — with matched names, sources, and release; uncoded
  terms are listed; each code is annotated with the target ingredients it reaches. Nothing is
  decided.
- Link derivation (`biosemalign lexicon link`): missing dictionary links written as build
  overlays beside the dictionary — `has_ingredient` for precise ingredients, multiple-ingredient
  concepts, and brand names (when all a brand's products agree); keys from FDA UNII Records,
  MeSH registry numbers, OHDSI mappings of SNOMED and ATC, and UMLS CUIs; and product listings
  linked only when name and composition agree. Every row names its `assertion_source`, and
  `link_report.md` explains, with examples, everything left unlinked.
- `biosemalign lexicon audit`: the share of each vocabulary and RxNorm-family class that reaches
  the target's ingredients, and the full code-to-ingredient crosswalk.
- `lexicon build` takes several `--source` directories (a dictionary plus overlays) and
  `--no-trigram`. Relationships may carry `assertion_source`, surfaced on statements.
- `relationship_families` in lexicon declarations (and manifests): vocabularies whose edges may
  cross, such as RxNorm Extension products whose ingredients are RxNorm concepts. Statements name
  the object's vocabulary (`object_vocabulary`) when it differs from the subject's.
- `task.source_terminologies`: vocabularies besides the target a source-level concept may come
  from. Analysis-level concepts stay in the target; validation, routing, and artifact checks
  look a source concept up in its own permitted vocabulary.
- `cross_vocab_concordance` option `conflict_vocabularies`: only duplicates among these (one
  pool) make a group a conflict.
- Resolution traces keep each tier's `concordance_state` and every exact match
  (`exact_matches`), so codes survive when a tier passes; artifact validation checks them
  against the package, and the crosswalk gains `exact_match_tier`, `exact_match_state`, and
  `exact_matches`.
- Target terminology specs (`biosemalign.targets`, v0.3 P1): the granularity model the drug
  rules hard-coded — which classes sit at which level, the relationship that reaches it, which
  edges are release-bound, class aliases (OMOP spellings), salt classes, and the semantic-loss
  rules — is data in `targets/rxnorm.yaml`. The adjudicator, the validator, enrichment, routing,
  the evidence index, the crosswalk, coding, audit, link building, and the OMOP importer all read
  it. Loss rules are a closed set of rule types with one implementation, so the losses a mapping
  reports and the losses validation expects can no longer drift apart. A profile names its spec
  with `task.target_spec` (a built-in id or a YAML path), or with the new `target:` section
  (`spec`, `terminology`, `granularity`, `source_terminologies`); by default the built-in spec of
  the target terminology is used. `code` and `lexicon audit` take a `granularity`.
- Coding → linking → resolution modules (v0.3 P2): `biosemalign.coding` (`Coder`, `CodeSet`, and the
  bulk `code_terms`), `biosemalign.linking` (`Linker`, `LinkSet`, `derive_link_set`, `targets_at`),
  and `biosemalign.resolution` (`Resolver`, `PassToNext`) name the retrieve, fuse, and propose
  stages. Operators register under `code`, `link`, and `resolve` as well as the stage kinds, and one
  implementing only the module method (`code`, `link`, `resolve`) is adapted; built-ins implement
  both. Tiers accept `coding:`, `linking:`, and `resolution:` keys (stored as `retrievers`,
  `fuser`, `proposer`). New operator names: `exact` (coder), `concordance` (linker), and
  `concordant` (resolver, with target-neutral pass reasons `no_target_link` and
  `non_target_code`).
- Typed links: packages carry a `LinkSet` — each exact match's links to target codes at the
  requested granularity, as `EQUIVALENCE` or `PROJECTION`, with predicates, assertion sources,
  and ambiguous codes listed. The kernel derives it after its leakage filter; artifact
  verification derives it again under a built-in spec, or checks each link against the package's
  evidence under a custom one.
- Workflow provenance records the target spec (`target_spec`, `target_spec_sha256`), and the spec
  is part of `workflow_id`.
- Bulk DuckDB work runs under a memory budget (half of physical memory, or
  `BIOSEMALIGN_DUCKDB_MEMORY_LIMIT`) and spills to disk beyond it.

### Fixed

- Abstentions now carry the SAU's preserved attributes instead of being rejected as
  `PRESERVED_ATTRIBUTES_MISMATCH` (deterministic adjudicator `0.2.0`).
- An abstention that requests review, or carries validator warnings, now routes to
  `HUMAN_REVIEW` instead of failing artifact validation (routing policy `0.2.0`).

### Changed
- `build_adjudicator` takes the pipeline's settings and cache; `vllm` builds the leashed LLM
  adjudicator instead of raising. The `llm` extra no longer installs the `openai` package.
- `knowledge.secondary_resources` is accepted. Startup verification fails if a declared resource
  is queried by no workflow retriever, so its evidence cannot be silently omitted.
- Candidates are identified by vocabulary and code throughout fusion, and target-terminology
  candidates rank first. The deterministic adjudicator takes its identity from the target
  terminology only, and exposes `project()` so another proposer can supply the identity.
- Routing sends a mention that exactly names more than one target concept to human review instead
  of accepting whichever ranked first.
- `crosswalk.vocabulary_release` reports the release of the decided concept's vocabulary.
- Class aliases apply everywhere the spec is read: an OMOP-spelled class (`Quant Clinical Drug`)
  now gets the same projection and loss rules as its RxNorm term type (`SCD`).
- RxNorm Extension edges count as release-bound terminology evidence, like RxNorm's: a target
  reached through one is inferred active and of the spec's class.
- The combination-integrity check walks the requested granularity's relationship instead of
  always `has_ingredient`; an invalid regular expression in a spec is a validation error.
- `concordant_exact` (`0.3.0`) reads the kernel's links; the deterministic adjudicator, the
  resolver, and link derivation share one definition of "what a concept reaches at a level"
  (`linking.targets_at`). `biosemalign.coding` is now a package (`coding.bulk`); its imports are
  unchanged.
- `concordant_exact` (`0.2.0`) passes to the next tier — `no_ingredient_link`, naming the
  identity — when the agreed identity cannot reach the requested granularity in the target
  terminology, instead of resolving to a review-only abstention; `non_target_ingredient` when it
  reaches only another vocabulary's ingredients. It may take its identity from
  `task.source_terminologies`. `cross_vocab_concordance` is `0.2.0`.

- `ValidationResult.rules_version` and provenance now name every validation operator, e.g.
  `0.1.0+drug_semantics@0.1.0`.
- Artifacts written under schema `0.2.0` keep verifying: their content hash is recomputed
  without the fields `0.3.0` added. Interrupted `align-batch` runs cannot be resumed across
  versions; the checkpoint's code and profile fingerprints refuse it.
- CI skips the test steps, with a notice, when the test suite is not part of the checkout; an
  opt-in pre-push hook runs the coverage gate locally.
