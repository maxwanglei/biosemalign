# Changelog

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

- `ValidationResult.rules_version` and provenance now name every validation operator, e.g.
  `0.1.0+drug_semantics@0.1.0`.
- Artifacts written under schema `0.2.0` keep verifying: their content hash is recomputed
  without the fields `0.3.0` added. Interrupted `align-batch` runs cannot be resumed across
  versions; the checkpoint's code and profile fingerprints refuse it.
- CI skips the test steps, with a notice, when the test suite is not part of the checkout; an
  opt-in pre-push hook runs the coverage gate locally.
