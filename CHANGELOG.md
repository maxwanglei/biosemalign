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
