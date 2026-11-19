# BioSemAlign [![DOI](https://zenodo.org/badge/1347719904.svg)](https://doi.org/10.5281/zenodo.23045154)

*A context-aware, knowledge-guided framework for biomedical mapping, coding, and translation.*

BioSemAlign converts heterogeneous structured and unstructured biomedical content (biomedical concepts, e.g., drug names, disease names) inputs into
contextualized **Semantic Alignment Units**. For each unit it retrieves candidate concepts from
terminology systems and assembles a task-specific knowledge package — definitions, terminology constraints, ontology structure, UMLS mappings, relationships from existing knowledge graphs, and coding
policies. A candidate-constrained adjudicator then chooses among the retrieved concepts,
classifies the semantic relationship between source and target, identifies unsupported
specificity or information loss, and abstains when no mapping is justified. Deterministic
terminology validation, calibrated routing, provenance tracking, and human review produce
auditable, reproducible outputs.


---

## Install

```bash
pip install -e ".[dev,data]"     # development
pip install biosemalign          # runtime only (once published)
pip install "biosemalign[lexicon]"  # local lexicon bundles (DuckDB)
pip install "biosemalign[embeddings]"  # dense retrieval with a sentence-transformers model
```

Requires Python 3.12+. No terminology content is bundled; RxNorm is queried through the public
[RxNav API](https://rxnav.nlm.nih.gov/), which needs no licence, or served from a local lexicon
bundle you compile from vocabularies you are licensed to use.

## Quick start

Live lexical lookup is an explicit privacy decision. For a local public-RxNav run, authorize the
deployment and the sensitive-text egress class; production deployments should set these through
their normal secret/configuration mechanism:

```bash
export BIOSEMALIGN_MODE=CACHE_PREFER
export BIOSEMALIGN_DEPLOYMENT_ENVIRONMENT=local
export BIOSEMALIGN_PERMITTED_EGRESS_CLASSES='["public_terminology", "sensitive_text"]'
```

```python
from biosemalign import BioSemAlign

with BioSemAlign.from_profile("faers_drug_normalization") as pipeline:
    artifact = pipeline.align_artifact(
        pipeline.make_request("TOPROL XL 50 MG", route="oral")
    )

decision = artifact.decision

print(decision.analysis_level_concept.label)          # metoprolol
print(decision.relationship_to_source)                # TARGET_BROADER_THAN_SOURCE
print([loss.kind for loss in decision.semantic_loss]) # strength, dose_form, salt_form, brand_identity
print(decision.routing_status)                        # AUTO_ACCEPT
```

The mapping is auto-accepted *because* everything the ingredient concept discards is reported.
A broader mapping that claims no loss is routed to review instead.

### Command line

```bash
# One value, with the reasoning at every stage
biosemalign inspect -p faers_drug_normalization -t "TOPROL XL 50 MG" \
  --show-knowledge --show-validation

# A batch: writes a flat crosswalk plus the full nested records alongside it
biosemalign align-batch -p faers_drug_normalization \
  -i faers_unique_products.csv -o faers_rxnorm_crosswalk.parquet \
  --value-column prod_ai --id-columns primaryid --id-columns drug_seq \
  --context-columns route --context-columns role_cod --checkpoint-every 100

# Continue a matching interrupted run without reprocessing durable rows
biosemalign align-batch -p faers_drug_normalization \
  -i faers_unique_products.csv -o faers_rxnorm_crosswalk.parquet \
  --value-column prod_ai --resume

biosemalign evaluate --gold gold.jsonl --predictions decisions.jsonl
biosemalign profiles
biosemalign cache-info
```

Evaluation gold rows use a stable `case_id`, the original `source_value`, the
`expected_concept_ids`, optional source `context`, and an optional `category`. Predictions are
joined by `extensions.case_id` when supplied, then by batch `record_id`, then by deterministic
`sau_id`; source text is never treated as a unique key. The report includes one-to-one gold and
prediction counts plus category-level metrics. Missing, unexpected, duplicated, or source-mismatched
cases make the command fail instead of silently shrinking the evaluation denominator.
When a gold row has nonempty `context`, predictions must be complete alignment artifacts: evaluation
checks the gold fields against the artifact's SAU context and retains its `mapping_key` in the
report. Legacy decision-only JSONL is accepted only when the corresponding gold context is empty,
because a compact decision does not contain enough evidence to verify context.

Component-level retrieval evaluation reports Recall@1/5/10 only over gold cases with one or more
expected concepts. Gold cases with an empty `expected_concept_ids` list are reported separately as
negative controls, including candidate-rejection specificity (the fraction for which retrieval
returned no candidates), so negatives cannot appear as unavoidable recall misses.

### Workflows and verification (v0.2, in development)

A profile may declare a cascade of tiers. Each tier transforms the SAU, retrieves, fuses, and
proposes; the first tier to propose hands off to the kernel, and a tier whose proposer passes
moves the SAU on to the next. Omitting `workflow` runs the single-tier v0.1 pipeline.

```yaml
workflow:
  tiers:
    - name: exact
      retrievers: [{op: rxnav, options: {channels: [normalized_lookup]}}]
      proposer: {op: adjudicator, options: {pass_on: [NO_VALID_MAPPING]}}
    - name: full            # defaults: rxnav, knowledge_package, adjudicator
      transforms: [unicode_normalize, extract_strength]
```

Every decision records the workflow it ran under and a resolution trace (`resolved_by_tier`, plus
the reason each earlier tier passed). Whatever the workflow, the kernel re-filters prohibited
relationships after fusion, hides `knowledge.blinded_context_fields` from proposers, rejects
transforms that rewrite source text or asserted attributes, and runs the mandatory validation
and routing. Operators can propose but never mint decisions. In Python, pass
`BioSemAlign(profile, workflow=builder)`, where `builder` receives an `OpContext` and returns a
`Cascade`; custom operators register with `biosemalign.ops.register_op`.

Downstream consumers can re-check persisted artifacts instead of trusting where they came from:

```bash
biosemalign verify -i faers_rxnorm_crosswalk.jsonl -p faers_drug_normalization
```

```python
from biosemalign import verify_artifact
assert verify_artifact(artifact, profile=profile).valid
```


### Local lexicon bundles and concordance (v0.2, in development)

Vocabularies you are licensed to use can be compiled into a read-only bundle and served with no
network access: the way to run inside a restricted enclave, and to pin every vocabulary release a
study used.

```bash
# From an Athena download (RxNorm, RxNorm Extension, ATC, NDC, ...)
biosemalign lexicon import-omop --athena athena/ --output tables/ \
  --vocabulary RxNorm --vocabulary "RxNorm Extension" --license public
# or from your own concepts, synonyms, relationships, and xrefs tables (Parquet, CSV, or TSV)
biosemalign lexicon build --source tables/ --declaration tables/vocabularies.yaml --output bundle/
biosemalign lexicon info bundle/
export BIOSEMALIGN_LEXICON_PATH=bundle/ BIOSEMALIGN_LEXICON_SHA256=<digest printed by build>
```

The contract is documented in `biosemalign.knowledge.lexicon.contract`. A bundle is verified by
SHA-256 when opened, and a vocabulary it serves takes precedence over any remote provider of the
same name. With RxNorm in the bundle, the default workflow uses the `lexicon` retriever (exact
names, then character trigrams, which tolerate misspellings) instead of RxNav.

Cross-vocabulary concordance turns agreement between dictionaries into evidence:

```yaml
knowledge:
  secondary_resources: [UNII, MeSH, UMLS]
workflow:
  tiers:
    - name: concordant
      retrievers: [{op: lexicon, options: {vocabularies: [RxNorm, UNII, MeSH, UMLS], methods: [exact]}}]
      fuser: cross_vocab_concordance
      proposer: {op: concordant_exact, options: {authority_sources: [RxNorm]}}
    - name: fallback
      retrievers: [{op: lexicon, options: {vocabularies: [RxNorm]}}]
```

Exact matches are grouped only through permitted equivalence keys: by default, cross-references to
another vocabulary in the bundle. A group holding two codes of one vocabulary, such as a salt and
its base merged by an over-broad key, is a conflict rather than agreement. Agreement counts
independent sources, not vocabularies, so a name UMLS redistributes from RxNorm counts once. Name
collisions, conflicts, mentions shorter than five characters, and names asserted by a single
non-authority source pass to the next tier. Decisions report every equivalent code
(`matched_codes`) and the `concordance_state`, and the crosswalk carries both, plus
`resolved_by_tier`. A mention that exactly names more than one target concept is never accepted
automatically.

Measured on a 2-CPU, 7 GB machine with a synthetic bundle of 300,000 concepts and 900,000 names:
the build took 57 s and produced a 256 MB index; opening and verifying took 0.3 s. Per query:
exact lookup 1.5 ms, candidate construction 9 ms (0.8 ms once memoized), word BM25 9 ms, and
trigram retrieval about 110 ms, which ranked the original concept first for 79 of 100
transposition typos.


### LLM adjudication, policy packs, and dense retrieval (v0.2, in development)

**Leashed LLM.** Set `adjudication.backend: vllm`, or use the `llm` proposer in one tier, to have a
model choose among retrieved candidates. It talks to any OpenAI-compatible endpoint (a vLLM server
inside an enclave, or a hosted API) with JSON-schema constrained decoding. The reply schema admits
only a candidate number `0..K` (`K` is at most 15 by default), so it cannot name an identifier; the
deterministic rules then project the chosen concept and derive relationship and semantic loss. A
model whose declared relationship or loss disagrees with that evidence, or that asks for review,
sends the decision to review. Prompts carry source text: hosts other than `llm_local_hosts` need
the `sensitive_text` egress class and HTTPS. Responses are cached under the full request and the
pinned `BIOSEMALIGN_LLM_MODEL_REVISION`, so FROZEN runs replay them exactly and fail closed on a
miss. Blinded context fields never reach the prompt.

```yaml
workflow:
  tiers:
    - name: rules
      proposer: {op: adjudicator, options: {pass_on_review_request: true}}
    - name: model
      retrievers: [{op: lexicon}, {op: dense, options: {index: sapbert}}]
      fuser: reciprocal_rank_fusion
      proposer: llm
```

**Policy packs.** Clinical guidance becomes a validation operator without engine code:
`validation.rules: [drug_semantics, {op: policy, options: {path: perinatal.yaml}}]`. A pack holds
three rule types, all validated on load: `preserve_distinction_pairs` (a source whose rollup loses a
material distinction, e.g. caffeine citrate), `context_compatibility_rules` (context contradicting a
formulation, e.g. intramuscular against a topical cream), and `non_target_category_tags` (controls
and drug classes). Findings are warnings, which route to review, unless a rule raises an error. The
pack's digest is part of the validation rules version recorded on every decision.

**Dense retrieval.** `biosemalign lexicon embed bundle/ --name sapbert --encoder
sentence_transformers --model /models/sapbert` encodes every name of a bundle into an embedding index
bound to the bundle digest and the model's weights digest; models load from local directories only.
The `dense` retriever searches it by exact cosine similarity inside DuckDB, and
`reciprocal_rank_fusion` merges it with lexical retrieval. A built-in `hashing` encoder (character
trigrams, not semantic) exercises the path without a model.

### Evaluation, calibration, and the crosswalk memo (v0.2, in development)

`biosemalign evaluate` now also reports metrics per resolving tier and cumulative coverage and
precision by tier prefix. These equal what a cascade truncated after each tier would accept, because
a later tier never changes an earlier acceptance. Gold cases may carry a `stratum`, such as whether
the report carries the studied outcome; the report gives each stratum's metrics and the largest gap
between them, an audit of differential misclassification.

```bash
# Fit thresholds on held-out labelled cases from an uncalibrated run ...
biosemalign calibrate --gold calibration_gold.jsonl --predictions calibration_run.jsonl \
  --alpha 0.05 --delta 0.1 -o calibration.json
# ... and evaluate only on cases it was not fitted on
biosemalign evaluate --gold test_gold.jsonl --predictions test_run.jsonl --calibration calibration.json
```

With `routing.calibration: calibration.json`, a decision the rules would accept automatically is
accepted only if its acceptance score clears its tier's certified threshold; with probability at
least `1 - delta`, the error rate among automatic acceptances is then at most `alpha`. Thresholds
are fitted by Learn-then-Test with exact binomial tests and a Bonferroni correction, per tier when a
tier has enough cases. The output reports the achieved bound and the counts behind it. Small
calibration sets often certify nothing, and then every eligible decision goes to review rather than
being accepted on an optimistic threshold. The guarantee assumes calibration and deployment cases
are exchangeable.

`align-batch --memo memo.sqlite` resolves each distinct input once across runs. A memo entry is
keyed on everything the proposal depended on: the mention, context, attributes, workflow digest, and
pinned resource identities. It requires a pinned RxNorm release when RxNav is used. Validation and
routing re-run on every reuse, and provenance names the run that first resolved the input. The run
manifest records memo hits and per-tier attempts, resolutions, and time; times stay out of artifacts
so that artifacts reproduce.

----

### Environment

| Variable | Purpose |
|---|---|
| `BIOSEMALIGN_MODE` | `ONLINE` / `CACHE_PREFER` / `FROZEN` (default) / legacy `OFFLINE` |
| `BIOSEMALIGN_CACHE_DIR` | Provider response cache |
| `BIOSEMALIGN_DEPLOYMENT_ENVIRONMENT` | Required live-mode authorization: `local`, `institutional_server`, `hpc`, or `restricted_enclave` |
| `BIOSEMALIGN_PERMITTED_EGRESS_CLASSES` | JSON list of permitted classes; defaults to public terminology only |
| `BIOSEMALIGN_RXNAV_TERMINOLOGY_RELEASE` | Required release pin for `FROZEN` RxNorm snapshots |
| `BIOSEMALIGN_IDENTIFIER_HMAC_KEY` | Optional 32-byte-or-longer secret for deployment-scoped SAU and mapping identifiers |
| `BIOSEMALIGN_LLM_BASE_URL`, `_API_KEY`, `_MODEL` | OpenAI-compatible model endpoint, e.g. a local vLLM server's `http://localhost:8000/v1` |
| `BIOSEMALIGN_LLM_MODEL_REVISION` | Pinned model identity; required in FROZEN mode, where responses are replayed |
| `BIOSEMALIGN_LLM_LOCAL_HOSTS` | JSON list of hosts inside the deployment boundary (default loopback); others need `sensitive_text` egress |
| `BIOSEMALIGN_EMBEDDING_MODEL_PATH` | Local sentence-transformers model directory for the `dense` retriever |
| `BIOSEMALIGN_UMLS_API_KEY` | UMLS licence key (v0.2) |
| `BIOSEMALIGN_LEXICON_PATH` | Local lexicon bundle directory; its vocabularies take precedence over remote providers |
| `BIOSEMALIGN_LEXICON_SHA256` | Pinned bundle digest (printed by `lexicon build`); required in FROZEN mode |

Credentials come from the environment only, never from profile YAML — profiles are committed and
shared between projects.


## Development baseline (M0)

The current cleaner is wrapped by
`biosemalign.input.adapters.ExistingCleanerAdapter`. Its `clean_records()`
method streams the existing `BatchAlignmentRecord` results, including skipped
and failed rows. The caller configures and closes the underlying `BioSemAlign`
instance.

The offline regression suite uses invented terminology responses and blocks
external sockets. Its expected mappings record existing behavior; they are not
clinically validated term lists. See [the M0 inventory](docs/inventory.md),
[policy decisions](docs/decisions.md), and [baseline reproduction](docs/baseline.md).

```bash
python -m biosemalign.schemas.export --check
pytest --cov=biosemalign --cov-report=term-missing
python -m examples.offline_baseline
```

Live RxNorm verification is pending until separate, independently verified public
benchmark files are available. The manually dispatched canary skips its probe
when those files are absent and requires an explicit expected release. Never use
synthetic fixture identifiers as real RxNorm codes.

## Layout

```text
src/biosemalign/
├── schemas/      Core data contracts (§11); JSON Schema committed to docs/schemas/
├── input/        Module 1 — adapters, context, readiness, SAU builder
├── knowledge/    Module 2 — providers, retrieval, fusion, enrichment, packaging
├── adjudication/ Module 3 — deterministic adjudicator, backend protocols, mocks
├── validation/   Module 4 — mandatory terminology checks; pluggable rule operators
├── routing/      Module 5 — features and routing policy
├── ops/          Workflow operator protocols, registry, and built-in operators
├── pipelines/    Public facade, workflow combinators, and the alignment kernel
├── profiles/     Task profiles and loader
├── resources/    Knowledge Resource Registry (licences, environments)
├── storage/      Cache, JSONL, Parquet/CSV crosswalk
├── evaluation/   Recall@k, one-to-one selective accuracy, category reports
└── verification.py  Consumer-side artifact re-validation
```

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

BioSemAlign bundles no terminology content. RxNorm is non-proprietary and needs no separate
licence; UMLS, MedDRA, and SNOMED CT require agreements obtained from their owners. See
[the resource registry](src/biosemalign/resources/registry.yaml) for what each resource requires
and where it may be used.
