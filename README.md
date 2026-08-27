# BioSemAlign

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
```

Requires Python 3.12+. No terminology content is bundled; RxNorm is queried through the public
[RxNav API](https://rxnav.nlm.nih.gov/), which needs no licence.

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
| `BIOSEMALIGN_LLM_BASE_URL`, `_API_KEY`, `_MODEL` | vLLM endpoint (v0.2) |
| `BIOSEMALIGN_UMLS_API_KEY` | UMLS licence key (v0.2) |

Credentials come from the environment only, never from profile YAML — profiles are committed and
shared between projects.


## Layout

```text
src/biosemalign/
├── schemas/      Core data contracts (§11); JSON Schema committed to docs/schemas/
├── input/        Module 1 — adapters, context, readiness, SAU builder
├── knowledge/    Module 2 — providers, retrieval, fusion, enrichment, packaging
├── adjudication/ Module 3 — deterministic adjudicator, backend protocols, mocks
├── validation/   Module 4 — terminology and drug semantic rules
├── routing/      Module 5 — features and routing policy
├── profiles/     Task profiles and loader
├── resources/    Knowledge Resource Registry (licences, environments)
├── storage/      Cache, JSONL, Parquet/CSV crosswalk
└── evaluation/   Recall@k, one-to-one selective accuracy, category reports
```

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

BioSemAlign bundles no terminology content. RxNorm is non-proprietary and needs no separate
licence; UMLS, MedDRA, and SNOMED CT require agreements obtained from their owners. See
[the resource registry](src/biosemalign/resources/registry.yaml) for what each resource requires
and where it may be used.
