# Evidence Attribution in Graph-Augmented and Multimodal RAG

Controlled document question-answering experiments that separate retrieval
failure, incomplete evidence, evidence-use failure, non-gold support,
closed-book answers, and unsupported answers when a RAG pipeline is augmented
with a knowledge graph (at the generation stage, `+KG`, or the retrieval
stage, `+KGret`) or with retrieved figure crops (`+multimodal`, `+both`).

Everything runs through one installed command, `rag`, from one package,
`multimodal_graph_rag`. There is a single implementation of every provider
call, retriever, graph, judge, and artifact builder; failures stop a run
explicitly instead of being scored as zero.

## Repository layout

```text
src/multimodal_graph_rag/
  cli.py                 the `rag` command
  clients.py             provider registry, frozen pricing, ModelClient (all model calls)
  config.py              ExperimentConfig with a deterministic identity hash
  schemas.py             QuestionRecord, EvidenceBundle, RunRecord, question file I/O
  errors.py              explicit error types (provider, pricing, cache, evidence)
  corpora/               loaders, captioning/OCR, and the five ingesters
  questions/             validators and the authoring protocols
  retrieval/             dense TextIndex, KnowledgeGraph, graph expansion,
                         CLIP image index, lexical/random controls, metrics
  pipelines/             systems under comparison, evidence-control arms,
                         and the evaluation runner
  evaluation/            judges, attribution states, leakage screening,
                         audit instruments, paired inference
  artifacts/             manifests, run selection, tables, figures
configs/                 experiment configs and the frozen pricing snapshot
data/questions/          released question sets
data/graphs/             extracted-triple caches (one per corpus)
artifacts/runs/legacy/   the run files behind the first submission
artifacts/{tables,figures}/legacy/   tables and figures as submitted
protocols/               human-annotation instruments
docs/                    architecture diagram
archive/                 invalid runs, superseded scripts, logs, old submissions
tests/                   offline test suite with a deterministic fake client
```

## Installation

The environment is conda-based (FAISS and Tesseract come from conda-forge;
the pip-installed package pins the versions used for the released runs).

```powershell
conda env create -f environment.yml
conda activate rag
Copy-Item .env.example .env      # add only the keys a run needs
rag --help
```

Provider keys are read from the environment (and `.env`) when a command that
needs them starts. `TESSERACT_CMD` points at the Tesseract binary if it is not
on `PATH`. No key is required to run the tests or a dry run.

## Workflow

Every experiment is described by a JSON config (`configs/*.json`) naming the
corpus, question set, systems, generators, candidate budget, bridge depth,
vision mode, evidence controls, and the pricing snapshot. The config hash and
the question-file checksum are written into every summary row.

```powershell
# 1. Build a corpus from its public dataset
rag ingest --config configs/spiqa_cross_paper.json --papers 100

# 2. Author question sets (each protocol is one command; all items need manual review)
rag questions --protocol text  --config configs/publaynet_figures.json --corpus-dir publaynet_corpus --out data/questions/questions_publaynet_text.json --target 35
rag questions --protocol figure --config configs/publaynet_figures.json --image-dir publaynet_images --out data/questions/questions_publaynet_figures.json --target 35
rag questions --protocol caption-matched --config configs/spiqa_cross_paper.json --pixel-file data/questions/questions_spiqa_figures.json --image-dir spiqa_images --out data/questions/questions_spiqa_figures_caption.json
rag questions --protocol cross-paper --config configs/spiqa_cross_paper.json --corpus-dir spiqa_corpus --out data/questions/questions_spiqa_multihop_cross.json --target 50

# 3. Inspect the plan without spending credit, then run
rag evaluate --config configs/primary_revision.json --dry-run
rag evaluate --config configs/primary_revision.json --limit 5      # pilot
rag evaluate --config configs/primary_revision.json

# 4. Retrieval-only A/B for graph expansion (pre-registered primary outcome)
rag retrieval-ab --config configs/primary_revision.json

# 5. Freeze the runs a reported table may use, validate, and build tables and figures
rag freeze-manifest --summary artifacts/runs/summary_*.csv --release-id r1 --question-set data/questions/questions_hotpotqa_bridge.json --output configs/release_manifest.json
rag validate-release --manifest configs/release_manifest.json
rag artifacts --manifest configs/release_manifest.json --out artifacts

# Exploratory tables from a directory scan (never a release artifact)
rag artifacts --scan artifacts/runs/legacy --out artifacts/exploratory
```

Each evaluation writes three timestamped files that are never overwritten:
`summary_*.csv` (one row per system), `detail_*.csv` (one row per question),
and `trace_*.jsonl` (one `RunRecord` per question and system with the full
evidence bundle: ranked passages, graph-bridged passages, graph facts, scored
and sent images, and the raw judge calls). A provider failure aborts the run
after writing the completed rows as `detail_partial_*.csv`.

### Systems and evidence controls

| system                 | context supplied to the generator                                  |
|------------------------|--------------------------------------------------------------------|
| `baseline`             | dense top-`k` passages                                             |
| `+KG`                  | baseline passages plus graph facts restricted to those passages    |
| `+KGret`               | dense top-3 plus graph-expanded passages, same total budget        |
| `+multimodal`          | baseline passages plus a CLIP-retrieved crop (pixels or caption)   |
| `+both`                | `+KG` and `+multimodal` together                                   |
| `control:closed-book`  | no context                                                         |
| `control:shuffled`     | unrelated passages of matched length                               |
| `control:oracle`       | passages from every gold document                                  |
| `control:partial-gold` | passages from the first gold document only                         |

Controls are switched on per config through the `controls` list.

### Audits and statistics

```powershell
rag audit-graph data/graphs/triples_cache_spiqa_corpus.json --corpus spiqa --count 200
rag audit-figures data/questions/questions_spiqa_figures.json spiqa_images/captions.json spiqa_corpus
rag paired-analysis artifacts/runs/detail_<run>.csv --treatment +KGret
rag clean-questions data/questions/questions_spiqa_text.json
```

The leakage screen labels a clean programmatic result `screened_unflagged`,
never "verified"; semantic checks and human review follow the protocol in
`protocols/ANNOTATION_GUIDE.md`. Paired contrasts report bootstrap confidence
intervals, exact McNemar tests, and continuity-corrected odds ratios;
generator rows are not independent replications.

## Provenance rules

- Reported tables are built only from runs listed in a checksummed release
  manifest. `rag validate-release` rejects missing, modified, archived, or
  error-containing run files.
- Cells are never backfilled across timestamps; a directory scan selects one
  newest valid run per question set and generator as a unit.
- Prices are frozen in `configs/pricing_<date>.json`; a model without a
  frozen price cannot be charged and the capture date is written into every
  summary row.
- The graph-seeded SPIQA cross-paper set is labelled `graph_seeded` and must
  be reported separately from any independently authored set.

## Status of the revision

The package implements the interfaces, controls, and safeguards for the major
revision. The released runs under `artifacts/runs/legacy/` are historical
evidence for the first submission; the 100-question sets, equal-budget
retrieval comparators, established graph-retriever comparison, document
visual retriever, oracle-image control, and 400-output human validation
described in the revision plan have not been run yet and must not be reported
as complete until their frozen manifests and adjudicated labels exist.

## Development

```powershell
ruff check .
ruff format --check .
pytest
```

The test suite runs offline against a deterministic fake client and a tiny
corpus; it exercises the full evaluation loop, caches, graph expansion,
manifests, tables, and figures without provider access.

