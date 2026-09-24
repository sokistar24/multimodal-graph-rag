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

`environment.yml` is CPU-only and takes the whole numerical stack from
conda-forge so that FAISS, NumPy, Matplotlib, and PyTorch share one OpenMP
runtime. For a local judge, a local generator arm, or a document-image
retriever, use `environment-gpu.yml` instead. It is also conda-forge only and
pins `cuda-version=12.8`, which Blackwell cards (RTX 50-series, `sm_120`)
require and which the CUDA 12.4 build behind PyTorch 2.6 does not provide.

```powershell
conda env create -f environment-gpu.yml
conda activate rag-gpu
python -c "import torch; print(torch.cuda.get_device_name(0), torch.version.cuda)"
```

Provider keys are read from the environment (and `.env`) when a command that
needs them starts. The revision roster (the fifth generator and every judge)
is served through OpenRouter under one `OPENROUTER_API_KEY`; the legacy
generators keep their direct providers so results stay comparable with the
first-submission runs. Prices are frozen per model in `configs/pricing_*.json`.
`TESSERACT_CMD` points at the Tesseract binary if it is not on `PATH`. No key
is required to run the tests or a dry run.

## Published comparator (HippoRAG 2)

The `+HippoRAG` system compares our graph expansion with a published graph
retriever at the same candidate budget. HippoRAG brings its own model stack,
so it runs in a separate environment and never inside an evaluation:

```powershell
# comparators environment, once per corpus
pip install -e .[comparators]
python -m multimodal_graph_rag.retrieval.hipporag_index --corpus-dir hotpotqa_corpus --questions data/questions/questions_hotpotqa_bridge.json --output .cache/hipporag/hotpotqa/retrievals_hotpotqa_bridge.json --save-dir .cache/hipporag/hotpotqa/index

# main environment: retrieval-only check, then point a config at the file
rag comparator-ab --questions data/questions/questions_hotpotqa_bridge.json --retrievals .cache/hipporag/hotpotqa/retrievals_hotpotqa_bridge.json --corpus-dir hotpotqa_corpus --k 5
```

The indexer feeds HippoRAG the same chunks the dense baseline embeds and maps
every returned text back to its source file, so gold-provenance scoring works
unchanged. The retrievals file records the corpus fingerprint and question-set
digest it was built for and refuses a corpus that has changed. Set
`comparator_retrievals` in the config and add `+HippoRAG` to `systems`.

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

# ...or convert SPIQA's own human-curated QA, which needs no model and no keys
rag questions --protocol spiqa-native --gold-file data/questions/spiqa_gold_qa.json --image-dir spiqa_images --out data/questions/questions_spiqa_native.json --target 0
rag questions --protocol spiqa-native --gold-file data/questions/spiqa_gold_qa.json --image-dir spiqa_images --out data/questions/questions_spiqa_native_primary.json --target 150 --exclude-flagged

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

### Question sets and who wrote them

Authorship is recorded per question in `construction_method`, because it
determines what a result can be claimed to show.

| set                                           | n                   | authored by                                                                            |
|-----------------------------------------------|---------------------|----------------------------------------------------------------------------------------|
| `questions_spiqa_native`                      | 579 (447 unflagged) | **SPIQA test-A curators — human question, human answer, human crop-level provenance**  |
| `questions_hotpotqa_bridge` / `_comparison`   | 50 + 50             | **HotpotQA annotators — human questions and human supporting facts**                   |
| `questions_spiqa_multihop_cross`              | 50                  | DeepSeek, seeded from the triple store (`graph_seeded`)                                |
| PubLayNet and SPIQA text/multihop/figure sets | 26–50 each          | DeepSeek (text) or Claude Haiku (figures)                                              |

Two consequences worth keeping in view. The graph-seeded cross-document set
is an **in-graph mechanism test**, not evidence that graph retrieval helps in
general: its questions were built from entities the graph already contains.
And the HotpotQA and SPIQA-native sets are independent of this project
entirely, so they carry no such construction advantage.

`spiqa-native` conversion runs the leakage screen against both the pipeline
caption and the author caption and labels each item; it does not filter the
benchmark. It also reports how many questions name their own modality (40 of
579), which the model-authored protocols forbid — a real difference between
the sets rather than something to normalise away.

### Audits and statistics

```powershell
# Check every extracted triple against its source chunk (whole graph, no sampling)
rag graph-grounding data/graphs/triples_cache_spiqa_corpus.json --corpus-dir spiqa_corpus --corpus spiqa --collisions

rag audit-figures data/questions/questions_spiqa_figures.json spiqa_images/captions.json spiqa_corpus
rag paired-analysis artifacts/runs/detail_<run>.csv --treatment +KGret
rag clean-questions data/questions/questions_spiqa_text.json

# Draws a stratified sample for an optional human audit; nothing depends on it
rag audit-graph data/graphs/triples_cache_spiqa_corpus.json --corpus spiqa --count 200
```

The leakage screen labels a clean programmatic result `screened_unflagged`,
never "verified". `graph-grounding` reports **grounding, not precision**: it
shows a triple's endpoints occur in the passage it claims to come from, which
makes a low rate strong evidence of extraction error, but it cannot tell
whether the relation is correct. Paired contrasts report bootstrap confidence
intervals, exact McNemar tests, and continuity-corrected odds ratios;
generator rows are not independent replications.

Every run also scores answers against the reference with `em`, `f1`, and
`contains` alongside the judge verdict. Where the reference is a short span —
HotpotQA in particular — exact match and F1 are the benchmark's official
metrics and should carry the headline claim, because they involve no judge
model. `protocols/ANNOTATION_GUIDE.md` records what this does and does not
license.

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
evidence for the first submission; the equal-budget retrieval comparators, the
established graph-retriever comparison, the document-image retriever, and the
oracle-image control have not been run yet and must not be reported as
complete until their frozen manifests exist.

**No new human annotation is being collected.** The study rests on the human
judgement already embedded in the benchmarks it uses — SPIQA's curated QA and
HotpotQA's questions and supporting facts — plus reference-based scoring
(`em`, `f1`, `contains`) computed against those human-written answers, the
four control arms, and `rag graph-grounding`. The consequences, including what
therefore cannot be claimed, are set out in `protocols/ANNOTATION_GUIDE.md`
and must be carried into the limitations.

## Development

```powershell
ruff check .
ruff format --check .
pytest
```

The test suite runs offline against a deterministic fake client and a tiny
corpus; it exercises the full evaluation loop, caches, graph expansion,
manifests, tables, and figures without provider access.
