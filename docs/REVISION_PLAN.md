# Revision plan: Evidence attribution in graph-augmented and multimodal RAG

Status: current as of 2026-09-10. Supersedes the 2026-09-03 plan and the
draft reply to Sokipriala. It is the working plan for the revision; the
write-up itself is kept outside this repository.

Decision IPM-D-26-06642 (2026-09-02) was an editorial-screening rejection
asking for "at least 40% expansion" of theory and empirical analysis. Read
that as depth, not length. The compiled 33-page submission already carries a
coherent contribution: a stage-controlled comparison of prompt-side graph
injection (+KG) against retrieval-stage graph expansion (+KGret), matched
evidence-deficit controls, caption-contamination checks, and
evidence-conditioned accuracy. What it lacks is the validation that makes
those results hold up.

## 1. Standing decisions

These are settled. Do not reopen them in the revision.

- **No new human annotation.** Decided 2026-09-08. Every validation step
  that the old plan gave to humans is done by a strong hosted model, under the
  rules in section 7, and is always labelled *model-reviewed* or
  *model-judged* in the paper, never *verified* or *human-annotated*.
- **No local-GPU judge.** Judging and reviewing use large hosted models. The
  local GPU (RTX 5060 Ti, `environment-gpu.yml`) is for the document-image
  retriever and, optionally, a local generator arm.
- **Headline claims rest on human-authored benchmark questions.** SPIQA
  test-A curated QA (579 items, 447 clean of caption leakage, 150 sampled as
  the primary set) and HotpotQA bridge and comparison questions were written
  and answered by the benchmarks' annotators with human provenance. They are
  the independent sets. Model-authored sets are mechanism tests.
- **Reference-based metrics are primary where the reference is a short span.**
  HotpotQA exact match and token F1 are the benchmark's own metrics and carry
  the graph claims. The judge is secondary there and primary only on
  discursive SPIQA-native answers.
- **Journal order (agreed with Soki, 2026-09-03):** Data & Knowledge
  Engineering, then Applied Artificial Intelligence, then whatever Elsevier
  transfer suggests. Information Systems is deprioritised because the paper
  was already submitted there. "Expert Systems" means the Wiley journal, not
  ESWA, and is a fallback. Journal of Web Semantics only if APC funding is
  confirmed, since it is fully OA and outside the Coventry hybrid coverage.
- **Authorship.** No Aston or Warwick author is added for OA reasons. Soki is
  the eligible Coventry submitting author. Confirm coverage with
  oa.lib@coventry.ac.uk in writing immediately before submission.
- **Cost is not the constraint.** The 2026-09-03 estimate was about $23 for
  the whole revision with DeepSeek judging. Stronger judges raise that by a
  multiple. Add the judge and reviewer models to the pricing snapshot and
  re-estimate before the paid runs. The budget question goes to Soki with the
  API keys, since no keys exist on this machine.
- **Window.** Four to six weeks from 2026-09-03. Michael leads; Opeoluwa is
  on thesis; Abiola is available for internal review.

## 2. Why the paper was rejected

1. **Two studies, no unifying framework.** Graph-at-which-stage and
   does-multimodal-gain-survive-leakage-removal are joined only by the phrase
   "evidence attribution", which the manuscript treats as an evaluation
   principle rather than a framework with states, hypotheses, and a protocol.
2. **The best graph result is graph-seeded.** The 50 SPIQA cross-paper
   questions were generated from entities in the same graph that +KGret then
   searches, and each question names the linking entity. That is a valid
   mechanism test and an invalid basis for a general claim.
3. **Claims outrun the evidence.** Conditions of 26 to 50 questions with no
   intervals, effect sizes, paired tests, or multiplicity correction. Four
   generators on the same questions and retrievals are not four replications.
   Binary judges never validated. "Parametric floor" asserted where only
   "correct without complete gold provenance" is shown. "Verified pixel-only"
   means exact-string screened. Recall@3 shown as a ceiling where only the
   top-1 crop is supplied. A 25/34 cell in a nominal n=35 table.
4. **No proof the graph is necessary.** +KGret supplies up to two chunks
   beyond dense top-3 and is never compared with equal-budget dense, BM25,
   hybrid, lexical entity expansion, random padding, or a published graph
   method. Tens of thousands of extracted triples are never quality-checked.

## 3. Contribution and framing

Title (decided 2026-09-10): *Evidence Attribution in Graph-Augmented and
Multimodal RAG: A Controlled Evaluation for Document Question Answering*.
It keeps the "controlled evaluation" phrase from the arXiv title (2607.16604,
*When Do Multimodal and Graph-Augmented RAG Help? A Controlled Evaluation
for Document Question Answering*) so the lineage is visible, and puts the
new primary contribution first. Alternatives considered: *Where Does the
Answer Come From? Evidence Attribution for Graph-Augmented and Multimodal
RAG*; *Attributing Answers to Evidence in Graph-Augmented and Multimodal
RAG for Document QA*. The next arXiv version carries the chosen title.

Evidence attribution is the primary contribution. Graph augmentation and
multimodal augmentation are its two tests.

**Evidence states** (already implemented as `EvidenceState` in
`evaluation/attribution.py`):

| state | meaning | how it is assigned |
| --- | --- | --- |
| evidence unavailable | required source absent from the searchable corpus | corpus manifest |
| retrieval failure | source exists, not retrieved | gold provenance vs. evidence bundle |
| incomplete retrieval | part of multi-source evidence supplied | same |
| complete retrieval | every gold unit supplied | same |
| evidence-use failure | complete evidence, wrong answer | same, plus score |
| closed-book | correct with no retrieved context | `control:closed-book` arm |
| non-gold-supported | correct without full gold, supported by another passage | **model-judged sufficiency check** (7.4), reported as model-judged |
| unsupported | correct by reference match with no identifiable support | residual after the above |

**Research questions**, pre-specified and fixed before the paid runs:

- RQ1. Does graph augmentation improve retrieval and answer quality after
  controlling for candidate-set size and non-graph query expansion?
- RQ2. Is graph augmentation beneficial primarily under measurable evidence
  deficits?
- RQ3. How do caption, corpus-text, and model-prior paths change the
  estimated value of multimodal retrieval?
- RQ4. How much end-to-end failure is attributable to retrieval, evidence
  completeness, evidence use, or unsupported generation?

## 4. Question sets

| set | file | now | target | authored by | role |
| --- | --- | --- | --- | --- | --- |
| HotpotQA bridge | `questions_hotpotqa_bridge.json` | 100 (since 2026-09-10) | 100+ | HotpotQA annotators | **independent multi-document; primary for RQ1, RQ2, RQ4** |
| HotpotQA comparison | `questions_hotpotqa_comparison.json` | 100 (since 2026-09-10) | 100+ | HotpotQA annotators | independent multi-document; primary |
| SPIQA native primary | `questions_spiqa_native_primary.json` | 150 | 150 | SPIQA curators | **independent visual; primary for RQ3** |
| SPIQA native full | `questions_spiqa_native.json` | 579 | 579 (447 clean) | SPIQA curators | robustness |
| SPIQA cross-paper | `questions_spiqa_multihop_cross.json` | 50 | 50 | DeepSeek, graph-seeded | **in-graph mechanism set only** |
| SPIQA text, multihop | `questions_spiqa_text.json`, `_multihop.json` | 35, 30 | keep | DeepSeek | mechanism |
| SPIQA figures | `questions_spiqa_figures.json` | 35 | keep | Claude Haiku | mechanism, pixel path |
| PubLayNet text, multihop | `questions_publaynet_text.json`, `_multihop.json` | 35, 30 | keep | DeepSeek | mechanism |
| PubLayNet figures | `questions_publaynet_figures.json` | 35 | 100+ | Claude Haiku | mechanism, pixel-only |
| PubLayNet caption-matched | `questions_publaynet_figures_caption.json` | 35 | match figures | derived | leakage control |

Rules:

- Done 2026-09-10: `rag ingest --config configs/primary_revision.json --pool 600 --eval 200`
  rebuilt the HotpotQA corpus from a 600-question pool (5,911 paragraphs,
  10,108 chunks; the legacy corpus was a 300-question pool with seed 42) and
  wrote 100 bridge and 100 comparison questions. The pool had to double
  because only about a fifth of HotpotQA is comparison-type, so the new sets
  are not supersets of the legacy 50s and the corpus is harder. 4,974 chunks
  have no cached triples and need one extraction call each (about $1) once
  an OpenAI key exists.
- Every model-authored item passes the model review in 7.3 before it enters
  a config. Record the reviewer verdict on the `QuestionRecord`, drop
  rejects, and report the rejection rate per set in the appendix.
- The graph-seeded flag and `construction_method` are carried into every
  table. The cross-paper set is never pooled with the HotpotQA sets.
- Optional, only if time remains: a second SPIQA cross-paper set authored by
  a strong model with no access to graph entities or edges, reviewed by a
  model from a different family. Not required for submission, because
  HotpotQA already provides independent multi-document evidence.

## 5. Graph experiments (RQ1, RQ2)

Compare every retriever at the same candidate budget on HotpotQA bridge and
comparison (100+ each) and, separately labelled, on the SPIQA cross set:

| arm | status |
| --- | --- |
| dense top-3 (current baseline) | exists |
| dense top-5 (equal budget) | exists |
| BM25 top-5 | `retrieval/baselines.py` |
| BM25 + dense reciprocal-rank fusion top-5 | `retrieval/baselines.py` |
| lexical entity expansion top-5, no graph edges | `retrieval/baselines.py` |
| random passages matched to the +KGret budget | `retrieval/baselines.py` |
| +KG (prompt-side injection) | exists |
| +KGret top-5 | exists |
| one published graph method (HippoRAG or LightRAG) on the primary sets | **not built; design in 5.1, HippoRAG 2 wrapped as a retrieval-only system** |

Report for every arm: Recall@k, MRR, full-evidence completeness, candidate
count, retrieved tokens, EM, F1, `contains`, judge accuracy, and the
evidence-state distribution. Run `rag retrieval-ab` before any generation to
see whether the retrievers differ at all.

### 5.1 Published graph comparator: design

**Decision: wrap the official package as a retrieval-only system. Do not
reimplement the method.** Reviewers want the comparator as published; a
reimplementation invites "you got it wrong", and our pipeline only needs a
ranked list of passages from it. Everything downstream (generator, prompts,
judges, control arms, statistics) stays ours, so the comparison isolates
retrieval.

**Primary comparator: HippoRAG 2** (`hipporag` on PyPI, OSU NLP). It was
built for multi-hop QA and evaluated on HotpotQA, which is our primary set,
and its API separates indexing from retrieval:

```python
from hipporag import HippoRAG
rag = HippoRAG(save_dir=".cache/hipporag/hotpotqa", llm_model_name=..., embedding_model_name=...)
rag.index(docs=[chunk.text for chunk in corpus_chunks])          # OpenIE + graph, once
solutions = rag.retrieve(queries=[question], num_to_retrieve=k)  # docs + doc_scores
rag.get_graph_info()                                              # node/triple counts
```

**Secondary, only if time remains: LightRAG** (`lightrag-hku`). Retrieval
without generation is `aquery_data` or `QueryParam(only_need_context=True,
mode="mix", chunk_top_k=k)`. Its context also carries entity and relation
text, so the equal-budget comparison uses its chunk list only.

**How it plugs in.** (Built 2026-09-11: `retrieval/comparators.py`,
`retrieval/hipporag_index.py`, the `+HippoRAG` system, `comparator_retrievals`
in the config, and `rag comparator-ab`; 8 stub-backed tests. Not yet done:
installing the package and producing the real retrievals file, which needs
the OpenAI key.)


1. `retrieval/comparators.py`: a `HippoRagRetriever` with the same shape as
   `TextIndex.retrieve`: `retrieve(question, k) -> tuple[Passage, ...]`. It
   is fed our own corpus chunks, not raw documents, so every returned text
   maps back to a chunk `source` id through a text-to-id dictionary and the
   gold-provenance and evidence-state logic works unchanged. Score and rank
   come from `doc_scores`.
2. `pipelines/systems.py`: a `+HippoRAG` entry in `SYSTEMS`, implemented
   like `run_baseline` but calling the comparator with
   `k = ctx.candidate_budget`. Same prompt as the baseline. `metadata`
   records the graph statistics from `get_graph_info()` once per run.
3. `retrieval_method` in `ExperimentConfig` gains the value `hipporag`, and
   `rag retrieval-ab` includes it so Recall@k and completeness are compared
   before any generation is paid for.
4. The index is built once per corpus, frozen under `.cache/hipporag/<corpus>/`,
   checksummed, and its package version, OpenIE model, and embedding model
   recorded in the run record. Its OpenIE is LLM-based and nondeterministic
   across rebuilds, so never rebuild between runs that share a table.
5. The dependency goes in an optional extra (`pip install .[comparators]`)
   because it brings its own torch and model stack. The base environment
   stays as it is.
6. Tests: a unit test with a stub retriever proving `+HippoRAG` produces a
   valid `EvidenceBundle` at the candidate budget; a five-question smoke run
   on the desktop GPU before the paid runs.

**Fairness rules.**

- Same chunks, same candidate budget (5), same generator, same prompt as
  every other arm.
- Its indexing LLM should be the same family and tier as our own triple
  extractor. If the defaults differ, say so in the cost table; the extraction
  cost is preprocessing and reported separately, alongside ours.
- Its default embedding model is a large GPU model. Run it on the 16 GB
  desktop, or configure the OpenAI embedding we already use if the package
  supports it, and record which.
- HippoRAG's authors tuned on HotpotQA. State that in the limitations; it
  favours the comparator, not us, so it does not weaken our claim.

**Risks.** Package pins (Python and torch versions) may conflict with the
conda-forge environment, hence the optional extra and a separate env if
needed. Windows support is unverified; plan on the desktop or WSL. OpenIE
over every chunk is one LLM call per chunk, so estimate from chunk counts
before indexing. Budget two to four working days including the smoke run.

**Decisions taken 2026-09-10.**

- Embedding: `text-embedding-3-small`, the same OpenAI embedding our dense
  baseline uses, so the comparator differs from the baseline only in the
  graph. The package's own dense baseline example uses it; confirm the
  `HippoRAG` class accepts it at install time. No GPU needed, so indexing
  runs on the laptop.
- Indexing LLM: `gpt-4o-mini`, which is `EXTRACTION_MODEL` for our own
  triples. Same family and tier, so the preprocessing cost is directly
  comparable. Index HotpotQA (5,122 chunks) and SPIQA (8,474 chunks) only.
- Environment: a separate conda env `rag-comparators` built from an
  `environment-comparators.yml`, plus the `[comparators]` extra in
  `pyproject.toml`. The adapter writes retrieval results to a JSON cache
  that the main `rag` env reads, so the two environments never import each
  other. Windows first; WSL only if the package fails to install.

## 6. Evidence-attribution controls (RQ4)

Every primary graph condition runs the five arms already in
`configs/primary_revision.json`: normal, closed-book, shuffled unrelated
context with matched token count, oracle gold context, and partial-gold
context for multi-source questions. These replace the old sample audit of
correct-but-incomplete answers by measuring the same thing for every
question. The closed-book and shuffled arms bound how much of "accuracy
without complete gold provenance" is prior knowledge. Use that exact phrase
in every table and caption. "Parametric floor" is retired.

## 7. Validation with large hosted models

This section replaces the human annotation in the old plan: two reviewers on
200 triples, 400 double-annotated outputs with kappa, human review of flagged
leakage items, and a 20% human audit of unflagged items. None of it is done
by people.

### 7.1 Family rules

The generators are gpt4o-mini (OpenAI), gemini-flash-lite (Google),
llama4-scout and llama4-maverick (Meta), plus whatever section 9 adds. Text
and cross-paper questions were authored by DeepSeek; figure questions by
Claude Haiku.

- **A judge is a tier above every generator it grades and from a family that
  is not any generator's family.** With OpenAI, Google, and Meta all in the
  generator roster, the primary text judge is a **Claude-class model**
  (Sonnet or Opus tier). Second text judge: DeepSeek (current) or another
  non-generator family. Report agreement between the two.
- **A reviewer is never from the family that authored the items it
  reviews.** DeepSeek-authored sets are reviewed by the Claude-class model.
  Claude-Haiku-authored figure sets are reviewed by a GPT-4o-class or
  Gemini-Pro-class vision model.
- **Vision judge.** Upgrade from Claude Haiku. Because the figure questions
  were authored by Claude Haiku, use two vision judges from different
  families (one Claude-class, one GPT-4o-class or Gemini-Pro-class) and
  report their agreement. State the author-judge family overlap in the
  limitations.
- Temperature 0 for every judge and reviewer call. Log the full raw call.
- If section 9 adds a Claude generator, the primary judge family must move
  or that generator's rows must be graded by the second judge only. Decide
  before the runs, not after.

### 7.2 Judge protocol

- Keep the binary accuracy verdict, but have the strong judge write a
  one-sentence justification before the digit, stored in the run record for
  audit. Parsing stays binary.
- Drop the relevancy judge. It is nearly always 1 and was a third of all judge
  calls. Keep accuracy and faithfulness.
- **Synthetic calibration, run once per judge and prompt version.** Feed the
  judge (a) the human reference answer as if generated, which must score 1,
  and (b) the reference from a different question of the same set, which must
  score 0. Report the resulting false-negative and false-positive floors. This
  is a sanity bound, not calibration against human labels, and is described
  as such.
- Report, per set and generator: judge accuracy, EM, F1, `contains`, the
  judge-versus-F1 disagreement rate, and two-judge agreement (raw and
  Cohen's kappa). Where reference metrics exist they are the headline. Where
  the judge is the headline (SPIQA-native), the disagreement rate shows how
  much rides on it.
- Judge stability: re-judge a stratified 20% subset three times and report
  the flip rate.
- Coverage follows the lean design in section 15: the strong judge grades
  every output only where it carries the headline (SPIQA-native, PubLayNet
  figures) and a stratified 20% elsewhere; DeepSeek covers everything
  cheap; Opus 5 adjudicates disagreements only.

### 7.3 Model review of model-authored question sets

For each item in the DeepSeek- and Claude-Haiku-authored sets, the reviewer
model (family per 7.1) receives the question, the reference answer, the gold
provenance chunk or crop, and any caption or corpus text that mentions the
answer, and returns four verdicts with reasons:

1. **Answerable:** the reference answer follows from the gold provenance
   alone.
2. **Unique:** the question has one defensible answer.
3. **Well-formed:** the question is specific, not a template fragment.
4. **Leakage:** the answer, a paraphrase, or a numeric equivalent appears in
   a caption or in retrievable corpus text (pixel-only sets only).

An item enters a config only if 1 to 3 pass. Verdict 4 sets
`leakage_status` and, for pixel-only sets, excludes the item from the
pixel-only headline. Store reviewer model, verdicts, and reasons on the
`QuestionRecord`. Report per-set rejection counts. The paper calls these
sets *model-reviewed*.

### 7.4 Model-judged sufficiency for non-gold support

For every correct answer produced without complete gold provenance, the
strong judge is given only the retrieved context and the question and asked
whether the reference answer is derivable from that context. A yes assigns
`NON_GOLD_SUPPORTED` as a *model-judged* state. This lifts the annotation
guide's restriction that the state cannot be assigned, provided every table
that uses it says model-judged. Update `protocols/ANNOTATION_GUIDE.md`
accordingly.

### 7.5 Visual leakage audit

Replaces exact-string "verification":

1. normalised string and numeric-equivalence check (existing screen);
2. semantic-similarity screen against captions and retrievable text;
3. reviewer-model entailment check (7.3 verdict 4) on every flagged item and
   on every item of the pixel-only headline set, not a 20% sample, since the
   sets are small enough to check in full.

Report construction intent and leakage status as two separate columns.
Retire "verified pixel-only"; use "model-screened pixel-only".

### 7.6 Graph quality

- `rag graph-grounding` over the whole graph: subject, object, and relation
  presence in the source chunk, numeric endpoints, surface-form collisions.
  Called *grounding*, never *precision*.
- `rag audit-graph --count 300`, stratified by corpus and extraction
  frequency. The Claude-class model then labels each triple for source
  entailment, canonicalisation of subject and object, relation specificity,
  and provenance correctness, with reasons. Reported as *model-judged triple
  precision*, with a second judge on a 100-triple subset and agreement
  reported.
- `KnowledgeGraph.statistics()`: entity and edge counts, duplicate rate,
  isolated-node rate, relation distribution, provenance coverage,
  entity-matching failure categories.

## 8. Multimodal evaluation (RQ3)

- Primary set: SPIQA-native primary (150, human). PubLayNet pixel-only grows
  to 100+ and is model-reviewed under 7.3.
- Keep caption-authored and image-authored labels separate throughout.
- Compare CLIP ViT-B/32 with one document-oriented visual retriever
  (ColPali-class), run on the GPU env.
- Add an oracle-gold-image arm to separate image-retrieval failure from
  figure-reading failure.
- Report Recall@1 as the production ceiling, since only the top-1 crop is
  supplied. Recall@3 and @5 are diagnostic only.
- Reconcile the 25/34 vs. n=35 denominator and name the unscorable item.

## 9. Generator roster

The submitted paper used four generators from 2024 and early 2025. A
reviewer at DKE in late 2026 will ask why. The roster for the revision:

- **Keep the four legacy generators** (gpt4o-mini, gemini-flash-lite,
  llama4-scout, llama4-maverick). They give continuity with the submitted
  results, and one weak generator is needed because a strong generator
  raises the closed-book floor and shrinks the measured effect.
- **Add one cheap current open-weight model** (Qwen or DeepSeek family, via
  OpenRouter with the upstream provider pinned). Five generators in total;
  the lean design in section 15 runs all five only on the primary arms. Pick from
  `configs/pricing_2026-09-03.json` after adding them; take prices and
  version strings from the providers on the day, never from memory, and
  freeze them in the snapshot.
- **Adding a Claude generator changes the judge family** (7.1). Either the
  primary judge moves to a family with no generator, or the Claude generator
  is judged only by the second judge. If neither is acceptable, leave Claude
  out of the generator roster and keep it as the judge. That is the
  recommended default, because a clean judge is worth more than a fifth
  generator family.
- Every generator gets the same prompts, the same evidence bundles, and the
  same question checksums. Model version strings go in the run record.
- The generator dimension is a robustness axis, not a set of replications.
  Pooled results use the question-level cluster bootstrap (section 10).

## 10. Statistics

All implemented in `evaluation/statistics.py` and `rag paired-analysis`.
Apply them everywhere, not only where the result is favourable.

- Per-question paired bootstrap 95% intervals (10,000 draws) for every
  primary difference; absolute difference and odds ratio.
- Exact McNemar for paired binary comparisons.
- Holm correction within two pre-specified families: graph (RQ1, RQ2) and
  multimodal (RQ3).
- Question-level cluster bootstrap for anything pooled across generators.
  Generators are never treated as independent replications.
- Three repeated generations on a stratified 20% subset for hosted-model
  variability.

## 11. Claims, cost, reproducibility

- Rename "Explainability" to "Evidence provenance and audit traces".
- Split preprocessing cost (triple extraction, captioning, embedding,
  indexing) from online cost (retrieval, generation) and from judging and
  reviewing. Provider prices and access dates are frozen in
  `configs/pricing_2026-09-03.json`; add every new model before the runs.
- Do not contrast "open" and "closed" models as deployment categories when
  both are served through hosted APIs. Use "open-weight" and "proprietary"
  consistently.
- The limitations subsection must cover explicitly: the graph-seeded
  mechanism set; model-authored and model-reviewed questions with no human
  review; judges and triple labels not calibrated against human labels for
  this study; author-judge family overlap on figure questions; model-version
  drift; API nondeterminism; benchmark contamination; NON_GOLD_SUPPORTED
  being model-judged.
- Related-work table contrasting this study with GraphRAG, HippoRAG,
  LightRAG, ColPali, M3DocRAG, MMDocRAG, and existing RAG evaluation
  frameworks.
- Every table and figure is produced by `rag artifacts --manifest` from one
  frozen manifest with run IDs and checksums. `rag validate-release` passes.
  Legacy prefix-bug runs stay in `archive/invalid_runs`.

## 12. Manuscript and submission package

- Restore the author block, Coventry affiliations, corresponding author, and
  PDF metadata in the unblinded build. Produce a blinded build if the journal
  asks for one.
- Replace `\journal{Information systems}` with the target journal.
- Split the two "Float too large" tables (around lines 1153 and 1328) with
  `longtable` or per-corpus tables. Clear every visible overfull box of the
  48. Break the GitHub URL safely.
- Fix "Appendix Appendix A", encoding artefacts, open-weight/closed-weight
  wording, and every table denominator.
- Clean-checkout build: zero undefined references, zero oversized floats, no
  clipped content.

## 13. Acceptance criteria before resubmission

1. Primary graph comparisons on 100+ human-authored HotpotQA questions per
   type, at equal candidate budgets, with all arms in section 5 including one
   published graph method.
2. Graph-seeded and independent sets reported in separate tables.
3. Every headline difference has a paired interval, effect size, and
   Holm-corrected test.
4. Judges: family rules honoured, synthetic calibration reported, two-judge
   agreement and judge-vs-F1 disagreement reported.
5. Model review complete for every model-authored item in a config, with
   rejection rates reported.
6. Model-judged triple precision on 300 triples, plus whole-graph grounding
   and structural statistics.
7. All five control arms complete for every primary condition.
8. Visual leakage audit complete on the full pixel-only headline set;
   document-image retriever and oracle-image arms run.
9. One frozen manifest produces every table and figure; the validator
   passes.
10. PDF has correct metadata, authors, journal, denominators, and no clipped
    floats.
11. Internal methodological review by two coauthors.
12. OA eligibility confirmed in writing by oa.lib@coventry.ac.uk.

## 14. Order of work

1. Done 2026-09-10: HotpotQA scaled to 100 per type (section 4). Checksums are
   in the question files; freeze them in the release manifest with the runs.
2. Done 2026-09-10: roster decided (section 15), registered in `clients.py`,
   priced in `configs/pricing_2026-09-10.json`, and wired into every config.
   Still open: an `OPENROUTER_API_KEY` and an OpenAI key from Soki, and the
   check of OpenRouter `:batch` semantics.
3. Model review of all model-authored sets (7.3); re-freeze.
4. Synthetic judge calibration (7.2).
5. `rag retrieval-ab` across all arms. The `+HippoRAG` system is built (5.1);
   install the package in the comparators env and run the indexer once the
   OpenAI key exists, then `rag comparator-ab` before any generation.
6. Paid generation runs with all control arms; judging.
7. Graph audit (7.6), visual audit (7.5), sufficiency judgements (7.4).
8. Statistics, artifacts, manifest, validator.
9. Manuscript rewrite around section 3; LaTeX fixes; internal review.
10. OA confirmation; submit to DKE.

## 15. Provider access and cost estimate (2026-09-10)

### Where the models come from today

Every generator and judge is a hosted API; nothing runs locally. Five keys:

| registry name | model | provider | key |
| --- | --- | --- | --- |
| gpt4o-mini, gpt4o, text-embedding-3-small | OpenAI | OpenAI direct | `OPENAI_API_KEY` |
| gemini-flash-lite | gemini-3.1-flash-lite | Google, OpenAI-compatible endpoint | `GEMINI_API_KEY` |
| llama4-scout, llama4-maverick | Meta Llama 4 | DeepInfra | `DEEPINFRA_API_KEY` |
| deepseek | deepseek-chat | DeepSeek direct | `DEEPSEEK_API_KEY` |
| claude-haiku | claude-haiku-4-5 | Anthropic direct, native SDK | `ANTHROPIC_API_KEY` |

None of these keys exist on the laptop. Everything except the document-image
retriever runs on hosted models, so the laptop is sufficient for the whole
revision once keys exist.

### OpenRouter

Adopted; the roster, provider rules, and batch caveat are in the subsection
after the lean design below.

### Lean design (adopted 2026-09-10)

The first estimate (about $300 to $420) came from judging every output with
a strong model and running every generator on every arm. Both are waste.
The judge is only load-bearing where the reference answer is discursive
(SPIQA-native), and secondary retrieval arms only need to show whether they
match +KGret, which two generators establish. The lean design keeps every
acceptance criterion in section 13 and cuts spend by roughly six times.

**Generator matrix.** Five generators: the four legacy ones plus one cheap
current open-weight model (Qwen or DeepSeek family, via OpenRouter).

| arm group | arms | generators | note |
| --- | --- | --- | --- |
| primary graph arms | dense top-3, dense top-5, +KG, +KGret, HippoRAG | all 5 | full factorial, headline tables |
| secondary retrieval arms | BM25, RRF, lexical entity expansion, random padding | 2 (one weak, one strong) | show whether a non-graph expansion matches +KGret |
| controls, per-generator | closed-book, oracle | all 5 | the parametric floor and the ceiling are generator properties |
| controls, shared | shuffled, partial-gold | 2 | context manipulations, not generator properties |
| visual primary (SPIQA-native 150) | text-only, +MM CLIP, +MM document retriever, +both, oracle image, closed-book | all 5 | RQ3 headline |
| visual mechanism (PubLayNet figures 100+) | pixel-only, caption-matched, +both, closed-book | all 5 | small set, cheap |
| text mechanism sets (about 200 q) | baseline, +KG, +KGret | 2 | continuity with the submitted results only |
| repeats | primary arms, 10% stratified subset, two extra runs | all 5 | variability estimate |

That is about 21,000 generated outputs instead of 60,000.

**Judge coverage.** Judge where the judge carries the claim; sample where it
corroborates.

| set | headline metric | full coverage | strong judge |
| --- | --- | --- | --- |
| HotpotQA, SPIQA cross, text mechanism | EM and F1 (benchmark metrics, free) | DeepSeek accuracy on every output; faithfulness only on non-control arms | Sonnet 5 on a stratified 20% for judge-vs-judge and judge-vs-F1 agreement |
| SPIQA-native primary | Sonnet 5 accuracy | Sonnet 5 accuracy and vision faithfulness on every output, via Batch | DeepSeek accuracy as second judge on every output; second vision family on 20% |
| PubLayNet figures | Sonnet 5 accuracy | Sonnet 5 accuracy and vision faithfulness on every output, via Batch | second vision family on 20%; author-judge overlap stated |
| repeats | EM, F1, DeepSeek | DeepSeek only | none |

Faithfulness is not judged on closed-book or shuffled arms, where it is
undefined. Opus 5 adjudicates only outputs where the two judges disagree.
All Anthropic judge calls go through the Batch API at half price.

**Other cuts.** HippoRAG indexed on HotpotQA only (5,122 chunks); the
SPIQA cross set is a mechanism set and does not need the comparator.
Corpora are copied from the desktop, not rebuilt. Relevancy judging is
dropped entirely.

### OpenRouter roster (chosen 2026-09-10)

Prices below were read from OpenRouter's public model list on 2026-09-10
(USD per million tokens, input / output). OpenRouter is OpenAI-compatible,
so each model is one registry entry with the OpenRouter base URL and an
`OPENROUTER_API_KEY`. Only the embedding model and HippoRAG's indexing need
the OpenAI key directly, because OpenRouter lists no embedding models.

| role | registry name | OpenRouter id | in / out | why |
| --- | --- | --- | --- | --- |
| generator (legacy) | gpt4o-mini | `openai/gpt-4o-mini` | 0.15 / 0.60 | same price as direct |
| generator (legacy) | gemini-flash-lite | `google/gemini-3.1-flash-lite` | 0.25 / 1.50 | dearer than the direct rate in the 09-03 snapshot (0.10 / 0.40); keep the Google key if that matters |
| generator (legacy) | llama4-scout | `meta-llama/llama-4-scout` | 0.10 / 0.30 | pin provider DeepInfra to match the legacy backend |
| generator (legacy) | llama4-maverick | `meta-llama/llama-4-maverick` | 0.20 / 0.70 | pin provider DeepInfra |
| generator (new, 5th) | qwen3.7-flash | `qwen/qwen3.7-flash` | 0.03 / 0.13 | cheapest current vision-capable open-weight model, Alibaba family, released 2026-07 |
| primary judge, text and vision | claude-sonnet-5 | `anthropic/claude-sonnet-5` | 2.00 / 10.00 | outside every generator family; `:batch` variant at 1.00 / 5.00 |
| adjudicator on disagreements | claude-opus-5 | `anthropic/claude-opus-5` | 5.00 / 25.00 | `:batch` at 2.50 / 12.50 |
| second judge, text | deepseek-v4-pro | `deepseek/deepseek-v4-pro-0813` | 0.58 / 1.74 | a tier above the generators, no vision; replaces deepseek-chat |
| second judge, vision (20%) | grok-4.3 | `x-ai/grok-4.3` | 1.25 / 2.50 | xAI is outside every generator and author family |
| figure-question author | claude-haiku | `anthropic/claude-haiku-4.5` | 1.00 / 5.00 | unchanged, keeps the new PubLayNet items consistent with the old |
| HippoRAG OpenIE | gpt4o-mini | OpenAI direct | 0.15 / 0.60 | matches our extractor |
| embeddings | text-embedding-3-small | OpenAI direct | 0.02 / 0 | not on OpenRouter |

Rejected for the roster: GPT-5.4-mini, Gemini 3.x Flash, and Claude Haiku as
the fifth generator, because a Claude generator would put the judge in a
generator family and the other two add nothing the legacy models do not
already represent. Gemini 3.1 Pro as a vision judge, because Google is a
generator family. Any `-exp` or `-preview` model, for reproducibility.

**Batch variants.** OpenRouter lists `:batch` ids for Claude at half
price. Our client calls synchronously, so before relying on them verify how
OpenRouter serves a `:batch` id (queued asynchronous completion or a
delayed synchronous reply). If it is asynchronous, judging needs a small
batch submitter; the saving is about $19 on the total below.

**Provider pinning.** For the two Llama models pass OpenRouter's provider
preference in the request body so every call goes to DeepInfra, the backend
the legacy runs used, and record the served provider from the response in
the run record. Without that, OpenRouter may route to a different
quantisation between runs.

### Estimate with the OpenRouter roster (superseded by the hard-cap section)

Computed from the lean matrix (22,300 outputs: 14,200 text, 8,100 pixel),
legacy token means (400 in / 15 out text generation, 1,500 in / 30 out
pixel generation, 150-token accuracy judge, 600-token text faithfulness,
1,500-token vision faithfulness, 40-token justifications) and the prices
above. The script is `docs/cost_model_2026-09-10.py`; rounded.

| item | standard ids | with `:batch` ids |
| --- | --- | --- |
| generation, five generators, all arms and repeats | 3 | 3 |
| DeepSeek V4 Pro second judge, full coverage | 8 | 8 |
| Sonnet 5 primary judge (all pixel outputs, 20% of text) | 33 | 16 |
| Grok 4.3 second vision judge on 20% of pixel outputs | 4 | 4 |
| Opus 5 adjudication on about 10% disagreements | 5 | 2 |
| model-judged sufficiency checks (3,000) | 5 | 5 |
| question review, triple audit, synthetic calibration | 3 | 3 |
| HippoRAG indexing, HotpotQA only | 2 | 2 |
| new PubLayNet figure questions | 1 | 1 |
| **total** | **about 63** | **about 44** |

Corpus rebuild, only if the desktop copy is unavailable: up to 35 extra.
Half of the standard total is the vision-faithfulness judging of pixel
outputs; if it must go lower, judge vision faithfulness on the SPIQA-native
primary set only and the total drops to about 45 without batch.

### Cost configuration (the 60-dollar cap was lifted on 2026-09-11)

A $60 ceiling was set on 2026-09-10 and lifted on 2026-09-11. The three
changes made under it stay, because each is also the better design; they
bring the standard-id estimate to about $54 (about $40 with `:batch` ids,
after the HotpotQA rebuild). No config carries a spend cap any more;
`budget_usd` remains an optional per-run guard in the pipeline:

- **Judge split by question set, in the configs.** The text sets
  (`primary_revision.json`, `spiqa_cross_paper.json`) are judged in full by
  DeepSeek V4 Pro with EM and F1 as the headline; the visual sets
  (`spiqa_native_visual.json`, `publaynet_figures.json`) are judged by
  Sonnet 5, which is load-bearing there. The 20% Sonnet sample on text sets
  is dropped from the paid runs; the judge-vs-judge agreement figure comes
  from the visual sets and the synthetic calibration instead.
- **Relevancy judging removed from the pipeline.** It was a third of all
  judge calls and almost always 1. Runs now record accuracy and
  faithfulness only; the flat table tolerates legacy summaries that still
  carry the column.
- **Vision faithfulness only where an image was supplied.** The pipeline
  already sends the image only for image arms, so text-only arms in the
  visual sets get the cheap text prompt. The cost model now reflects that.

**Optional per-run guard (`budget_usd`).** If set in a config, a run that
passes the value (generation plus judging) aborts after the current
question, writes its partial rows, and raises `BudgetExceededError`; the
value is part of the config hash. Since 2026-09-11 no config sets it. Set it
only for a pilot where a runaway judge loop would be expensive.

| item | standard ids | with `:batch` ids |
| --- | --- | --- |
| generation, five generators, all arms and repeats | 3 | 3 |
| DeepSeek V4 Pro, text sets in full | 8 | 8 |
| Sonnet 5, visual sets (accuracy on all, vision faithfulness on image arms) | 25 | 12 |
| Grok 4.3 second vision judge on 20% of image-arm outputs | 2 | 2 |
| Opus 5 adjudication on about 10% disagreements | 4 | 2 |
| model-judged sufficiency checks (3,000) | 5 | 5 |
| question review, triple audit, synthetic calibration | 3 | 3 |
| HippoRAG indexing, HotpotQA only (10,108 chunks) | 5 | 5 |
| triple extraction for the 4,974 new HotpotQA chunks | 1 | 1 |
| new PubLayNet figure questions | 1 | 1 |
| **total** | **about 54** | **about 40** |

The estimate is a planning figure, not a limit.

### Preprint

The submitted paper is public as arXiv 2607.16604, titled *When Do
Multimodal and Graph-Augmented RAG Help? A Controlled Evaluation for
Document Question Answering*. The journal version must cite it, and the
new title in section 3 should replace the arXiv title on the next arXiv
version once the revision is complete.

## 16. Journal check (2026-09-10)

Targeting by reference venue does not work here: of the 47 references, 3
are journal papers (two TACL, one IEEE Transactions on Big Data); the rest
are conference and arXiv papers. The venue decision therefore rests on scope
fit, impact, and OA coverage.

**Coverage rule, from Elsevier's Jisc agreement page.** Eligibility is
decided by the corresponding author's affiliation alone; co-author
affiliations are irrelevant. Hybrid journals: no APC. Fully gold OA
journals: a discount only. Acceptance date must fall between 2026-01-01 and
2028-12-31. Elsevier's page has a "Find a participating journal" search;
check each candidate there and confirm in writing with
oa.lib@coventry.ac.uk. Soki must be corresponding author.

**Candidates for Soki's 6 to 9 impact-factor wish.** All three are Elsevier
hybrid titles, so they should be APC-free under the agreement; the
participating-journal search has not yet been run for them.

| journal | reported IF (2025) | fit | risk |
| --- | --- | --- | --- |
| Knowledge-Based Systems | about 7.6, Q1 | strong: knowledge graphs with LLMs, evaluation methodology, provenance | same tier as IPM, which desk-rejected; the revision must be complete |
| Expert Systems with Applications | about 8.5, Q1 | good: applied AI systems with thorough experiments | very high submission volume, heavy desk rejection, long queues |
| Engineering Applications of Artificial Intelligence | about 6.2, Q1 | moderate: expects an engineering application framing | framing mismatch for an evaluation paper |

**Proposed order to put to Soki** (changes the 2026-09-03 agreement, so it
is a proposal, not a decision): Knowledge-Based Systems, then Data &
Knowledge Engineering, then Expert Systems with Applications, then Applied
Artificial Intelligence. DKE stays in the list because its scope is the
closest match and its bar is more realistic; KBS goes first only because
the revised paper, with the comparator, controls, and statistics, is aimed
at exactly the tier that rejected it.

Impact figures come from third-party aggregator pages and should be
re-checked on Journal Citation Reports before the email to Soki.
