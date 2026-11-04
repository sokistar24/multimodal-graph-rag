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
  document-image retriever that the local GPU (RTX 5060 Ti,
  `environment-gpu.yml`) was reserved for was dropped on 2026-10-20 and is
  future work, so the revision needs no GPU.
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

Title (changed 2026-10-20, no colon): *A Controlled Evaluation of Evidence
Attribution in Graph-Augmented and Multimodal RAG for Document Question
Answering*. It keeps "controlled evaluation" from the arXiv title so the
lineage is visible. The 2026-09-10 title below is superseded.

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
  are not supersets of the legacy 50s and the corpus is harder. Done
  2026-10-20: the 4,969 uncached chunks were extracted with gpt4o-mini
  through OpenRouter (about $0.45), giving 72,946 triples over the corpus.
  The parser now accepts a complete list followed by a stray closing bracket;
  one chunk (`Duke_Tumatoe.txt`, a distractor and not gold for any question)
  never parsed because its source text has an unbalanced quote, and is cached
  with no triples.
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

| arm | system name |
| --- | --- |
| dense top-3 (current baseline) | `dense-top3` |
| dense top-5 (equal budget) | `baseline` |
| BM25 top-5 | `bm25` |
| BM25 + dense reciprocal-rank fusion top-5 | `hybrid-rrf` |
| lexical entity expansion top-5, no graph edges | `+lexical` |
| random passages matched to the +KGret budget | `+random` |
| +KG (prompt-side injection) | `+KG` |
| +KGret top-5 | `+KGret` |
| one published graph method on the primary sets | `+HippoRAG` (HippoRAG 2, retrieval file still to be produced, 5.1) |

**Built 2026-10-20.** Every arm above is a generation system, and
`rag retrieval-ab` reports gold completeness for all of them from the same
retrieval functions the generation runs call. Decisions taken while building
them:

- **+KG uses dense top-3**, as the manuscript defines it and as the submitted
  runs did; the package had silently moved it to the candidate budget (5).
  `dense-top3` is its passage-matched control.
- **One prompt for every passage arm, with no arm marker.** +KGret no longer
  labels its added passages "graph-bridged" in the prompt or the judge
  context, so the arms differ in retrieval alone. The trace still records
  why each passage was chosen. The judge and +KGret prompts changed, so the
  configs moved to `prompt_version` `evidence-attribution-v2`.
- **BM25 and the hybrid run over the dense index's own chunks.** RRF fuses
  the top 100 of each ranking with the usual constant 60; ties go to the
  earlier chunk.
- **Lexical expansion starts from the same entities as +KGret** (graph nodes
  named in the question, whole word, at least five characters) but looks for
  those names in chunk text instead of following edges to provenance. It
  searches the same 300 dense candidates as +KGret and, like it, may add
  fewer than two passages, so its passage count is at most five.
- **Random padding** adds two random chunks from sources not in the dense
  top-3, one per source, seeded per question.
- **Every arm runs on all seven generators in one config per question type**
  (`primary_revision.json` for bridge, `hotpotqa_comparison.json` for
  comparison). The lean design ran the secondary arms on two generators, but
  `rag paired-analysis` compares systems within one detail file, and the
  manuscript's headline table pools every arm over the same generators. The
  extra outputs are DeepSeek-judged text, about a dollar.

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

rag = HippoRAG(
    save_dir=".cache/hipporag/hotpotqa", llm_model_name=..., embedding_model_name=...
)
rag.index(docs=[chunk.text for chunk in corpus_chunks])  # OpenIE + graph, once
solutions = rag.retrieve(queries=[question], num_to_retrieve=k)  # docs + doc_scores
rag.get_graph_info()  # node/triple counts
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
  audit. Parsing stays binary. Built 2026-10-20 for every judge: the verdict
  must be alone on the last line (a `Grade:` label is tolerated), anything
  else stops the run, and the trace keeps the justification and the full
  reply. The token cap rose from 8 to 1,024 because GPT-6.1 Sol and Opus 5.5
  reason before answering and OpenRouter counts that against the cap. The
  2026-10-20 HotpotQA pilot showed DeepSeek V4 Pro reasons too (median 31,
  95th percentile 266, maximum 715 output tokens over 412 judge calls) and one
  call exhausted 1,024 and returned nothing, stopping the run, so the cap is
  now 4,096.
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
- Dropped 2026-10-20: the comparison of CLIP ViT-B/32 with a
  document-oriented visual retriever (ColPali-class). It needs a GPU, no
  headline claim depends on it, and it is listed as future work. The paper
  must say the visual results are for CLIP retrieval only.
- Add an oracle-gold-image arm to separate image-retrieval failure from
  figure-reading failure. Built 2026-10-20 as the control `oracle-image`: the
  baseline passages plus the gold crop (pixels) or its caption (captions),
  so its gap to +multimodal is image retrieval alone. It is on in all three
  figure configs, together with `+both`.
- The caption-flagged contrast set (132 items, every flagged item rather than
  a sample) is `data/questions/questions_spiqa_native_flagged.json`, run by
  `configs/spiqa_native_flagged.json` under the primary set's protocol.
  `rag questions --protocol spiqa-native --only-flagged` writes it.
- Reproducibility gap found 2026-10-20: the committed
  `spiqa_images/captions.json` (978 crops, from the August ingest) covers only
  35 of the 579 gold crops, so converting the gold QA on a fresh checkout
  yields 35 items, not 579. The 579-item set was converted on the desktop
  against a newer caption file. Commit that file, or record its checksum in
  the release manifest, before submission.
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
- **Add a strong tier (decided 2026-10-20).** Gemini 3.8 Flash
  (`google/gemini-3.8-flash`, 0.825 / 4.125) and Qwen 3.8 Max
  (`qwen/qwen3.8-max-0902`, a dated snapshot, 2.00 / 6.00), both
  vision-capable, read from OpenRouter on 2026-10-20 and frozen in
  `configs/pricing_2026-10-20.json`. Neither family judges, so the judge
  rules are unchanged. Both think before answering, with thinking left at
  the provider default as for Gemini Flash-Lite; the 800-token generation
  cap includes that thinking, so the pilot must confirm no answer comes back
  empty. Seven generators in total, in every config.
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
   oracle-image arm run. (The document-image retriever is future work.)
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

**Update 2026-10-20: OpenRouter only.** Every registry entry, including the
four legacy generators, the text and figure authors, and
`text-embedding-3-small`, is now served through OpenRouter, so a run needs only
`OPENROUTER_API_KEY`. Aliases are unchanged. `llama4-scout` is pinned to
DeepInfra (the legacy backend); DeepInfra no longer serves `llama4-maverick`
on OpenRouter, so it is pinned to Parasail's FP8 endpoint, the same FP8 build,
and the paper should say so. Prices are frozen in
`configs/pricing_2026-10-20.json`; Gemini Flash-Lite and Maverick cost more
through OpenRouter than the direct rates in earlier snapshots. Still open: the
HippoRAG indexer builds its own client from `OPENAI_API_KEY`; point it at
OpenRouter (key plus `https://openrouter.ai/api/v1` base URL) and test once
before indexing. **Judges changed on 2026-10-20.** GPT-6.1 Sol (OpenAI, the
Sonnet 5.5 tier at the same price) replaces Claude Sonnet 5 as the primary
judge; Claude Opus 5.5 replaces Opus 5 as adjudicator; Grok 4.7 replaces
Grok 4.3 as the second vision judge. DeepSeek V4 Pro is unchanged. The roster
table below reflects this. Because gpt4o-mini is a generator, a GPT judge now
shares a family with one generator, which section 7.1 would rule out. Decided
2026-10-20: no fallback judge. GPT-6.1 Sol scores every generator, including
gpt4o-mini, and the paper reports the overlap as a limitation (possible
same-family preference on the gpt4o-mini rows). GPT-6.1 Sol and Opus 5.5 are reasoning models whose OpenRouter
listing omits `temperature`, and judges are called with an 8-token cap, so run
the `--limit 5` pilot first and confirm every judge returns a bare 0 or 1.
The five-key table below is the pre-2026-10-20 setup.

**Cost re-estimated 2026-10-20 (later the same day).** The configs now run
seven generators, every retrieval arm on every generator, `+both` and the
oracle-image arm on the figure sets, and the 132-item caption-flagged set.
`docs/cost_model_2026-10-20.py` puts the whole revision at about **$174**
with no hidden reasoning, and about **$407** if each thinking generator call
spends 400 hidden tokens and each Sol or Opus call 300. GPT-6.1 Sol judging
the figure sets is most of it, and the two vision modes double that part.
The pilot measures the real reasoning spend; if it lands near the upper
figure, the cheapest cuts are judging the caption-mode runs of the flagged
set with a sample instead of in full, and checking whether OpenRouter serves
a half-price `:batch` id for Sol. The lean-design tables below predate this.

**Cost measured by the pilots, 2026-10-20.** Five-question pilots on all
seven generators, billed by OpenRouter, cost $0.49 for HotpotQA bridge (12
arms, DeepSeek V4 Pro judging), $0.90 for SPIQA native in pixel mode and
$0.85 in caption mode (6 arms, GPT-6.1 Sol judging). Per question that is
about $0.10 for a HotpotQA set with +HippoRAG added, and $0.35 for a SPIQA
set run in both vision modes. Scaled to the configs, HotpotQA bridge and
comparison come to about $21, SPIQA primary (150) $53, the flagged set (132)
$46, PubLayNet figures (35, both modes) about $10, and the SPIQA cross set
(50) about $3, so about $133 for the evaluation runs. Adjudication, the second
vision judge, sufficiency checks, and the repeat runs for hosted-model
variability add roughly $30 to $50, which puts the revision near **$170 to
$190**, close to the lower model estimate. Hidden reasoning stayed small. The
pilot spend itself was about $2.80.

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
| visual primary (SPIQA-native 150) | text-only, +MM CLIP, +both, oracle image, closed-book | all 5 | RQ3 headline (document retriever dropped, future work) |
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
| generator (new, 5th) | qwen3.7-flash | `qwen/qwen3.7-flash` | 0.03 / 0.13 | cheapest current vision-capable model, Alibaba family, released 2026-07; API-only, not open-weight (corrected 2026-10-20: OpenRouter lists no released weights for it) |
| primary judge, text and vision | gpt-6.1-sol | `openai/gpt-6.1-sol` | 2.00 / 10.00 | replaced Claude Sonnet 5 on 2026-10-20; same tier and price as Sonnet 5.5. OpenAI is also a generator family (gpt4o-mini): see the note above |
| adjudicator on disagreements | claude-opus-5.5 | `anthropic/claude-opus-5.5` | 4.00 / 20.00 | replaced Opus 5 on 2026-10-20 |
| second judge, text | deepseek-v4-pro | `deepseek/deepseek-v4-pro-0813` | 0.58 / 1.74 | a tier above the generators, no vision; replaces deepseek-chat |
| second judge, vision (20%) | grok-4.7 | `x-ai/grok-4.7` | 2.00 / 6.00 | xAI is outside every generator and author family; replaced Grok 4.3 on 2026-10-20 |
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

## 17. Run log (2026-10-21 to 2026-10-23)

**Which config wrote each trace.** The five HotpotQA bridge runs started while
the roster was changing, so their identities come from three versions of
`configs/primary_revision.json`: Llama 4 Scout from `a018a8c`, GPT-6 Luna,
Gemini 3.8 Flash and Qwen 3.8 Max from `83e0783`, and Gemma 4 31B from the
current file. Resume or amend a bridge trace with
`git show <commit>:configs/primary_revision.json` saved to a file. The
comparison and SPIQA cross-paper traces match the current configs.

**Measured cost, five generators.** HotpotQA bridge $6.11 generation and
$8.34 judging; comparison $5.30 and $7.30; SPIQA cross-paper $2.78 and $4.24.
The HotpotQA sufficiency pass (GPT-6.1 Sol) cost about $3.40 and the 20%
stability pass (DeepSeek V4 Pro, three repeats) about $4.90.

**Generator cap raised from 800 to 8,192 (2026-10-23).** OpenRouter counts
hidden reasoning against `max_tokens`. On the full runs 520 of 14,500 answers
reached the 800 cap and came back empty or cut off, which the five-question
pilots did not show: Qwen 3.8 Max 457 (across every arm, and 53 of 100
closed-book bridge answers), GPT-6 Luna 50 (mostly closed-book), Gemma 4 31B 10
and Llama 4 Scout 3 (closed-book). The cap is now 8,192 and part of the run
identity. The capped answers are regenerated from their stored evidence with
`rag evaluate --resume TRACE --regenerate-capped` (generation $2.65 to $23 by
dry run); originals go to `artifacts/runs/pre_regeneration/`. A replay of all
15,300 stored records through a stub client reproduced every evidence bundle.

**Containment fixed (2026-10-23).** Answer containment tested a raw
substring, so a reference of "no" matched "know" and "not". It now matches
whole tokens. Containment on comparison questions under dense top-5 falls
from 0.750 to 0.700; analyses rescore every stored answer.

**Pooled tests made cluster-aware (2026-10-23).** An exact McNemar test on the
500 question-generator pairs treats the five answers to one question as
independent and disagreed with the cluster-bootstrap intervals. Paired
p-values now come from a sign-flip permutation test on per-question mean
differences. Under it the +KGret comparison gain (judge +0.038) and the
HippoRAG 2 comparison loss (judge -0.048) are not significant after Holm.

**SPIQA cross-paper question 29** has "..." as its reference answer. DeepSeek
V4 Pro reasoned to the 8,192-token cap on it at every attempt, so a judge
reply that fills the cap is now asked again with reasoning off. The question
is excluded from analysis (49 questions reported).

**PubLayNet leakage.** 3 of the 35 caption-answerable figure questions are
flagged (`artifacts/runs/figure_integrity_publaynet.csv`).

**SPIQA text recoverability (2026-10-23).** `rag audit-figures` over the 150
primary questions flags 15 against the papers' own text and none against the
generated captions: 6 exact answer strings (short method or dataset names)
and 9 where every number in the answer occurs in the paper. The numbers rule
over-flags (one item is flagged for a lone "4"), which errs in the
conservative direction; the funnel therefore has 135 items that are not
text-recoverable. The exact-string check in `evaluation/integrity.py` is a
raw substring test, like containment was; it was left unchanged because the
released question sets were screened with it.

**Graph grounding (2026-10-23).** Both triple endpoints occur in the source
chunk for 79.3% of HotpotQA triples (72,946), 64.4% of SPIQA (50,123), and
29.8% of PubLayNet (36,670, OCR text); outputs in
`artifacts/runs/graph_grounding_*.json`.

**Analysis in the package (2026-10-23).** `rag pooled-analysis` and
`rag holm-family` produce every HotpotQA number in the manuscript and
`docs/paper_tables.py` turns the reports into table rows. Judge agreement
with token-bounded containment over all non-empty HotpotQA answers is 87.5%
(kappa 0.75) on bridge and 78.8% (kappa 0.55) on comparison questions; of
the replies that open by declining, the judge accepted 0.8% and 2.5%.
Re-scoring those 58 replies as wrong moves no reported judge difference by
more than 0.006. `rag judge-calibration` on the 200 HotpotQA questions
(DeepSeek V4 Pro, $0.07): 0 of 200 references rejected and 0 of 200 decoys
accepted (`artifacts/runs/calibration_hotpotqa_*.jsonl`).

**Lean RQ3 runs (2026-11-03 to 2026-11-04, on the desktop).** Four generators
(no Qwen), accuracy judged only: SPIQA screened pixels, flagged pixels, screened
caption arms, and PubLayNet pixel-only figures, $11.18 by the records, no errors
or empty answers. Traces are stamped 20261103_1958, 20261103_2228 to 2245, and
20261104_0102 to 0103; `limit=5` smoke traces stamped 20261103_1935 to 1951 are
not results. `rag multimodal-analysis` writes `artifacts/runs/multimodal_rq3.json`
and, with `--adjudicated`, `multimodal_rq3_adjudicated.json`. Pixels minus
text-only is +0.073 [+0.038, +0.110] on the screened set, +0.038 [+0.002, +0.078]
on the flagged set, and +0.062 [+0.024, +0.106] on the 85 items left after
removing text-recoverable and closed-book-recoverable ones, so H3 is not supported.
`metrics.complete` counts the gold crop among the five scored, while pixels send
one crop and captions three, so what a generator received comes from the evidence
items: 25/150 screened, 12/132 flagged, 8/35 PubLayNet, and 34/150 for captions.

**Second judge (2026-11-04).** DeepSeek V4 Pro on a stratified 20% of the figure
sets (1,248 answers, 96.0%, kappa 0.91) and GPT-6.1 Sol on 20% of HotpotQA (2,600
answers, 98.8%, kappa 0.97), with Claude Opus 5.5 on the 82 disagreements. Its
verdicts move no HotpotQA judge difference by more than 0.004 and no multimodal
test by more than 0.007, but the caption gain's Holm p goes from 0.03 to 0.06.
The `--dry-run` price for an accuracy pass counts the context, which that pass
does not send, so it ran about three times too high on HotpotQA. DeepSeek's records
understate the account's charge ($0.75 recorded against $1.35 charged), which is
not yet explained.

**Not run, waiting for money (balance $0.86):** SPIQA sufficiency, triple
precision, model review of model-authored questions, Qwen on the figure sets,
+both and shuffled, flagged caption arms, faithfulness on the figure sets, repeats.
