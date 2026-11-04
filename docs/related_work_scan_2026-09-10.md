# Related-work scan for the journal revision (2025-01 to 2026-09)

Scan date: 2026-09-10. Scope: papers not already in `paper/references.bib` (47 entries) that the
revised arXiv 2607.16604 must cite or contrast with. Every entry below was located by web search and
its arXiv abstract page was fetched; venue lines are taken from the arXiv "Comments" field or the
ACL Anthology / ACM DL page named in the row. Labels: **must** / should / optional.

## Novelty verdict

No paper found in the window runs the same design as ours: a single controlled document-QA
evaluation that (i) crosses graph augmentation with multimodal (figure/table) augmentation on one
corpus and one generator set, (ii) separates prompt-side graph injection from retrieval-stage graph
expansion, (iii) attributes every answer to an evidence state (unavailable / retrieval failure /
incomplete / complete / evidence-use failure / non-gold-supported / closed-book / unsupported) with
closed-book, shuffled-context, oracle and partial-gold arms, and (iv) holds retrieval budget fixed
across BM25, RRF, lexical entity expansion, random padding and HippoRAG 2. The closest three are:
**MegaRAG** (2512.20626, ACL 2026) and **RAG-Anything** (2510.12323), which both *build* a
multimodal knowledge graph and report headline gains over RAG baselines, but neither isolates the
graph contribution from the visual contribution under a fixed budget nor includes closed-book or
leakage controls; and **GraphRAG-Bench / "When to use Graphs in RAG"** (2506.05690, ICLR 2026),
which is a controlled text-only GraphRAG-vs-RAG study with pipeline-stage analysis but no
multimodal arm, no evidence-state attribution and no caption-leakage check. The already-cited
2606.28780 (Multimodal Graph RAG for VRDU) and MKG-RAG-Bench (2606.26458) remain system/benchmark
papers rather than controlled attributions. On the *evaluation-methodology* side the nearest
neighbours are Pair-ID (2608.08944, paired add-support/delete-nonsupport interventions on a fixed
retrieval state) and the facet-level "evidence override vs evidence failure" tracing (2604.09174);
both are text-only and neither crosses graph or visual augmentation. Novelty therefore rests on the
combination and on the attribution framework, and the revision must explicitly position against
these six.

Must-cite count: **17** (4 + 3 + 1 + 2 + 3 + 2 + 2 across the seven themes).

## 1. Evidence attribution, provenance and lineage in RAG

| id | title | venue | contrast with our paper | label |
|---|---|---|---|---|
| 2608.16004 | LineageRAG: Harnessing GraphRAG by Constructing Evidence Lineages with Source Grounding | arXiv (cs.IR), Aug 2026 | A *system*: initialises query-derived "evidence demands", expands one lineage per demand by demand-conditioned retrieval over a corpus graph, then grounds each supported demand in a verbatim source span; reports R@5/EM/F1 gains on HotpotQA/2Wiki/MuSiQue. Ours is an *evaluation* framework: lineage/provenance is used to classify outcomes into evidence states, not to improve retrieval; ours is document-level and multimodal, theirs text-only Wikipedia multi-hop. Name overlap ("evidence lineage") must be disambiguated in the text. | must |
| 2608.08944 | What Would Fix This RAG Failure? Auditing Counterfactual Response with Paired Evidence Interventions (Pair-ID) | arXiv (cs.IR), Aug 2026 | Holds query, retrieval state and reader fixed and crosses "add missing support" with "delete verified non-support" over ~20k queries; finds repair rates only partially predictable from the observed failure. Closest methodological neighbour to our oracle / partial-gold arms; text-only, no graph or visual axis. | must |
| 2604.09174 | Facet-Level Tracing of Evidence Uncertainty and Hallucination in RAG | arXiv, Apr 2026 (v2 May) | Decomposes questions into atomic facets and builds a facet x chunk relevance/faithfulness matrix under Strict-RAG / Soft-RAG / LLM-only modes; reports "evidence override" (28.4%) as 4x more common than "evidence failure" (7%). Maps directly onto our evidence-use failure vs retrieval failure states, but is text QA (medical, HotpotQA) with no retrieval-budget control. | must |
| 2411.06037 | Sufficient Context: A New Lens on Retrieval Augmented Generation Systems (Joren et al.) | ICLR 2025 | Classifies instances by whether context is sufficient and stratifies errors accordingly; larger models answer wrongly rather than abstain when context is insufficient. Direct ancestor of evidence-conditioned accuracy; no closed-book, shuffled or graph arms. Cite even though arXiv is Nov 2024, since the published version is 2025. | must |
| 2608.20627 | When Failures Propagate: Causal Failure Attribution in Agentic RAG (AgenticRAG-FP) | arXiv, Aug 2026 | Injects faults at specific hops of 3-hop agentic RAG and asks whether coverage-based vs counterfactual diagnosis can localise them (coverage 0.91 at hop 1, 0.00 deeper). Complements our single-shot attribution; small (80 questions), one model. | should |
| 2607.09349 | Deceptive Grounding: Entity Attribution Failure in Clinical RAG | arXiv, Jul 2026 | Defines a failure invisible to faithfulness/citation checks: evidence is real but about the wrong entity. Sharpens our "non-gold-supported" state; clinical, text-only. | should |
| 2605.14192 | Why Retrieval-Augmented Generation Fails: A Graph Perspective | arXiv, May 2026 | Mechanistic: circuit-tracing attribution graphs distinguish correct (deep, question-constrained) from failed (shallow, context-dominated) routing. Different level of analysis from our behavioural states. | optional |
| 2605.01284 | Chain of Evidence: Pixel-Level Visual Attribution for Iterative RAG | arXiv, May 2026 | Attribution as bounding boxes on document screenshots via a fine-tuned VLM. Relevant to figure-question provenance; it is a method, not a controlled comparison. | optional |

## 2. Controlled GraphRAG vs vanilla RAG, equal-budget or fixed-subset protocols

| id | title | venue | contrast with our paper | label |
|---|---|---|---|---|
| 2506.05690 | When to use Graphs in RAG: A Comprehensive Analysis for Graph RAG (GraphRAG-Bench) | ICLR 2026 | Benchmark across fact retrieval, complex reasoning, summarisation and creative generation with stage-wise (construction/retrieval/generation) analysis of when graphs help. Text-only; no multimodal arm; no evidence-state attribution; no matched budget vs lexical expansion or random padding. | must |
| 2604.09666 | Do We Still Need GraphRAG? Benchmarking RAG and GraphRAG for Agentic Search Systems (RAGSearch) | arXiv, Apr 2026 | Unifies datasets, backbones, retrieval budgets and protocols to compare dense RAG vs GraphRAG under static and agentic (incl. RL) search; agentic search narrows the gap, GraphRAG keeps an edge on multi-hop and amortised cost. Shares our equal-budget stance; text-only, no attribution states. | must |
| 2502.14802 | From RAG to Memory: Non-Parametric Continual Learning for LLMs (HippoRAG 2) | ICML 2025 | Our retrieval-stage graph comparator. Must be cited as such; it claims to remove HippoRAG's factual-memory regression relative to dense RAG, which our equal-budget results can confirm or qualify. | must |
| 2503.04338 | In-depth Analysis of Graph-based RAG in a Unified Framework | arXiv, Mar 2025 (v2 Apr 2026) | Re-implements many graph-RAG variants in one framework and evaluates on specific-to-abstract QA. Component-level comparison, no budget matching or controls. | should |
| 2506.02404 | GraphRAG-Bench: Challenging Domain-Specific Reasoning for Evaluating Graph RAG (Xiao et al.) | arXiv, Jun 2025 | Textbook-derived college-level multi-hop benchmark evaluating nine GraphRAG methods end-to-end. Distinct from 2506.05690 despite the shared name; cite both to avoid confusion. | should |
| 2608.02195 | MEGRAG: Multi-Granular Evidence Graphs for Answer-Aware Multi-Hop RAG | arXiv, Aug 2026 | Iterative system linking passages to sentences and triples in a cross-granularity index with answer-aware stopping. A method, not a controlled evaluation; no budget parity or evidence attribution. Contrast for retrieval-stage graph expansion. | optional |
| 2507.03226 | Towards Practical GraphRAG: Efficient KG Construction and Hybrid Retrieval at Scale | arXiv, Jul 2025 (v3 Dec) | Dependency-parse triples reach 94% of LLM-extraction quality; hybrid retrieval fuses vector and graph scores with RRF. Relevant to our RRF comparator and to KG cost; enterprise data, no controls. | optional |

## 3. Parametric knowledge vs retrieved evidence: closed-book, shuffled and counterfactual controls

| id | title | venue | contrast with our paper | label |
|---|---|---|---|---|
| 2605.27105 | Lost in the Evidence? Reproducing Document Position and Context Size Effects in RAG | SIGIR 2026 (reproducibility) | 1,500 open-domain questions; varies depth with original rank, reversed rank and random shuffle, and contrasts retriever-mediated with oracle access; shows idealised (oracle) conclusions do not transfer. Justifies our shuffled-context and oracle arms; text-only. | must |
| 2605.14473 | Does RAG Know When Retrieval Is Wrong? Diagnosing Context Compliance under Knowledge Conflict | arXiv, May 2026 | Frames "did the model follow evidence, its prior, or rationalise post hoc" as an observability problem; misconception-injection drops standard RAG to 15%. Motivates our closed-book and evidence-use states; no document QA. | should |
| 2605.28721 | LiveBrowseComp: Are Search Agents Searching, or Just Verifying What They Already Know? | arXiv, May 2026 | Agents answer up to 44.5% of BrowseComp closed-book; a fresh-fact benchmark drops closed-book accuracy below 2%. Same argument as our closed-book control for contamination; web search agents rather than document RAG. | should |
| 2606.29645 | Metadata, Structure, or Strategy? A Decomposition of RAG Context Enrichment | ECML-PKDD 2026 | Controlled decomposition across six benchmarks / four models: most prompt-side enrichment *reduces* accuracy; a "processability hierarchy" predicts which metadata models can use. Direct parallel to our prompt-side graph injection result; text metadata, not graphs or figures. | should |
| 2604.25313 | Faithfulness-QA: A Counterfactual Entity Substitution Dataset for Training Context-Faithful RAG Models | arXiv, Apr 2026 | 99k counterfactual-substitution samples from SQuAD/TriviaQA to train context faithfulness. Training data, not evaluation; cite as background on the prior-vs-context tug of war. | optional |
| 2506.20051 | Controlled Retrieval-augmented Context Evaluation for Long-form RAG (CRUX) | arXiv, Jun 2025 (v2 Jan 2026) | Coverage-based, question-driven evaluation of retrieved context against human summaries for report generation. "Controlled context" idea, but long-form and text-only. | optional |
| 2503.15888 | Parameters vs. Context: Fine-Grained Control of Knowledge Reliance in Language Models (CK-PLUG) | arXiv, Mar 2025 | Tunes reliance on parametric vs contextual knowledge in counterfactual RAG. Method paper; background only. | optional |

## 4. Multimodal document RAG evaluation, contamination and caption leakage

| id | title | venue | contrast with our paper | label |
|---|---|---|---|---|
| 2601.08620 | ViDoRe V3: A Comprehensive Evaluation of RAG in Complex Real-World Scenarios | ACL 2026 (long) | 10 datasets, ~26k pages, 3,099 human-verified queries with relevance, bounding-box and answer annotations; visual retrievers beat textual, hybrid context helps generation, models still fail on non-textual elements. No caption-answerable vs pixel-only split and no graph arm. | must |
| 2508.03644 | Are We on the Right Way for Assessing Document RAG? (Double-Bench) | arXiv, Aug 2025 (in submission) | 3.2k documents / 72k pages / 5,168 queries with exhaustively verified evidence pages; finds the text-visual embedding gap narrowing and an "over-confidence dilemma" (answers without evidence). The latter is our "unsupported" state at benchmark scale; no graph axis. | must |
| 2602.17687 | IRPAPERS: A Visual Document Benchmark for Scientific Retrieval and QA | arXiv, Feb 2026 | 166 scientific papers, 3,230 pages; text and image retrieval tie on recall, hybrid wins, and text-based QA aligns better with ground truth (0.82 vs 0.71). Same domain as ours; needle-in-haystack questions, no figure-only protocol. | should |
| 2502.14864 | Benchmarking Multimodal RAG through a Chart-based Document QA Generation Framework (Chart-MRAG Bench) | arXiv, Feb 2025 | 4,738 chart QA pairs; MLLMs reach only 58% even with perfect retrieval and show "text-over-visual modality bias". Supports our finding that generator figure-reading, not only retrieval, caps accuracy. | should |
| 2601.15487 | MiRAGE: A Multiagent Framework for Generating Multimodal Multihop QA Datasets for RAG Evaluation | arXiv, Jan 2026 (submitted to ACL) | Agentic generation of verified multimodal multi-hop QA; ablations show it "can be powered by LLMs if textual descriptions of the images are available" while "visual grounding remains a frontier". That is the caption-leakage risk we test explicitly; MiRAGE does not audit it in the resulting datasets. | should |
| 2511.16654 | Comparison of Text-Based and Image-Based Retrieval in Multimodal RAG LLM Systems | arXiv, Nov 2025 | Direct image embeddings beat caption-summary indexing by 13 mAP@5 on financial charts. Retrieval-only comparison, supports our CLIP-vs-caption discussion. | optional |
| 2605.15019 | From Scenes to Elements: Multi-Granularity Evidence Retrieval for Verifiable Multimodal RAG (GranuRAG) | arXiv, May 2026 | Element-level retrieval with attribution-constrained generation on a landmark VQA benchmark. Verifiability framing similar to ours; not documents. | optional |
| 2601.17644 | Do Multimodal RAG Systems Leak Data? Membership Inference and Image Caption Retrieval Attacks | ACL 2026 Findings | Privacy sense of "leakage" (caption extraction attacks). Cite only to disambiguate our use of caption leakage. | optional |
| 2608.30163 | Doc-REFRAG: Rethinking Multimodal Document RAG | EMNLP 2026 main | DocLongRAG (343k QA, ~37 images/query) and RL-selected visual-token expansion. Scale/efficiency, not controlled evaluation. | optional |
| 2604.16313 | MARA: A Multimodal Adaptive Retrieval-Augmented Framework for Document QA | arXiv, Feb 2026 | Query-aligned region encoder plus self-reflective evidence controller; six benchmarks. System paper. | optional |
| 2605.22829 | LFRAG: Layout-oriented Fine-grained RAG on Multimodal Document Understanding | arXiv, Apr 2026 | Block-level retrieval units from layout segmentation (cf. our PubLayNet layout regions). System paper. | optional |

## 5. LLM-as-judge validation for RAG

| id | title | venue | contrast with our paper | label |
|---|---|---|---|---|
| 2606.19544 | Reliability without Validity: A Systematic, Large-Scale Evaluation of LLM-as-a-Judge Models Across Agreement, Consistency, and Bias | arXiv, Jun 2026 | 21 judges, 541k judgments; raw agreement overstates kappa by 33-41 pp; judge rankings unstable across sets; high test-retest with position bias. Sets the bar for reporting our two-judge agreement as kappa, not raw accuracy. | must |
| 2606.00093 | Agreement Metrics for LLM-as-Judge Evaluation: What to Report and Why | arXiv, May 2026 | Shows protocol choices (scale, abstention, pooling) move reported accuracy from 0.55 to 0.90 with identical verdicts; provides a reporting checklist. Our judge-agreement section should follow it. | must |
| 2410.20833 | LLMs are Biased Evaluators But Not Biased for Retrieval Augmented Generation | ACL 2025 Findings | Finds no significant self-preference in RAG reranking/generation judgments across NQ/MARCO/TriviaQA and five models; factual accuracy dominates. Directly supports using a generator-family model as judge, with caveat that our judged outputs are longer free-form answers. | must |
| 2510.09738 | Judge's Verdict: A Comprehensive Analysis of LLM Judge Capability Through Human Agreement | arXiv, Oct 2025 (ICLR 2026 submission) | 54 judges scoring RAG/agentic answers vs ground truth; correlation filter then Cohen's kappa human-likeness test. Same task as our judge; we can cite its judge tiers when choosing models. | should |
| 2604.22891 | Quantifying and Mitigating Self-Preference Bias of LLM Judges | arXiv, Apr 2026 | Equal-quality pairs to separate discriminability from self-preference across 20 LLMs; multi-dimensional rubric cuts bias 31.5%. Background for the self-preference caveat. | should |
| 2605.27789 | A Fixed-Budget, Cluster-Aware Standard for LLM-as-a-Judge Evaluation: A Multi-Hop RAG Stress Test | arXiv, May 2026 | Argues judge comparisons need fixed evidence budgets and cluster-aware tests; with clustering only one result survives Bonferroni. Parallels our equal-budget and per-document clustering concerns. | should |
| 2608.18091 | Self- and Other-Labels Induce Bidirectional Bias in LLM Judges | EMNLP 2026 Findings | Self-preference vanishes under blind conditions but appears with authorship labels. Supports blinding judge prompts to system identity. | optional |

## 6. Knowledge-graph quality for LLM-extracted triples in GraphRAG

| id | title | venue | contrast with our paper | label |
|---|---|---|---|---|
| 2510.14271 | Less is More: Denoising Knowledge Graphs for RAG (DEG-RAG) | arXiv, Oct 2025 | First systematic study of entity resolution plus triple reflection on LLM-built graphs; smaller graphs improve QA across graph-RAG variants. Explains one route by which our prompt-side triples could hurt; we do not denoise. | must |
| 2607.03447 | TRIAGE: Trustworthy Retrieval Instrumentation And Graph Evaluation | arXiv, Jul 2026 | Instruments KG construction (triple confidence, source coverage, canonicalisation), structure and usage (retrieval coverage, faithfulness) as a chain of necessary conditions whose first broken link localises failure. Graph-side analogue of our evidence-state attribution; no QA controls. | must |
| 2605.21974 | Format-Constraint Coupling in Knowledge Graph Construction from Statistical Tables (CSVFidelity-Bench) | arXiv, May 2026 (ARR/EMNLP 2026) | Documents that MS GraphRAG, LightRAG, HippoRAG, nano-GraphRAG and RAG-Anything evaluate by downstream QA only and none measures graph fidelity; releases 1,892 gold facts. Use to justify reporting triple-quality metrics. | should |
| 2605.28004 | Beyond Chunk-Local Extraction: Cross-Chunk Graph Augmentation for GraphRAG (CrossAug) | arXiv, May 2026 | Missing cross-chunk relations are a systematic gap in chunk-local extraction; GNN-guided completion. Relevant to why graph expansion adds little on single-passage questions. | optional |
| 2601.01844 | Clinical Knowledge Graph Construction and Evaluation with Multi-LLMs via RAG | arXiv, Jan 2026 | Two-tier EAV-triple evaluation (coverage, correctness, hallucination) without gold data. Domain-specific. | optional |
| 2607.00003 | From "Strings" to "Things" for Personal KGs: Evaluating LLM Triple Extraction | arXiv, Apr 2026 | Small-model triple extraction; downstream utility not proportional to extraction F1. Background. | optional |

## 7. Graph plus multimodal augmentation in one document-QA evaluation (novelty check)

| id | title | venue | contrast with our paper | label |
|---|---|---|---|---|
| 2512.20626 | MegaRAG: Multimodal Knowledge Graph-Based RAG | ACL 2026 | Injects visual cues into KG construction, retrieval and generation; beats RAG baselines on textual and multimodal corpora. Same authors as cited 2606.28780. Combines both axes but as a single fused system: no graph-only / visual-only factorial, no budget parity, no closed-book or leakage arms. | must |
| 2510.12323 | RAG-Anything: All-in-One RAG Framework | arXiv, Oct 2025 | Dual-graph (cross-modal relations plus textual semantics) over text, images, tables, equations with hybrid structural-semantic retrieval. Abstract reports no controlled ablation separating graph from modality contribution. Closest "graph + multimodal document RAG" system; ours is the controlled evaluation it lacks. | must |
| 2507.20804 | MMGraphRAG: Bridging Vision and Language with Interpretable Multimodal Knowledge Graphs | arXiv, Jul 2025 (v3 Jul 2026) | Fuses visual scene graphs with text KGs via spectral cross-modal entity linking; evaluated on DocBench and MMLongBench. Interpretability via graph, not evidence-state attribution. | should |
| 2606.15906 | MAGE-RAG: Multigranular Adaptive Graph Evidence for Agentic Multimodal RAG in Long-Document QA | arXiv, Jun 2026 | Offline layout/section/semantic evidence graph plus an online controller that opens, searches and prunes under explicit budgets on LongDocURL and MMLongBench-Doc. Budgeted, graph-structured and multimodal, but a system with standard accuracy reporting. | should |
| 2508.05318 | mKG-RAG: Leveraging Multimodal Knowledge Graphs in RAG for Knowledge-intensive VQA | SIGIR 2026 | MMKG extraction from multimodal documents with dual-stage retrieval for encyclopedic VQA. Not document QA over figures; cite as the VQA-side counterpart. | should |
| 2602.12735 | VimRAG: Navigating Massive Visual Context in RAG via Multimodal Memory Graph | arXiv, Feb 2026 | DAG over agent states and visual evidence with RL credit assignment. Agentic, not controlled. | optional |
| 2601.07329 | BayesRAG: Probabilistic Mutual Evidence Corroboration for Multimodal RAG | arXiv, Jan 2026 | Dempster-Shafer fusion of text-image candidates by cross-modal and layout consistency. Fusion method for our late-fusion baseline discussion. | optional |

## Already-cited entries that now have a published version (update bibtex)

| bib key | current entry | update to |
|---|---|---|
| `ragvsgraphrag` | arXiv 2502.11371 | Proceedings of the 32nd ACM SIGKDD Conference (KDD '26), V.2, doi 10.1145/3770855.3817575 (dl.acm.org listing). |
| `guo2024lightrag` | arXiv 2410.05779 | Findings of EMNLP 2025, ACL Anthology 2025.findings-emnlp.568. |
| `faysse2024colpali` | arXiv 2407.01449 | ICLR 2025 (proceedings.iclr.cc, paper hash 99e9e141...). |
| `cho2024m3docrag` | arXiv 2411.04952 | ICCV 2025 Workshops (Findings) under the title "M3DocVQA: Multi-modal Multi-page Multi-document Understanding"; verify whether to cite the workshop paper or keep the arXiv title. |
| `wang2025pixelrag` | arXiv, id TODO | arXiv 2606.28344 (June 2026); replace the TODO note. |
| `peng2024graphragsurvey` | arXiv 2408.08921 | ACM Transactions on Information Systems, vol. 44, no. 2, doi 10.1145/3777378; confirm year/pages on the ACM DL page. |
| `dong2025mmdocrag` | @article with note | Convert to @inproceedings, NeurIPS 2025 Datasets and Benchmarks Track (note already states acceptance). |
| `wang2026mkgragbench` | @article with note | Convert to @inproceedings, KDD 2026 (note already states acceptance). |
| `edge2024graphrag` | arXiv 2404.16130 | No published venue found (v2 Feb 2025); keep as arXiv. |
| `hipporag` | NeurIPS 2024 | Unchanged; add `2502.14802` (HippoRAG 2, ICML 2025) as a separate entry since it is a comparator. |

## Coverage notes

Searches were run per theme with several phrasings (evidence attribution / provenance / lineage;
GraphRAG controlled evaluation / equal budget; closed-book / shuffled / counterfactual context;
multimodal document RAG benchmarks and caption leakage; LLM-judge agreement and self-preference;
LLM-built KG quality; graph plus multimodal document QA). Searches for a combined graph x
multimodal x document-QA *controlled* study returned only our own arXiv page and the system papers in
Section 7. Not found in the window: any paper that measures caption-answerable vs pixel-only figure
accuracy, or that reports evidence-conditioned accuracy under matched retrieval budgets for both a
graph and a visual arm.
