# Validation without new human annotation

No further human annotation is being carried out for this revision. This
document records what human judgement the study *does* rest on, what replaces
the annotation that will not happen, and what therefore cannot be claimed.

## Human judgement the study already has

All of it is external and already complete; none of it is ours to redo.

- **SPIQA test-A curated QA** (`data/questions/questions_spiqa_native.json`):
  579 items written and answered by SPIQA's curators, each carrying human
  crop-level provenance. 447 pass the caption leakage screen.
- **HotpotQA bridge and comparison**: questions written by HotpotQA's
  annotators, with human-labelled supporting paragraphs as gold provenance.
- **Caption-contaminated SPIQA crops**: five crops identified in the earlier
  integrity check and excluded from every headline visual claim.

Headline claims should rest on these sets. The model-authored sets
(PubLayNet text/multi-hop/figure, SPIQA text/multi-hop/figure, and the
graph-seeded cross-document set) remain useful as mechanism tests but carry
the construction caveats recorded in each question's `construction_method`.

## What replaces the planned annotation

### Judge calibration -> reference-based scoring

Instead of calibrating the LLM judge against fresh human labels, every run now
also reports metrics computed directly against the reference answers, which
were themselves written by the benchmarks' annotators:

- `em` — exact match after the standard normalisation.
- `f1` — token-overlap F1.
- `contains` — the normalised reference occurring inside the answer.

On HotpotQA, whose answers are short spans, exact match and F1 are the
benchmark's own official metrics. **Report them as the primary outcome for the
graph experiments.** The judge is then a secondary, corroborating measure
rather than the load-bearing instrument, and the graph claims stop depending on
an uncalibrated judge entirely.

On SPIQA-native, answers are discursive (median 112 characters), so exact match
is far too strict and the judge is doing real work. Report the judge, report F1
beside it, and report their disagreement rate. A large gap marks exactly where
the judge is load-bearing and where the uncalibrated-judge limitation bites.

### Evidence attribution -> control arms

The audit of correct-but-unsupported answers is replaced by the control arms,
which measure the same thing programmatically for every question rather than
for a sample:

| question | arm |
| --- | --- |
| answerable with no evidence at all? | `control:closed-book` |
| answer survives unrelated context of the same length? | `control:shuffled` |
| answerable when all gold evidence is supplied? | `control:oracle` |
| answerable from one gold document only? | `control:partial-gold` |

Of the four inputs to an evidence-attribution state, three are now measured.
The remaining one — whether a retrieved non-gold passage was *sufficient* — is
not measured, so `EvidenceState.NON_GOLD_SUPPORTED` cannot be assigned
automatically and must not be reported as an established category.

### Graph triple audit -> automated grounding

`rag graph-grounding` checks every triple in the cache against the chunk it was
extracted from, over the whole graph rather than a 200-item sample:

```powershell
rag graph-grounding data/graphs/triples_cache_spiqa_corpus.json `
  --corpus-dir spiqa_corpus --corpus spiqa --collisions
```

It reports how often the subject, the object, both, or neither occur in the
source text, how often the relation string does, how many endpoints are
numeric, and how many distinct surface forms merge into a single node.
`KnowledgeGraph.statistics()` supplies the structural counts alongside it.

**Call this triple grounding, never triple precision.** It establishes that a
triple's endpoints are present in the passage it claims to come from, so a low
rate is strong evidence of extraction error. It does not establish that the
triple is true, that the relation is correct, or that the relation is specific
enough to be useful — a model can join two entities that both appear in a chunk
with a relation the chunk does not support, and this check will pass it.

## What cannot be claimed

State these plainly in the limitations rather than leaving them implicit.

1. The LLM judges are **not calibrated against human labels collected for this
   study**. No agreement rate, kappa, or judge false-positive/false-negative
   rate can be reported. Where reference-based metrics are available they are
   the primary outcome for that reason.
2. Triple **precision, relation correctness, and relation specificity are not
   measured**. Only lexical grounding and structural statistics are.
3. `NON_GOLD_SUPPORTED` is **not an assignable state**; accuracy without
   complete gold provenance is reported as such, with the closed-book and
   shuffled arms bounding how much of it is prior knowledge.
4. The model-authored question sets have **had no human review recorded** in
   this repository. Treat their construction caveats as live.

## Instruments retained but unused

`graph_triple_audit.csv` and `human_output_annotation.csv` remain in this
directory, and `rag audit-graph` still draws a stratified sample, so the audit
can be run later without rebuilding anything. Neither has been completed, and
no result in this repository depends on either.
