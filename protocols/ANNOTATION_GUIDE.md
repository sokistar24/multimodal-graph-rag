# Human validation protocol

Two reviewers independently label every assigned row without seeing model identity. Disagreements are adjudicated only after both independent labels are frozen.

## Output audit

- `answer_correct`: the response is substantively equivalent to the reference answer.
- `answer_faithful`: every material claim is supported by the supplied evidence.
- `non_gold_support`: a retrieved non-gold passage is sufficient to answer the question.
- `closed_book_supported`: the same answer is correct in the paired no-context run.
- `reference_error`: the reference answer or gold provenance is incorrect or underspecified.
- `evidence_state`: choose one state defined in `evaluation/attribution.py`; do not infer parametric memory merely because gold provenance is incomplete.

The validation sample must contain at least 400 outputs stratified across corpora, systems, generators, correctness, and evidence-completeness states. Report raw agreement, Cohen's kappa, judge accuracy, false-positive rate, and false-negative rate before using automated judgements as headline outcomes.

## Graph audit

Sample at least 200 triples stratified by corpus and extraction-frequency band. Independently score triple correctness, source entailment, entity canonicalisation, relation specificity, and provenance correctness. Retain source excerpts for adjudication but do not place copyrighted full documents in the release.

