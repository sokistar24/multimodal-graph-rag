# Archive

Superseded material kept for provenance. Nothing here is executed, imported,
or eligible for a release manifest; `rag validate-release` rejects any run
path under this directory.

- `invalid_runs/prefix_bug/` — results produced before the source-prefix fix.
  Known invalid; retained only so the correction is auditable.
- `legacy_scripts/diagnostics/` — one-off diagnostic programs from the first
  submission cycle.
- `legacy_scripts/question_generation/` — the original 100-page PubLayNet
  question generators.
- `legacy_scripts/superseded_by_package/` — the root-level scripts whose logic
  now lives in `src/multimodal_graph_rag/` (ingestion, question authoring,
  the evaluation runner, result collection, tables, figures, audits). Kept
  verbatim so behaviour can be compared; they depend on modules that no
  longer exist and will not run.
- `logs/legacy/` — execution logs from the question-generation runs.
- `old_submissions/` — the earlier report and the 100-page question sets.
- `environment/` — the `pip freeze` captured from the environment used for the
  released runs (UTF-16 encoded, as exported).
