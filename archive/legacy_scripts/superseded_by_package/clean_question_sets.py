"""
Removes items violating the new validators (container references, source-id
leaks, over-length, orphan references, modality leaks) from already-generated
question files, so generate_questions.py can be re-run to top the sets back up
to their targets (it is resumable and skips pages already used).

Usage:
    python clean_question_sets.py questions_spiqa_text.json questions_spiqa_multihop.json
"""

import json
import sys

from generate_questions import (
    MAX_Q_CHARS,
    bad_reference,
    has_orphan_reference,
    leaks_modality,
)

for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    kept, removed = [], []
    for item in items:
        q = item["q"]
        reason = (
            "container/source-id"
            if bad_reference(q)
            else "over-length"
            if len(q) > MAX_Q_CHARS
            else "orphan reference"
            if has_orphan_reference(q)
            else "modality leak"
            if leaks_modality(q)
            else None
        )
        (removed if reason else kept).append((item, reason))
    with open(path, "w", encoding="utf-8") as f:
        json.dump([i for i, _ in kept], f, indent=2, ensure_ascii=False)
    print(f"{path}: kept {len(kept)}, removed {len(removed)}")
    for item, reason in removed:
        print(f"  [-] ({reason}) {item['q'][:90]}")
