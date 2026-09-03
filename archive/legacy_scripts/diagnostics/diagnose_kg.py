"""
diagnose_kg.py — why does +KG make text answers WORSE?

Observed at n=35 on gpt4o-mini (consistent from n=10, so not noise):

    metric          baseline    +KG      delta
    AnswerAcc       0.657       0.600    -0.057
    Faithfulness    0.714       0.571    -0.143   <- the tell
    AnswerRel       0.800       0.743    -0.057

Faithfulness falling hardest means the model is asserting things the evidence
does not support. That is the signature of irrelevant "facts" being injected
into the prompt — the generator treats them as evidence and follows them.

The original 100-page run had +KG identical to baseline on text (0.833/0.833).
Two things changed since: the corpus grew 10x (730 -> 36,513 triples, extracted
from OCR-noisy text), and graph_facts_for_query now matches against 50,515 nodes
instead of a few thousand.

Hypotheses this script tests
────────────────────────────
  H1  OCR garbage — nodes are mangled OCR strings, so the injected facts are
      nonsense. Would explain the faithfulness collapse directly.
  H2  Spurious matching — with 50k nodes, common words ("patients", "treatment")
      match almost any question and drag in unrelated facts from other papers.
      min_len=5 was tuned when the graph was small; it may be too permissive now.
  H3  Cross-document contamination — matched facts come from pages OTHER than
      the one holding the answer. The model then blends facts about a different
      study into its answer.
  H4  Volume — max_facts=8 of mostly-noise crowds out the real context.

Read-only: builds nothing, calls no API, changes nothing.

Usage:
    python diagnose_kg.py
    python diagnose_kg.py --n 10          # inspect more questions
    python diagnose_kg.py --qfile questions_publaynet_multihop.json
"""

import re
import sys
import json
import random
import argparse
from collections import Counter

# Colours
GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[1m", "\033[0m"
)


def _load_dictionary(corpus_dir="publaynet_corpus"):
    """
    Build a vocabulary of words that are real in THIS corpus.

    Why not a character heuristic: vowel-ratio and consonant-run rules fail
    here. 'cheetiyy' is vowel-rich but not a word; 'p70S6K' is a real protein
    name that looks like garbage to any character-level rule.

    Why not only a system dictionary: it exists on Linux but not Windows, and
    it would not contain biomedical vocabulary ('leptospirosis', 'p70S6K').

    So: take a system dictionary if present, and ALWAYS add the corpus's own
    frequent words. A token appearing on many different pages is real
    vocabulary; OCR damage is random and rarely repeats across documents.
    """
    import glob
    import os
    from collections import Counter

    words = set()

    # System dictionary if available (Linux/mac; absent on Windows).
    for path in ("/usr/share/dict/words", "/usr/dict/words"):
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                words |= {w.strip().lower() for w in f if len(w.strip()) > 2}
            break
        except (FileNotFoundError, OSError):
            continue

    # Corpus vocabulary: words appearing on >=3 distinct pages are real.
    # OCR errors are random, so they almost never recur across documents.
    page_counts = Counter()
    files = sorted(glob.glob(os.path.join(corpus_dir, "*.txt")))
    for path in files:
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = f.read().lower()
        except OSError:
            continue
        for tok in set(re.findall(r"[a-z]{3,}", text)):
            page_counts[tok] += 1

    corpus_vocab = {w for w, c in page_counts.items() if c >= 3}
    words |= corpus_vocab

    print(f"{DIM}  vocabulary: {len(words):,} words "
          f"({len(corpus_vocab):,} from the corpus itself){RESET}")
    return words


_DICT = None      # built lazily in main(), needs the corpus path


def looks_like_ocr_garbage(s: str) -> bool:
    """
    True if the string looks like OCR damage rather than real text.

    Rule: skip technical identifiers (digits present, ALL-CAPS acronyms), treat
    mid-word capitals as an OCR signature ('uCnpeiNn'), then check remaining
    words against the vocabulary. Majority unknown -> garbled.

    Conservative by design: returns False when unsure, so we under-report rather
    than cry wolf on real biomedical vocabulary.
    """
    if not _DICT:
        return False
    tokens = re.findall(r"[A-Za-z]{3,}", s)
    if not tokens:
        return False

    # Technical identifiers contain digits — 'p70S6K', 'CD4+', 'IL-6'.
    if any(ch.isdigit() for ch in s):
        return False

    checkable = []
    for t in tokens:
        if t.isupper():                 # acronym: BMI, WHO, EORTC
            continue
        if t[1:].lower() != t[1:]:      # mid-word capitals: 'uCnpeiNn'
            return True                 # a reliable OCR signature
        checkable.append(t.lower())

    if not checkable:
        return False

    unknown = sum(1 for t in checkable if t not in _DICT)
    # Half or more unrecognised -> garbled. (Not >0.5: a two-word string with
    # one nonsense word, e.g. 'Yemnapest grep', should be flagged.)
    return unknown / len(checkable) >= 0.5


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose why +KG hurts.")
    ap.add_argument("--qfile", default="questions_publaynet_text.json")
    ap.add_argument("--corpus-dir", default="publaynet_corpus")
    ap.add_argument("--n", type=int, default=6,
                    help="how many questions to inspect in detail")
    args = ap.parse_args()

    print(f"\n{'=' * 76}")
    print("  KG DIAGNOSTIC — why does +KG reduce accuracy and faithfulness?")
    print(f"{'=' * 76}")

    # Build the vocabulary from the corpus itself — see _load_dictionary.
    global _DICT
    print(f"\n{DIM}Building vocabulary from the corpus...{RESET}")
    _DICT = _load_dictionary(args.corpus_dir)

    # Import late so the failure message is clear if deps are missing.
    from graph_aware import build_triples, build_graph, graph_facts_for_query

    print(f"\n{DIM}Loading cached triples (no extraction, no API calls)...{RESET}")
    triples = build_triples()
    graph   = build_graph(triples)

    with open(args.qfile, encoding="utf-8") as f:
        questions = json.load(f)

    # ── H1: how much of the graph is OCR garbage? ───────────────────────────
    print(f"\n{BOLD}H1 — is the graph full of OCR garbage?{RESET}")
    nodes = list(graph.nodes)
    sample = random.Random(42).sample(nodes, min(400, len(nodes)))
    garbage = [n for n in sample if looks_like_ocr_garbage(n)]
    pct = len(garbage) / len(sample) * 100
    colour = RED if pct > 25 else (YELLOW if pct > 10 else GREEN)
    print(f"  nodes total        : {len(nodes):,}")
    print(f"  sampled            : {len(sample)}")
    print(f"  look like garbage  : {colour}{len(garbage)} ({pct:.0f}%){RESET}")
    if garbage:
        print(f"  {DIM}examples: {', '.join(repr(g[:28]) for g in garbage[:5])}{RESET}")

    # ── H2: which nodes are match magnets? ──────────────────────────────────
    print(f"\n{BOLD}H2 — do common nodes match almost every question?{RESET}")
    print(f"  {DIM}min_len=5 was tuned on a small graph; with "
          f"{len(nodes):,} nodes it may be too permissive{RESET}")
    match_counts = Counter()
    for item in questions:
        ql = item["q"].lower()
        for node in nodes:
            if len(node) < 5:
                continue
            if re.search(r"\b" + re.escape(node) + r"\b", ql):
                match_counts[node] += 1

    magnets = [(n, c) for n, c in match_counts.most_common(12)
               if c >= max(2, len(questions) * 0.1)]
    if magnets:
        print(f"  {YELLOW}nodes matching many different questions:{RESET}")
        for node, count in magnets:
            deg = graph.in_degree(node) + graph.out_degree(node)
            print(f"    {count:>3}/{len(questions)} questions  "
                  f"{node[:36]:<38} {DIM}(degree {deg}){RESET}")
        print(f"  {DIM}each match pulls in that node's facts — from ANY paper "
              f"in the corpus{RESET}")
    else:
        print(f"  {GREEN}no obvious match magnets{RESET}")

    # ── H3/H4: what actually gets injected? ─────────────────────────────────
    print(f"\n{BOLD}H3/H4 — what facts are actually injected, and from where?{RESET}")

    n_with_facts = 0
    n_facts_total = 0
    n_garbage_facts = 0
    n_offsource = 0

    for item in questions:
        facts = graph_facts_for_query(item["q"], graph)
        if facts:
            n_with_facts += 1
            n_facts_total += len(facts)
            n_garbage_facts += sum(1 for f in facts if looks_like_ocr_garbage(f))

    print(f"  questions receiving facts : {n_with_facts}/{len(questions)} "
          f"({n_with_facts/len(questions)*100:.0f}%)")
    if n_with_facts:
        print(f"  mean facts per question   : {n_facts_total/n_with_facts:.1f} "
              f"{DIM}(cap is max_facts=8){RESET}")
        gpct = n_garbage_facts / n_facts_total * 100 if n_facts_total else 0
        colour = RED if gpct > 25 else (YELLOW if gpct > 10 else GREEN)
        print(f"  facts that look garbled   : {colour}{n_garbage_facts}"
              f"/{n_facts_total} ({gpct:.0f}%){RESET}")

    # ── The detail: show real examples ──────────────────────────────────────
    print(f"\n{BOLD}Sample — exactly what the +KG prompt receives{RESET}")
    shown = 0
    for item in questions:
        if shown >= args.n:
            break
        facts = graph_facts_for_query(item["q"], graph)
        if not facts:
            continue
        shown += 1
        print(f"\n  {'-' * 72}")
        print(f"  Q      : {item['q'][:70]}")
        print(f"  ANSWER : {item['answer'][:70]}")
        print(f"  SOURCE : {item['source']}")
        print(f"  {DIM}injected facts ({len(facts)}):{RESET}")
        for fact in facts:
            bad = looks_like_ocr_garbage(fact)
            mark = f"{RED}garbled{RESET}" if bad else f"{DIM}ok{RESET}"
            print(f"    [{mark}] {fact[:82]}")

    # ── Verdict ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 76}")
    print(f"{BOLD}  READ THE FACTS ABOVE{RESET}")
    print(f"{'=' * 76}")
    print(f"""
  Ask of each sample: would these facts help a model answer that question?

  If they are garbled or come from unrelated papers, the faithfulness drop is
  explained — the model is being handed noise labelled as evidence, and it
  follows it. That is a REPORTABLE FINDING, not merely a bug: it is the
  "OCR noise degrades KG-entity quality" limitation from the original report,
  now measured at 10x scale.

  Levers, if a fix is wanted:
    • raise min_len in graph_facts_for_query  (fewer spurious matches)
    • lower max_facts from 8                  (less crowding of real context)
    • filter garbled nodes at build_graph time
    • restrict facts to the retrieved pages   (kills cross-document bleed)

  Note: any change alters the +KG condition for ALL FOUR generators, so it must
  be decided before the remaining runs — not after.
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
