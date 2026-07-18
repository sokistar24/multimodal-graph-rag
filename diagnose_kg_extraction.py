"""
diagnose_kg_extraction.py — is the KG failing at EXTRACTION or at RETRIEVAL?

The question this settles
─────────────────────────
+KG does not help. Two hypotheses are already dead:
  x  OCR garbage            — only 2% of injected facts were garbled
  x  cross-document leakage — restricting facts to retrieved pages: no recovery

Measured (gpt4o-mini, n=35 text / n=30 multi-hop):
    text       baseline 0.657  ->  +KG 0.571   (acc)
    multi-hop  baseline 0.433  ->  +KG 0.400   (acc)

Multi-hop is where the KG exists to help, and it does nothing. So either:

  A) EXTRACTION never captured the answering fact.
     Suspect: fixed-size 500-char chunking cuts mid-sentence, so a relation
     spanning a chunk boundary is never seen whole by the extractor.
     "The only vaccine used in China for leptospirosis is the multivalent
      inactivated vaccine"  ->  split across two chunks  ->  triple never formed.
     If so, the fix is semantic chunking + re-extraction (real cost).

  B) RETRIEVAL has the fact but never surfaces it.
     graph_facts_for_query matches whole-word node names against the question,
     so it can only find facts whose SUBJECT/OBJECT is literally named in the
     question. If so, the fix is in matching — no re-extraction needed.

This script decides which, per question:

  1. Locate the chunks belonging to the question's source page.
  2. Pull the cached triples for exactly those chunks.
  3. Ask: does any triple contain the reference answer?         -> extraction OK?
  4. Ask: does graph_facts_for_query actually return it?         -> retrieval OK?

  answer in triples & returned      -> KG works, something else is wrong
  answer in triples, NOT returned   -> (B) RETRIEVAL is the bottleneck
  answer NOT in triples             -> (A) EXTRACTION is the bottleneck

It also measures boundary damage directly: what fraction of chunks start or end
mid-sentence, and whether answer-bearing sentences get severed.

Read-only. No API calls. Uses the existing triples_cache.json.

Usage:
    python diagnose_kg_extraction.py
    python diagnose_kg_extraction.py --qfile questions_publaynet_multihop.json
    python diagnose_kg_extraction.py --n 8      # show more worked examples
"""

import os
import re
import sys
import json
import glob
import argparse

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[1m", "\033[0m"
)
TICK, CROSS, WARN = f"{GREEN}✓{RESET}", f"{RED}✗{RESET}", f"{YELLOW}!{RESET}"


def normalise(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — for loose matching."""
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()


def content_words(s: str) -> set:
    """Content words of a string, minus stopwords. Used for partial matching."""
    stop = {"the", "a", "an", "of", "in", "for", "to", "and", "or", "is", "was",
            "are", "were", "with", "by", "at", "on", "that", "this", "it", "as",
            "from", "be", "been", "has", "have", "had", "which", "who", "used"}
    return {w for w in normalise(s).split() if w not in stop and len(w) > 2}


def answer_in_text(answer: str, text: str, threshold: float = 0.6) -> bool:
    """
    Is the answer present in the text?

    Exact substring first; otherwise require most of the answer's content words
    to appear. Loose on purpose — we want to know whether the FACT is there at
    all, not whether the phrasing matches.
    """
    na, nt = normalise(answer), normalise(text)
    if na and na in nt:
        return True
    aw = content_words(answer)
    if not aw:
        return False
    tw = content_words(text)
    return len(aw & tw) / len(aw) >= threshold


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Decide whether the KG fails at extraction or retrieval.")
    ap.add_argument("--qfile", default="questions_publaynet_text.json")
    ap.add_argument("--corpus-dir", default="publaynet_corpus")
    ap.add_argument("--cache", default="triples_cache.json")
    ap.add_argument("--n", type=int, default=5, help="worked examples to show")
    args = ap.parse_args()

    print(f"\n{'=' * 78}")
    print("  KG DIAGNOSTIC — is the failure in EXTRACTION or RETRIEVAL?")
    print(f"{'=' * 78}")

    if not os.path.exists(args.cache):
        print(f"\n{RED}{args.cache} not found — run compare_all.py first.{RESET}")
        return 1

    from rag_basics import load_file, chunk_text
    from graph_aware import _chunk_key, build_triples, build_graph, graph_facts_for_query

    with open(args.cache, encoding="utf-8") as f:
        cache = json.load(f)
    with open(args.qfile, encoding="utf-8") as f:
        questions = json.load(f)

    # ── Map each source page -> its chunks and its cached triples ────────────
    print(f"\n{DIM}Mapping pages to chunks and cached triples...{RESET}")
    page_chunks, page_triples = {}, {}
    for path in sorted(glob.glob(os.path.join(args.corpus_dir, "*"))):
        src = os.path.basename(path)
        chunks = list(chunk_text(load_file(path)))
        page_chunks[src] = chunks
        triples = []
        for c in chunks:
            triples.extend(cache.get(_chunk_key(c), []))
        page_triples[src] = triples

    # ── Chunk boundary damage: the mechanism behind hypothesis (A) ───────────
    print(f"\n{BOLD}Chunk boundary damage{RESET}")
    print(f"  {DIM}fixed-size chunking cuts blind; a relation split across two "
          f"chunks is never seen whole by the extractor{RESET}")
    all_chunks = [c for cs in page_chunks.values() for c in cs]
    if all_chunks:
        starts_mid = sum(1 for c in all_chunks
                         if c and not c.lstrip()[:1].isupper())
        ends_mid = sum(1 for c in all_chunks
                       if c.rstrip()[-1:] not in ".!?" if c.strip())
        print(f"  chunks total          : {len(all_chunks):,}")
        print(f"  start mid-sentence    : {starts_mid:,} "
              f"({starts_mid/len(all_chunks)*100:.0f}%)")
        print(f"  end mid-sentence      : {ends_mid:,} "
              f"({ends_mid/len(all_chunks)*100:.0f}%)")

    # ── The core test, per question ─────────────────────────────────────────
    graph = build_graph(build_triples(with_sources=True))

    n_ans_in_page      = 0    # the page genuinely contains the answer
    n_ans_in_triples   = 0    # extraction captured it
    n_ans_retrieved    = 0    # graph_facts_for_query actually returned it
    n_no_triples       = 0
    examples = []

    for item in questions:
        src, answer, q = item["source"], item["answer"], item["q"]
        if src not in page_chunks:
            continue

        page_text = " ".join(page_chunks[src])
        in_page   = answer_in_text(answer, page_text)
        if in_page:
            n_ans_in_page += 1

        triples = page_triples.get(src, [])
        if not triples:
            n_no_triples += 1
        triple_text = " ".join(" ".join(t) for t in triples)
        in_triples  = answer_in_text(answer, triple_text) if triples else False
        if in_triples:
            n_ans_in_triples += 1

        # What retrieval actually hands the model (restricted, as in compare_all)
        facts = graph_facts_for_query(q, graph, allowed_sources={src})
        retrieved = answer_in_text(answer, " ".join(facts)) if facts else False
        if retrieved:
            n_ans_retrieved += 1

        if len(examples) < args.n and in_page:
            examples.append((q, answer, src, in_triples, retrieved,
                             facts, len(triples)))

    n = len([i for i in questions if i["source"] in page_chunks])

    # ── Funnel ──────────────────────────────────────────────────────────────
    print(f"\n{BOLD}The funnel — where does the answering fact get lost?{RESET}\n")
    def bar(count, total):
        pct = count / total * 100 if total else 0
        filled = int(pct / 5)
        colour = GREEN if pct >= 60 else (YELLOW if pct >= 25 else RED)
        return f"{colour}{'█'*filled}{'░'*(20-filled)} {count:>3}/{total} ({pct:>3.0f}%){RESET}"

    print(f"  answer is on the source page   {bar(n_ans_in_page, n)}")
    print(f"  {DIM}└─ survived triple extraction{RESET}  {bar(n_ans_in_triples, n)}")
    print(f"     {DIM}└─ returned by KG retrieval{RESET}  {bar(n_ans_retrieved, n)}")
    if n_no_triples:
        print(f"\n  {WARN} {n_no_triples} question(s) had NO triples at all "
              f"for their source page")

    # ── Worked examples ─────────────────────────────────────────────────────
    print(f"\n{BOLD}Worked examples{RESET}")
    for q, ans, src, in_tri, retr, facts, n_tri in examples:
        print(f"\n  {'-' * 74}")
        print(f"  Q       : {q[:72]}")
        print(f"  ANSWER  : {ans[:72]}")
        print(f"  page    : {src}  ({n_tri} triples extracted)")
        print(f"  in triples? {TICK if in_tri else CROSS}    "
              f"returned by retrieval? {TICK if retr else CROSS}")
        if facts:
            print(f"  {DIM}facts actually injected:{RESET}")
            for f in facts[:4]:
                print(f"    - {f[:74]}")
        else:
            print(f"  {DIM}(no facts injected){RESET}")

    # ── Verdict ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print(f"{BOLD}  VERDICT{RESET}")
    print(f"{'=' * 78}\n")

    extract_rate  = n_ans_in_triples / n_ans_in_page if n_ans_in_page else 0
    retrieve_rate = n_ans_retrieved / n_ans_in_triples if n_ans_in_triples else 0

    if extract_rate < 0.4:
        print(f"  {RED}(A) EXTRACTION is the bottleneck.{RESET}")
        print(f"      The answering fact survives extraction only "
              f"{extract_rate*100:.0f}% of the time,")
        print(f"      so the KG cannot help no matter how retrieval is tuned.")
        print(f"\n      Consistent with fixed-size chunking severing relations "
              f"mid-sentence.")
        print(f"      Testing that properly = semantic chunking + re-extraction "
              f"(~$23, ~3h).")
        print(f"\n      {DIM}Reportable either way: 'KG extraction quality is "
              f"bounded by chunking{RESET}")
        print(f"      {DIM}strategy; fixed-size chunks, chosen to tolerate OCR "
              f"noise, sever the{RESET}")
        print(f"      {DIM}relations the KG depends on.'{RESET}")
    elif retrieve_rate < 0.4:
        print(f"  {YELLOW}(B) RETRIEVAL is the bottleneck.{RESET}")
        print(f"      Extraction captures the fact {extract_rate*100:.0f}% of the "
              f"time, but whole-word")
        print(f"      node matching surfaces it only {retrieve_rate*100:.0f}% of "
              f"the time.")
        print(f"\n      Fixable in graph_facts_for_query — no re-extraction needed.")
        print(f"      Options: semantic matching over triples, or fuzzy/embedding "
              f"node match.")
    else:
        print(f"  {GREEN}Neither stage is obviously broken.{RESET}")
        print(f"      extraction {extract_rate*100:.0f}%, "
              f"retrieval {retrieve_rate*100:.0f}%.")
        print(f"      The facts reach the model but do not help — the KG may be")
        print(f"      redundant when the answer is already in the retrieved text.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
