"""
HotpotQA ingestion — the corpus where the KG can actually be tested.

Why this dataset exists in the project
──────────────────────────────────────
PubLayNet, DocLayNet and DocBank are layout-analysis corpora: sampled pages from
unrelated papers. They share no entities, so a knowledge graph has nothing to
connect. Measured on PubLayNet (gpt4o-mini, n=35 text / n=30 multi-hop), +KG did
not help and mildly hurt:

    text       baseline 0.657 acc / 0.714 faith  ->  +KG 0.571 / 0.571
    multi-hop  baseline 0.433 acc                ->  +KG 0.400

Three diagnostics ruled out OCR noise (2% of facts garbled), cross-document
leakage (restricting to retrieved pages gave no recovery), and extraction
failure (81% of answers survived extraction). The KG had nothing to do because
the corpus has no cross-document structure.

HotpotQA is the control: Wikipedia-derived, entity-linked, every question
genuinely 2-hop. If +KG helps here and not on PubLayNet, the claim is that KG
value is a property of the CORPUS, not of the implementation.

Design
──────
  pool from 300 questions  -> ~2,500 unique paragraphs (the corpus)
  evaluate on 100 of them  -> 50 bridge + 50 comparison

Two numbers, not one. HotpotQA ships 10 paragraphs per question (2 gold + 8
distractors), so the corpus is built by pooling. Pooling from only the 100 eval
questions would give ~800 paragraphs — small enough that retrieval returns both
gold paragraphs nearly every time, the answer sits fully in context, and the KG
again has no job. Pooling from 300 makes retrieval miss a gold paragraph often
enough for the graph to have a bridge to build.

Why the bridge/comparison split
───────────────────────────────
  bridge     — "birthplace of the director of Inception?" The answer needs
               Inception -> Nolan -> London. The second paragraph mentions
               neither "Inception" nor "director", so if retrieval misses it,
               only a graph edge reaches it. THE KG'S HOME TURF.
  comparison — "which was founded first, Oxford or Cambridge?" Both entities are
               named in the question; both paragraphs retrieve directly. No
               bridge to follow. A BUILT-IN CONTROL.

If +KG lifts bridge but not comparison, the mechanism is proven rather than
inferred — the same role text questions played in the PubLayNet ablation.

Outputs (same contract as the other ingesters, so the pipeline is unchanged):
  hotpotqa_corpus/<title>.txt          – one paragraph per file
  questions_hotpotqa_bridge.json       – 50 items
  questions_hotpotqa_comparison.json   – 50 items

No images: HotpotQA is text-only, so +multimodal/+both do not apply. Only
baseline and +KG are run.

Schema note — `source` is a LIST here, not a string. Each question has 2+ gold
paragraphs, unlike PubLayNet's single source. compare_all.py scores this with
partial and complete recall.

No OCR, no captions, no API calls: paragraphs are clean Wikipedia text.

Setup:
    pip install datasets python-dotenv

Usage:
    python ingest_hotpotqa.py --smoke          # 5 questions, verbose
    python ingest_hotpotqa.py                  # pool 300, eval 100 (50/50)
    python ingest_hotpotqa.py --pool 500 --eval 100
"""

import os
import re
import sys
import json
import glob
import shutil
import argparse
import unicodedata
from collections import Counter

from datasets import load_dataset

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[1m", "\033[0m"
)
TICK, CROSS, WARN = f"{GREEN}✓{RESET}", f"{RED}✗{RESET}", f"{YELLOW}!{RESET}"

CORPUS_DIR = "hotpotqa_corpus"
HF_DATASET = "hotpotqa/hotpot_qa"
HF_CONFIG  = "distractor"      # 10 paragraphs per question: 2 gold + 8 distractors
SEED       = 42


def safe_filename(title: str) -> str:
    """
    Wikipedia titles -> safe, ASCII-only filenames.

    Two problems being solved:

    1. Windows rejects  : / ? * " < > |  in filenames, and Wikipedia titles are
       full of them ("Boston, Lincolnshire", "AC/DC", "Who? (album)").

    2. Non-ASCII titles ("Beyoncé", "Tōkyō") produce filenames that any tool
       reading without an explicit encoding will choke on — Windows defaults to
       cp1252, which cannot decode them. Transliterating to ASCII keeps every
       downstream reader safe regardless of its default encoding.

    Transliteration can collide (Beyoncé and Beyonce both -> "Beyonce"), which
    would silently merge two distinct paragraphs, so a short hash of the ORIGINAL
    title is appended whenever anything was stripped.
    """
    import hashlib

    t = unicodedata.normalize("NFC", title)

    # Transliterate accents to ASCII: é -> e, ō -> o.
    ascii_t = (unicodedata.normalize("NFKD", t)
               .encode("ascii", "ignore").decode("ascii"))

    cleaned = re.sub(r'[<>:"/\\|?*,()\'\[\]]', "_", ascii_t)
    cleaned = re.sub(r"\s+", "_", cleaned.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    cleaned = cleaned[:100]

    if not cleaned:
        cleaned = "untitled"

    # Anything lost to transliteration or truncation risks a collision, so
    # disambiguate with a hash of the original title.
    if ascii_t != t or len(t) > 100:
        h = hashlib.md5(t.encode("utf-8")).hexdigest()[:6]
        cleaned = f"{cleaned}_{h}"

    return cleaned


def norm_title(title: str) -> str:
    """Canonical title for matching supporting_facts against context titles."""
    return unicodedata.normalize("NFC", title).strip()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Ingest HotpotQA: pool a corpus, emit bridge/comparison sets.")
    ap.add_argument("--pool", type=int, default=300,
                    help="questions whose paragraphs form the corpus (default 300)")
    ap.add_argument("--eval", type=int, default=100,
                    help="questions to evaluate on, split evenly (default 100)")
    ap.add_argument("--smoke", action="store_true",
                    help="5 questions, verbose, writes nothing permanent")
    ap.add_argument("--clean", action="store_true",
                    help="wipe the corpus dir first (avoids stale-corpus mixing)")
    args = ap.parse_args()

    pool_n = 5 if args.smoke else args.pool
    eval_n = 4 if args.smoke else args.eval

    print(f"\n{'=' * 74}")
    print("  HOTPOTQA INGESTION — the corpus with real graph structure")
    print(f"{'=' * 74}")
    print(f"{DIM}  pool {pool_n} questions -> corpus | evaluate on {eval_n} "
          f"(half bridge, half comparison){RESET}")

    # Stale corpora have bitten this project twice (captions.json, old question
    # files). Offer an explicit wipe rather than silently mixing.
    if args.clean and os.path.exists(CORPUS_DIR):
        shutil.rmtree(CORPUS_DIR)
        print(f"{DIM}  wiped {CORPUS_DIR}/{RESET}")
    if os.path.exists(CORPUS_DIR) and glob.glob(os.path.join(CORPUS_DIR, "*.txt")):
        n_existing = len(glob.glob(os.path.join(CORPUS_DIR, "*.txt")))
        print(f"\n  {WARN} {CORPUS_DIR}/ already holds {n_existing} files.")
        print(f"      Re-run with --clean for a fresh corpus, or these will be "
              f"mixed in.")

    os.makedirs(CORPUS_DIR, exist_ok=True)

    print(f"\n{DIM}Loading {HF_DATASET} [{HF_CONFIG}] validation split...{RESET}")
    # validation (7,405 rows) not train (90k): smaller download, standard for eval.
    ds = load_dataset(HF_DATASET, HF_CONFIG, split="validation",
                      streaming=True, trust_remote_code=False)

    paragraphs   = {}          # safe_filename -> paragraph text
    title_to_file = {}         # normalised title -> safe_filename
    candidates   = {"bridge": [], "comparison": []}

    n_seen = n_empty_filtered = n_bad_gold = 0

    for row in ds:
        if n_seen >= pool_n:
            break
        n_seen += 1

        titles    = row["context"]["title"]
        sentences = row["context"]["sentences"]
        qtype     = row.get("type", "bridge")

        # ── Write every paragraph (gold AND distractor) into the pool ────────
        # Distractors are the point: they are what retrieval must sift through.
        row_files = {}
        for title, sents in zip(titles, sentences):
            text = " ".join(s.strip() for s in sents if s.strip()).strip()
            if not text:
                # HotpotQA contains a non-trivial number of empty paragraphs;
                # keeping them would put empty documents in the index.
                n_empty_filtered += 1
                continue
            fname = safe_filename(title)
            paragraphs[fname] = text                 # dedupe: titles recur
            title_to_file[norm_title(title)] = fname
            row_files[norm_title(title)] = fname

        # Resolve gold paragraphs for this question.
        # NOTE the ".txt": build_index records sources as os.path.basename(path),
        # i.e. WITH the extension. The gold list must match that exactly or every
        # recall comparison silently returns 0.
        gold_titles = {norm_title(t) for t in row["supporting_facts"]["title"]}
        gold_files  = sorted({f"{row_files[t]}.txt"
                              for t in gold_titles if t in row_files})

        # Only ~70% of questions have exactly two supporting paragraphs, and a
        # gold title can be missing if its paragraph was empty. Require >=2 so
        # every eval question genuinely needs multiple documents.
        if len(gold_files) < 2:
            n_bad_gold += 1
            continue

        item = {
            "q":      row["question"],
            "source": gold_files,          # LIST — differs from PubLayNet
            "answer": row["answer"],
            "type":   f"hotpot_{qtype}",
            "level":  row.get("level", "unknown"),
        }
        if qtype in candidates:
            candidates[qtype].append(item)

        if args.smoke:
            print(f"\n  --- {row['id'][:12]} [{qtype}] ---")
            print(f"  Q     : {row['question'][:66]}")
            print(f"  A     : {row['answer'][:66]}")
            print(f"  gold  : {gold_files}")
            print(f"  paras : {len(row_files)} kept from {len(titles)}")

    # ── Write the corpus ────────────────────────────────────────────────────
    if not args.smoke:
        for fname, text in paragraphs.items():
            with open(os.path.join(CORPUS_DIR, f"{fname}.txt"),
                      "w", encoding="utf-8") as f:
                f.write(text)

    # ── Select the eval questions: half bridge, half comparison ─────────────
    import random
    rng = random.Random(SEED)
    per_type = eval_n // 2
    selected = {}
    for t in ("bridge", "comparison"):
        pool = candidates[t]
        rng.shuffle(pool)
        selected[t] = pool[:per_type]

    if not args.smoke:
        for t, items in selected.items():
            path = f"questions_hotpotqa_{t}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(items, f, indent=2, ensure_ascii=False)

    # ── Report ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 74}")
    print(f"{BOLD}  CORPUS{RESET}")
    print(f"  questions pooled      : {n_seen}")
    print(f"  unique paragraphs     : {len(paragraphs):,}"
          f"  {DIM}(from {n_seen * 10} slots — titles recur across questions){RESET}")
    if n_empty_filtered:
        print(f"  empty paras filtered  : {n_empty_filtered}")
    if n_bad_gold:
        print(f"  questions dropped     : {n_bad_gold} "
              f"{DIM}(<2 resolvable gold paragraphs){RESET}")

    print(f"\n{BOLD}  QUESTIONS{RESET}")
    for t in ("bridge", "comparison"):
        got, want = len(selected[t]), per_type
        colour = GREEN if got >= want else YELLOW
        role = ("KG's home turf — needs a bridge entity" if t == "bridge"
                else "control — both entities named, no bridge needed")
        print(f"  {colour}{got:>3}/{want}{RESET} {t:<11} {DIM}{role}{RESET}")
        if got < want:
            print(f"      {WARN} short — raise --pool to find more")

    if args.smoke:
        print(f"\n{DIM}  (smoke run — nothing written){RESET}")
        print(f"\n{DIM}  Next: python ingest_hotpotqa.py --clean{RESET}\n")
        return 0

    total = sum(len(v) for v in selected.values())
    print(f"\n  corpus    -> {CORPUS_DIR}/  ({len(paragraphs):,} files)")
    print(f"  questions -> questions_hotpotqa_bridge.json, "
          f"questions_hotpotqa_comparison.json  ({total} items)")

    # Cost/time estimate: chunking is ~500 chars with overlap, paragraphs ~60 words.
    est_chunks = int(len(paragraphs) * 1.2)
    print(f"\n{DIM}  ~{est_chunks:,} chunks -> KG extraction at gpt-4o-mini "
          f"≈ ${est_chunks * 0.0002:.2f}, ~{est_chunks * 1.65 / 3600:.1f}h{RESET}")
    print(f"\n{YELLOW}  Before running compare_all.py on this corpus:{RESET}")
    print(f"{DIM}    • EXTRACT_MODEL must be gpt-4o-mini (not gpt-4o){RESET}")
    print(f"{DIM}    • CORPUS_DIR must point at {CORPUS_DIR}{RESET}")
    print(f"{DIM}    • triples_cache.json is PubLayNet's — this corpus needs "
          f"its own{RESET}")
    print(f"{DIM}    • only baseline and +KG apply (no images){RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
