"""
diag_context.py — prints the EXACT context string each path builds, for one question.

No generator call. Retrieval only (CLIP + text index). Costs nothing.

Shows, side by side:
  - baseline text context
  - +multimodal caption block  (the --no-vlm path)
  - which image the VLM path would send

Usage:  python diag_context.py
        python diag_context.py 7          (question index)
        python diag_context.py 7 questions_old_100page\questions_publaynet_figures.json
"""
import json, os, sys

from rag_basics import build_index, retrieve
from rag_multimodal import build_image_index, retrieve_images

QFILE = "questions_publaynet_figures.json"
IDX   = 0

if len(sys.argv) > 1:
    IDX = int(sys.argv[1])
if len(sys.argv) > 2:
    QFILE = sys.argv[2]

with open(QFILE, encoding="utf-8") as f:
    qs = json.load(f)

q    = qs[IDX]
query = q["q"]
gold  = q["source"]

print("=" * 74)
print(f"FILE     : {QFILE}")
print(f"INDEX    : {IDX} of {len(qs)}")
print(f"QUESTION : {query}")
print(f"GOLD SRC : {gold}")
print(f"GOLD ANS : {q['answer']}")
print("=" * 74)

text_index, chunks, sources = build_index()
img_index, img_names, captions = build_image_index()

# ---------- is the gold image even in the index? ----------
print(f"\ngold image in CLIP index : {gold in img_names}")
print(f"gold image in captions   : {gold in captions}")
if gold in captions:
    print(f"gold caption             : {captions[gold]}")

# ---------- text retrieval (what baseline sees) ----------
text_retrieved = retrieve(query, text_index, chunks, sources, k=3)
ctx = "\n\n".join(f"[{s}] {c}" for _, s, c in text_retrieved)
print("\n" + "-" * 74)
print("BASELINE TEXT CONTEXT (exact string sent to generator)")
print("-" * 74)
print(ctx)
print(f"--- END ({len(ctx)} chars, ~{len(ctx)//4} tokens) ---")

# ---------- image retrieval at k=1 (CURRENT behaviour) and k=3 ----------
for k in (1, 3):
    imgs = retrieve_images(query, img_index, img_names, captions, k=k)
    block = "\n".join(f"- ({name}) {cap}" for _, name, cap in imgs)
    ranks = [name for _, name, _ in imgs]
    hit   = gold in ranks
    rank  = ranks.index(gold) + 1 if hit else None
    print("\n" + "-" * 74)
    print(f"IMAGE RETRIEVAL k={k}   {'<-- CURRENT (hardcoded in rag_multimodal.py)' if k==1 else '<-- what Recall@3 claims to measure'}")
    print("-" * 74)
    for i, (score, name, cap) in enumerate(imgs, 1):
        mark = "  <== GOLD" if name == gold else ""
        print(f"  {i}. {score:.4f}  {name}{mark}")
    print(f"gold found: {hit}" + (f" at rank {rank}" if hit else ""))
    print(f"\nCAPTION BLOCK sent to generator on --no-vlm at k={k}:")
    print(block if block else "  <EMPTY>")
    print(f"--- END ({len(block)} chars, ~{len(block)//4} tokens) ---")

print("\n" + "=" * 74)
print("Compare the k=1 vs k=3 gold-found lines above.")
print("If gold appears at rank 2 or 3, your Recall@3 is undercounting.")
print("=" * 74)
