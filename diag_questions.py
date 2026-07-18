"""
diag_questions.py — offline audit. No API calls, no cost.

Answers three things:
  1. Are the new questions' gold sources present in captions.json / the image index?
  2. Are the gold answers cell-values (pixel-only) or prose (caption-answerable)?
  3. What fraction of gold answers appear in their own caption?  <-- ceiling for --no-vlm

Usage:  python diag_questions.py
"""
import json, os, re, sys

NEW  = "questions_publaynet_figures.json"
OLD  = os.path.join("questions_old_100page", "questions_publaynet_figures.json")

# captions.json lives in the image dir; try the usual suspects
CAP_CANDIDATES = [
    "captions.json",
    os.path.join("publaynet_images", "captions.json"),
    os.path.join("images", "captions.json"),
    os.path.join("publaynet_corpus", "captions.json"),
]

def find_captions():
    for p in CAP_CANDIDATES:
        if os.path.exists(p):
            return p
    for root, _, files in os.walk("."):
        if "captions.json" in files:
            return os.path.join(root, "captions.json")
    return None

def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

def norm(s):
    """lowercase, strip punctuation/whitespace — for loose containment checks"""
    return re.sub(r"[^a-z0-9.%±]", "", s.lower())

def numeric_ratio(s):
    if not s:
        return 0.0
    return sum(c.isdigit() for c in s) / len(s)

def audit(label, qs, caps, cap_path):
    print("\n" + "=" * 72)
    print(f"{label}   ({len(qs)} questions)")
    print("=" * 72)

    # --- 1. gold source coverage ---
    missing = [q["source"] for q in qs if q["source"] not in caps]
    print(f"gold source present in captions.json     : {len(qs)-len(missing)}/{len(qs)}")
    if missing:
        print(f"  MISSING (first 5): {missing[:5]}")

    # --- 2. answer shape ---
    cell_like = [q for q in qs if numeric_ratio(q["answer"]) > 0.15]
    mean_len  = sum(len(q["answer"]) for q in qs) / len(qs)
    print(f"answers that are mostly numeric (cell)   : {len(cell_like)}/{len(qs)}")
    print(f"mean gold answer length (chars)          : {mean_len:.0f}")

    # --- 3. caption ceiling: is the gold answer IN its own caption? ---
    hits, checked = 0, 0
    examples = []
    for q in qs:
        cap = caps.get(q["source"])
        if cap is None:
            continue
        checked += 1
        a, c = norm(q["answer"]), norm(cap)
        # loose: does a distinctive chunk of the answer occur in the caption?
        probe = a[:14] if len(a) >= 14 else a
        if probe and probe in c:
            hits += 1
        elif len(examples) < 3:
            examples.append((q["q"][:60], q["answer"][:40], cap[:70]))
    if checked:
        print(f"gold answer findable in its own caption  : {hits}/{checked}"
              f"   <-- CEILING for any --no-vlm run")
    for qq, aa, cc in examples:
        print(f"    Q   : {qq}...")
        print(f"    gold: {aa}")
        print(f"    cap : {cc}...")
        print()

    # --- 4. type field sanity ---
    types = {}
    for q in qs:
        types[q.get("type", "<none>")] = types.get(q.get("type", "<none>"), 0) + 1
    print(f"type field                               : {types}")
    keys = sorted({k for q in qs for k in q})
    print(f"schema keys                              : {keys}")


def main():
    cap_path = find_captions()
    if cap_path is None:
        print("ERROR: could not locate captions.json. Edit CAP_CANDIDATES.")
        sys.exit(1)
    caps = load(cap_path)
    lens = [len(v.split()) for v in caps.values()]
    print(f"captions.json: {cap_path}")
    print(f"  {len(caps)} entries, mean {sum(lens)/len(lens):.0f} words "
          f"(~{sum(lens)/len(lens)*1.3:.0f} tokens each)")
    print(f"  -> k=1 caption block should add ~{sum(lens)/len(lens)*1.3:.0f} tokens")
    print(f"  -> k=3 caption block should add ~{sum(lens)/len(lens)*1.3*3:.0f} tokens")

    if os.path.exists(NEW):
        audit("NEW  (1000-page set)", load(NEW), caps, cap_path)
    else:
        print(f"\n!! not found: {NEW}")

    if os.path.exists(OLD):
        audit("OLD  (100-page set)", load(OLD), caps, cap_path)
    else:
        print(f"\n!! not found: {OLD}")


if __name__ == "__main__":
    main()
