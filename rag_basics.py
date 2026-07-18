"""
Baseline RAG: multiple local documents (.txt, .docx, .csv), source tracking, FAISS.

The chunk -> embed -> retrieve -> generate loop is the same as the toy version.
What's new:
  - loaders that turn each file type into plain text
  - every chunk remembers which file it came from (provenance)
  - FAISS holds the vectors instead of a Python list + manual cosine

This is the baseline the other systems build on: graph_aware, rag_multimodal,
and rag_full all import build_index, retrieve, and generate from here.

Setup:
    pip install openai numpy faiss-cpu python-docx python-dotenv
    echo 'OPENAI_API_KEY=sk-...' > .env
    python rag_basics.py
"""
import os
import csv
import glob

import numpy as np
import faiss
from openai import OpenAI
from docx import Document
from dotenv import load_dotenv

# explainability wrapper, shared by all systems
from explainability import format_explanation, text_evidence, parse_level

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o"
CORPUS_DIR  = "publaynet_corpus"


# ---------- LOADERS: each file type -> plain text ----------
def load_txt(path):
    with open(path, encoding="utf-8") as f:
        return f.read()

def load_docx(path):
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def load_csv(path):
    """Turns each row into a readable sentence so it can be embedded."""
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # e.g. "planet: Earth | diameter_km: 12742 | moons: 1 | fact: ..."
            rows.append(" | ".join(f"{key}: {value}" for key, value in row.items()))
    return "\n".join(rows)

def load_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":  return load_txt(path)
    if ext == ".docx": return load_docx(path)
    if ext == ".csv":  return load_csv(path)
    raise ValueError(f"Unsupported file type: {ext}")


# ---------- CHUNK ----------
def chunk_text(text, chunk_size=500, overlap=50):
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap          # step back by overlap so chunks share a margin
    return chunks


# ---------- EMBED ----------
# OpenAI caps a single embeddings request at 300k tokens. A 1,000-page corpus is
# ~8,300 chunks ≈ 845k tokens, so one request fails. We batch under the cap.
# 1,000 chunks x ~125 tokens ≈ 125k — comfortable headroom for long chunks.
EMBED_BATCH   = 1000
EMBED_RETRIES = 5


def embed(texts, quiet=True):
    """
    Embed a list of texts, batching to stay under the per-request token cap.

    Retries each batch with exponential backoff: a transient 429 partway through
    an 8,000-chunk index build shouldn't discard the batches already done.
    """
    import time

    if isinstance(texts, str):
        texts = [texts]

    all_vecs = []
    n_batches = (len(texts) + EMBED_BATCH - 1) // EMBED_BATCH

    for b in range(n_batches):
        batch = texts[b * EMBED_BATCH:(b + 1) * EMBED_BATCH]

        for attempt in range(EMBED_RETRIES):
            try:
                resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
                all_vecs.extend(d.embedding for d in resp.data)
                break
            except Exception as e:
                if attempt == EMBED_RETRIES - 1:
                    raise
                wait = 2 ** attempt
                print(f"  [embed] batch {b+1}/{n_batches} failed "
                      f"({str(e)[:60]}); retry in {wait}s")
                time.sleep(wait)

        if not quiet and n_batches > 1:
            print(f"  [embed] {b + 1}/{n_batches} batches "
                  f"({len(all_vecs)}/{len(texts)} chunks)")

    return np.array(all_vecs, dtype="float32")


# ---------- BUILD INDEX (with provenance + cache) ----------
def _corpus_fingerprint(corpus_dir):
    """
    Hash of the corpus contents: filenames + sizes + mtimes.

    Keys the cache so it invalidates automatically when the corpus changes.
    Without this, pointing the pipeline at DocLayNet would silently reuse
    PubLayNet's index — the same contamination trap as stale captions or
    stale question files.
    """
    import hashlib
    h = hashlib.md5()
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*"))):
        st = os.stat(path)
        h.update(f"{os.path.basename(path)}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


def build_index(corpus_dir=CORPUS_DIR, rebuild=False):
    """
    Build (or load) the FAISS text index.

    The index is IDENTICAL across every run — that is the experimental design:
    only the generator varies. Re-embedding 8,300 chunks for each of the 12
    planned runs would cost 12x for no benefit, so vectors are cached to disk
    and keyed on the corpus fingerprint.

    rebuild=True forces a fresh build.
    """
    import pickle

    fp = _corpus_fingerprint(corpus_dir)
    cache_path = f".index_cache_{os.path.basename(corpus_dir)}_{fp}.pkl"

    if not rebuild and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            vectors, chunks, sources = data["vectors"], data["chunks"], data["sources"]
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            print(f"Loaded cached index: {len(chunks)} chunks from "
                  f"{len(set(sources))} files.")
            return index, chunks, sources
        except Exception as e:
            print(f"  cache unreadable ({str(e)[:40]}); rebuilding")

    chunks, sources = [], []          # parallel lists: chunk text + its source file
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*"))):
        text = load_file(path)
        for chunk in chunk_text(text):
            chunks.append(chunk)
            sources.append(os.path.basename(path))

    print(f"Embedding {len(chunks)} chunks from {len(set(sources))} files...")
    vectors = embed(chunks, quiet=False)
    faiss.normalize_L2(vectors)       # normalised vectors make inner product == cosine

    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"vectors": vectors, "chunks": chunks,
                         "sources": sources}, f)
        print(f"Index cached -> {cache_path}")
    except Exception as e:
        print(f"  could not cache index ({str(e)[:40]}); continuing")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    print(f"Indexed {len(chunks)} chunks from {len(set(sources))} files.")
    return index, chunks, sources


# ---------- RETRIEVE ----------
def retrieve(query, index, chunks, sources, k=3):
    query_vec = embed([query])
    faiss.normalize_L2(query_vec)
    scores, idxs = index.search(query_vec, k)
    results = []
    for score, i in zip(scores[0], idxs[0]):
        results.append((float(score), sources[i], chunks[i]))   # (score, source, chunk)
    return results


# ---------- GENERATE ----------
def generate(query, retrieved, model="gpt4o-mini"):
    """
    Baseline generation: text context only.

    `model` selects the generator from llm_client.MODELS, which is how
    compare_all.py swaps between the four systems under comparison. Returns a
    GenResult (text + tokens + latency + cost), not a bare string, so the
    caller can log per-call efficiency data. Use .text for the answer.
    """
    from llm_client import call
    context = "\n\n".join(f"[{source}] {chunk}" for _, source, chunk in retrieved)
    return call(
        model,
        system="Answer using only the provided context. "
               "If the answer isn't in it, say you don't know.",
        user=f"Context:\n{context}\n\nQuestion: {query}",
    )


def ask(query, index, chunks, sources, k=3, model="gpt4o-mini"):
    retrieved = retrieve(query, index, chunks, sources, k=k)
    return generate(query, retrieved, model=model).text, retrieved


# ---------- interactive loop ----------
if __name__ == "__main__":
    LEVEL = parse_level(default=2)        # read --level N once
    index, chunks, sources = build_index()
    print(f"Ask questions about the corpus (explain level={LEVEL}). Type 'quit' to exit.\n")

    while True:
        query = input("Question: ").strip()
        if query.lower() in ("quit", "exit", "q", ""):
            print("Done.")
            break

        answer, retrieved = ask(query, index, chunks, sources)

        # baseline evidence is text only: no figure, no graph facts
        evidence = {
            "text": text_evidence(retrieved),     # retrieved (score, source, chunk) tuples
            "reasoning": "Answer drawn from the retrieved text passages.",
        }

        print("\n" + format_explanation(answer, evidence, LEVEL) + "\n" + "-" * 60)