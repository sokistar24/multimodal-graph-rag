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
def embed(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data], dtype="float32")


# ---------- BUILD INDEX (with provenance) ----------
def build_index(corpus_dir=CORPUS_DIR):
    chunks, sources = [], []          # parallel lists: chunk text + the file it came from
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*"))):
        text = load_file(path)
        for chunk in chunk_text(text):
            chunks.append(chunk)
            sources.append(os.path.basename(path))

    vectors = embed(chunks)
    faiss.normalize_L2(vectors)       # normalised vectors make inner product == cosine
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
def generate(query, retrieved):
    context = "\n\n".join(f"[{source}] {chunk}" for _, source, chunk in retrieved)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system",
             "content": "Answer using only the provided context. "
                        "If the answer isn't in it, say you don't know."},
            {"role": "user",
             "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    )
    return resp.choices[0].message.content


def ask(query, index, chunks, sources, k=3):
    retrieved = retrieve(query, index, chunks, sources, k=k)
    return generate(query, retrieved), retrieved


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