"""
Multimodal RAG, built on top of rag_basics.

The image is made retrievable by CLIP (image + query embedded in a shared space),
so a TEXT question can pull the right IMAGE by visual similarity. The matched
image's CAPTION is then fed to the generator alongside the retrieved text chunks
(late fusion).

So: multimodal RETRIEVAL (CLIP), text GENERATION (the caption stands in for the
image). This mirrors the KG version's structure -- a second source fused with
text retrieval.

Pipeline:
  1. CLIP-embed every image  -> image FAISS index
  2. query -> CLIP text embed -> retrieve top image(s)
  3. fuse: text chunks + retrieved image caption(s) -> generator

Setup:
    pip install torch open-clip-torch pillow
    (plus the rag_basics deps, and publaynet_images/ produced by ingest_publaynet.py)
"""
import os
import json
import glob

import numpy as np
import faiss
import torch
import open_clip
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

from rag_basics import build_index, retrieve, CHAT_MODEL

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
IMAGE_DIR = "publaynet_images"

# ---------- load CLIP once ----------
print("Loading CLIP...")
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
clip_model.eval()


# ---------- 1. CLIP-embed images -> image index ----------
def embed_image(path):
    img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        vec = clip_model.encode_image(img)
    return vec[0].cpu().numpy().astype("float32")

def embed_text_clip(text):
    tokens = clip_tokenizer([text])
    with torch.no_grad():
        vec = clip_model.encode_text(tokens)
    return vec[0].cpu().numpy().astype("float32")

def build_image_index(image_dir=IMAGE_DIR):
    with open(os.path.join(image_dir, "captions.json"), encoding="utf-8") as f:
        captions = json.load(f)

    img_names, vectors = [], []
    for path in sorted(glob.glob(os.path.join(image_dir, "*.png"))):
        img_names.append(os.path.basename(path))
        vectors.append(embed_image(path))

    vectors = np.array(vectors, dtype="float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    print(f"Image index: {len(img_names)} images")
    return index, img_names, captions


# ---------- 2. retrieve images for a text query ----------
def retrieve_images(query, img_index, img_names, captions, k=1):
    query_vec = embed_text_clip(query).reshape(1, -1)
    faiss.normalize_L2(query_vec)
    scores, idxs = img_index.search(query_vec, k)
    results = []
    for score, i in zip(scores[0], idxs[0]):
        name = img_names[i]
        results.append((float(score), name, captions[name]))   # (score, name, caption)
    return results


# ---------- 3. fuse text chunks + image captions ----------
def generate_multimodal(query, text_retrieved, image_retrieved):
    context = "\n\n".join(f"[{source}] {chunk}" for _, source, chunk in text_retrieved)
    img_block = "\n".join(f"- ({name}) {caption}" for _, name, caption in image_retrieved)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system",
             "content": "Answer using only the provided text context and image descriptions. "
                        "If the answer isn't there, say you don't know."},
            {"role": "user",
             "content": (
                 f"Text context:\n{context}\n\n"
                 f"Relevant images (described):\n{img_block}\n\n"
                 f"Question: {query}"
             )},
        ],
    )
    return resp.choices[0].message.content

def ask_multimodal(query, text_index, chunks, sources, img_index, img_names, captions, k=3):
    text_retrieved = retrieve(query, text_index, chunks, sources, k=k)
    image_retrieved = retrieve_images(query, img_index, img_names, captions, k=1)
    answer = generate_multimodal(query, text_retrieved, image_retrieved)
    return answer, text_retrieved, image_retrieved


# ---------- demo ----------
if __name__ == "__main__":
    text_index, chunks, sources = build_index()
    img_index, img_names, captions = build_image_index()

    print("\nMultimodal RAG. Type 'quit' to exit.\n")
    while True:
        question = input("Question: ").strip()
        if question.lower() in ("quit", "exit", "q", ""):
            break
        answer, text_retrieved, image_retrieved = ask_multimodal(
            question, text_index, chunks, sources, img_index, img_names, captions
        )
        print("\nRetrieved image (score : name):")
        for score, name, caption in image_retrieved:
            print(f"  {score:.3f} : {name}")
        print(f"\nAnswer: {answer}\n" + "-" * 60)