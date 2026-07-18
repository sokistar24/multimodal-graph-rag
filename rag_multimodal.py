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

def _image_fingerprint(image_dir):
    """
    Hash of the image directory: filenames + sizes + mtimes.

    Keys the CLIP cache so it invalidates when the crops change — pointing the
    pipeline at DocLayNet must not silently reuse PubLayNet's image index.
    """
    import hashlib
    h = hashlib.md5()
    for path in sorted(glob.glob(os.path.join(image_dir, "*.png"))):
        st = os.stat(path)
        h.update(f"{os.path.basename(path)}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


def build_image_index(image_dir=IMAGE_DIR, rebuild=False):
    """
    Build (or load) the CLIP image index.

    CLIP-embedding ~600 crops is local compute — no API cost, but slow on CPU,
    and it is IDENTICAL on every run (only the generator varies across runs).
    Caching it saves that work on each of the 12 planned runs.

    rebuild=True forces a fresh build.
    """
    import pickle

    with open(os.path.join(image_dir, "captions.json"), encoding="utf-8") as f:
        captions = json.load(f)

    fp = _image_fingerprint(image_dir)
    cache_path = f".clip_cache_{os.path.basename(image_dir)}_{fp}.pkl"

    if not rebuild and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            vectors, img_names = data["vectors"], data["img_names"]
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            print(f"Loaded cached image index: {len(img_names)} images")
            return index, img_names, captions
        except Exception as e:
            print(f"  CLIP cache unreadable ({str(e)[:40]}); rebuilding")

    paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    print(f"CLIP-embedding {len(paths)} images (one-off; cached after this)...")

    img_names, vectors = [], []
    for i, path in enumerate(paths, 1):
        img_names.append(os.path.basename(path))
        vectors.append(embed_image(path))
        if i % 100 == 0:
            print(f"  [clip] {i}/{len(paths)}")

    vectors = np.array(vectors, dtype="float32")
    faiss.normalize_L2(vectors)

    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"vectors": vectors, "img_names": img_names}, f)
        print(f"Image index cached -> {cache_path}")
    except Exception as e:
        print(f"  could not cache image index ({str(e)[:40]}); continuing")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    print(f"Image index: {len(img_names)} images")
    return index, img_names, captions


# ---------- 2. retrieve images for a text query ----------
def retrieve_images(query, img_index, img_names, captions, k=3):
    query_vec = embed_text_clip(query).reshape(1, -1)
    faiss.normalize_L2(query_vec)
    scores, idxs = img_index.search(query_vec, k)
    results = []
    for score, i in zip(scores[0], idxs[0]):
        name = img_names[i]
        results.append((float(score), name, captions[name]))   # (score, name, caption)
    return results


# ---------- 3. fuse text chunks + images ----------
def generate_multimodal(query, text_retrieved, image_retrieved,
                        model="gpt4o-mini", vlm=True, image_dir=IMAGE_DIR):
    """
    +multimodal generation. Returns GenResult.

    Two variants, per the paper's Method section:

      vlm=True  (default) — the retrieved figure is passed as PIXELS to the
                 generator for direct visual reasoning.
      vlm=False — only the figure's caption is passed as text.

    Both share identical CLIP retrieval, so any accuracy gap between them
    isolates what the caption loses relative to reading the image.

    Text-only models cannot take pixels. llm_client drops images for
    them automatically, so we must supply the caption or they get nothing at all.
    That fallback is not a bug: it is the caption-mediated variant, and the
    paper reports it as such.
    """
    from llm_client import call, MODELS

    context   = "\n\n".join(f"[{source}] {chunk}" for _, source, chunk in text_retrieved)
    img_block = "\n".join(f"- ({name}) {caption}" for _, name, caption in image_retrieved)

    can_see   = MODELS[model]["vision"]
    use_pixels = vlm and can_see and bool(image_retrieved)

    if use_pixels:
        paths = [os.path.join(image_dir, name) for _, name, _ in image_retrieved]
        paths = [p for p in paths if os.path.exists(p)]
        return call(
            model,
            system="Answer using only the provided text context and the attached "
                   "image(s). If the answer isn't there, say you don't know.",
            user=(f"Text context:\n{context}\n\n"
                  f"Question: {query}"),
            images=paths,
        )

    # Caption-mediated: text-only model, or vlm=False explicitly.
    return call(
        model,
        system="Answer using only the provided text context and image descriptions. "
               "If the answer isn't there, say you don't know.",
        user=(f"Text context:\n{context}\n\n"
              f"Relevant images (described):\n{img_block}\n\n"
              f"Question: {query}"),
    )

def ask_multimodal(query, text_index, chunks, sources, img_index, img_names,
                   captions, k=3, model="gpt4o-mini", vlm=True, k_img=None):
    """
    Interactive demo path.

    k_img defaults to 1 on the pixel path and 3 on the caption path, mirroring
    compare_all.py's K_IMAGES_PIXELS / K_IMAGES_CAPTION. The experiment passes
    its own prefix; this default only keeps the demo from drifting away from it.
    """
    if k_img is None:
        k_img = 1 if vlm else 3
    text_retrieved = retrieve(query, text_index, chunks, sources, k=k)
    image_retrieved = retrieve_images(query, img_index, img_names, captions, k=k_img)
    answer = generate_multimodal(query, text_retrieved, image_retrieved,
                                 model=model, vlm=vlm).text
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