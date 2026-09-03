"""
SPIQA (test-A) ingestion: same output contract as ingestion.py (PubLayNet)
and ingest_doclaynet.py.

Produces:
  spiqa_corpus/<paper_id>.txt          - extracted paper paragraphs (clean TeX text)
  spiqa_images/<figure_filename>       - figure/table images copied from SPIQA
  spiqa_images/captions.json           - {filename: GPT-4o one-sentence caption}
  spiqa_images/author_captions.json    - {filename: original author caption} (sidecar,
                                          NOT used by the pipeline; analysis only)
  spiqa_gold_qa.json                   - SPIQA's native human-curated QA for the
                                          sampled papers (optional extra evaluation)

Key differences from the PubLayNet/DocLayNet ingests (state these in the paper):
  * No OCR. SPIQA text is TeX-derived, so the text channel is clean. This is the
    favourable / upper-bound condition for the graph hypothesis.
  * Provenance unit is the PAPER, not the page. `source` for text questions is
    "<paper_id>.txt"; for figure questions it is the SPIQA image filename.
  * Figure/table crops are NOT re-cropped; SPIQA ships them pre-extracted.
    We re-caption them with GPT-4o so the caption channel has identical
    semantics across all three corpora. Author captions go to a sidecar only.

Dataset: google/spiqa on Hugging Face (CC-BY-4.0). We use the test-A split
(manually curated; covered by the extracted-paragraphs zip).

Files fetched (once, cached by huggingface_hub):
  test-A/SPIQA_testA.json                          - metadata keyed by paper_id
  test-A/SPIQA_testA_Images.zip                    - full-res figure/table images
  SPIQA_train_val_test-A_extracted_paragraphs.zip  - per-paper paragraph text

Setup:
    pip install huggingface_hub pillow openai python-dotenv
    (May require: huggingface-cli login)

Usage:
    python ingest_spiqa.py --probe                 # inspect one paper's structure, no writes
    python ingest_spiqa.py --papers 100            # sample 100 papers deterministically
    python ingest_spiqa.py --papers 5 --smoke      # tiny run, verbose, no captioning
"""

import argparse
import base64
import io
import json
import os
import random
import shutil
import zipfile
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from openai import OpenAI
from PIL import Image

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

REPO_ID = "google/spiqa"
CORPUS_DIR = "spiqa_corpus"
IMAGE_DIR = "spiqa_images"
DOWNLOAD_DIR = "spiqa_download"  # hf files + unzipped content live here
CAPTION_MODEL = "gpt-4o"
SEED = 42  # deterministic paper sampling

META_FILE = "test-A/SPIQA_testA.json"
IMAGES_ZIP = "test-A/SPIQA_testA_Images.zip"
PARAS_ZIP = "SPIQA_train_val_test-A_extracted_paragraphs.zip"


# ─────────────────────────── download & unzip ────────────────────────────────


def fetch(filename: str) -> str:
    """Download one file from the SPIQA repo (cached across runs)."""
    return hf_hub_download(
        repo_id=REPO_ID, filename=filename, repo_type="dataset", local_dir=DOWNLOAD_DIR
    )


def ensure_unzipped(zip_path: str, marker_name: str) -> Path:
    """Unzip into DOWNLOAD_DIR/<marker_name>/ once; return the extraction root."""
    out = Path(DOWNLOAD_DIR) / marker_name
    if not out.exists() or not any(out.iterdir()):
        out.mkdir(parents=True, exist_ok=True)
        print(f"Unzipping {os.path.basename(zip_path)} -> {out} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out)
    return out


def index_files(root: Path) -> dict[str, Path]:
    """Map basename -> path for every file under root (zips differ in nesting)."""
    return {p.name: p for p in root.rglob("*") if p.is_file()}


# ─────────────────────────── paragraph loading ───────────────────────────────


def load_paragraphs(paper_id: str, para_files: dict[str, Path]) -> str | None:
    """
    Return the paper's full text, handling the two plausible storage formats:
      <paper_id>.txt   - plain text (paragraphs separated by blank lines)
      <paper_id>.json  - a list of paragraph strings, or {"paragraphs": [...]}
    """
    for ext in (".txt", ".json"):
        p = para_files.get(paper_id + ext)
        if p is None:
            continue
        raw = p.read_text(encoding="utf-8", errors="replace")
        if ext == ".txt":
            return raw.strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return raw.strip()
        if isinstance(data, dict):
            data = data.get("paragraphs") or data.get("text") or list(data.values())
        if isinstance(data, list):
            parts = [x if isinstance(x, str) else json.dumps(x) for x in data]
            return "\n\n".join(s.strip() for s in parts if s and s.strip())
        return str(data)
    return None


# ─────────────────────────── captioning ──────────────────────────────────────


def caption_image(path: Path, smoke: bool = False) -> str:
    """GPT-4o caption, identical prompt semantics to the other two ingests."""
    if smoke:
        return "(caption skipped in smoke mode)"
    img = Image.open(path).convert("RGB")
    img.thumbnail((1024, 1024))  # cap payload; full-res files can be large
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    resp = client.chat.completions.create(
        model=CAPTION_MODEL,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this region from a document in one concise sentence. "
                        "It may be a figure, chart, table, or diagram. Be specific about "
                        "the content, chart type, axes, or table values where visible.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
    )
    return resp.choices[0].message.content.strip()


# ─────────────────────────── probe mode ──────────────────────────────────────


def probe() -> None:
    """Print the real structure of the metadata and both zips for one paper,
    without writing any corpus files. Run this first on a new machine."""
    meta_path = fetch(META_FILE)
    meta = json.load(open(meta_path, encoding="utf-8"))
    print(f"metadata: {len(meta)} papers")
    pid = sorted(meta.keys())[0]
    entry = meta[pid]
    print(f"\n--- paper {pid} ---")
    print("top-level keys:", list(entry.keys()))
    figs = entry.get("all_figures", {})
    print(f"all_figures: {len(figs)} entries")
    if figs:
        fname, fmeta = next(iter(figs.items()))
        print("  example figure key:", fname)
        print(
            "  figure meta keys  :",
            list(fmeta.keys()) if isinstance(fmeta, dict) else type(fmeta),
        )
        if isinstance(fmeta, dict):
            print("  content_type      :", fmeta.get("content_type"))
            cap = fmeta.get("caption", "")
            print(
                "  author caption    :", (cap[:120] + "...") if len(cap) > 120 else cap
            )
    qa = entry.get("qa", [])
    print(f"qa: {len(qa)} items")
    if qa:
        print("  qa[0] keys:", list(qa[0].keys()))

    paras_root = ensure_unzipped(fetch(PARAS_ZIP), "paragraphs")
    para_files = index_files(paras_root)
    print(
        f"\nparagraphs zip: {len(para_files)} files; sample names:",
        list(para_files)[:3],
    )
    text = load_paragraphs(pid, para_files)
    print(
        f"paragraphs for {pid}: "
        + (f"{len(text)} chars; starts: {text[:150]!r}" if text else "NOT FOUND")
    )

    imgs_root = ensure_unzipped(fetch(IMAGES_ZIP), "images")
    img_files = index_files(imgs_root)
    print(f"\nimages zip: {len(img_files)} files; sample names:", list(img_files)[:3])
    if figs:
        hit = next(iter(figs)) in img_files
        print(f"figure key '{next(iter(figs))}' found in images zip: {hit}")


# ─────────────────────────── main ingestion ──────────────────────────────────


def ingest(n_papers: int, smoke: bool = False) -> None:
    os.makedirs(CORPUS_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    meta = json.load(open(fetch(META_FILE), encoding="utf-8"))
    paras_root = ensure_unzipped(fetch(PARAS_ZIP), "paragraphs")
    imgs_root = ensure_unzipped(fetch(IMAGES_ZIP), "images")
    para_files = index_files(paras_root)
    img_files = index_files(imgs_root)

    # Deterministic sample: sort ids, shuffle with fixed seed, take first n.
    paper_ids = sorted(meta.keys())
    random.Random(SEED).shuffle(paper_ids)
    paper_ids = paper_ids[:n_papers]

    captions_path = os.path.join(IMAGE_DIR, "captions.json")
    author_captions_path = os.path.join(IMAGE_DIR, "author_captions.json")
    captions = json.load(open(captions_path)) if os.path.exists(captions_path) else {}
    author_captions = (
        json.load(open(author_captions_path))
        if os.path.exists(author_captions_path)
        else {}
    )

    gold_qa: list[dict] = []
    n_pages_equiv = n_images = n_skipped_text = n_missing_img = 0

    for i, pid in enumerate(paper_ids, 1):
        entry = meta[pid]

        # ── text: one corpus file per paper ───────────────────────────────────
        text = load_paragraphs(pid, para_files)
        if not text:
            n_skipped_text += 1
            print(f"[warn] no paragraphs for {pid}; skipping paper")
            continue
        with open(os.path.join(CORPUS_DIR, f"{pid}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        n_pages_equiv += max(1, len(text) // 3000)  # rough page-equivalent tally

        # ── images: copy pre-extracted figure/table crops ─────────────────────
        for fname, fmeta in entry.get("all_figures", {}).items():
            src = img_files.get(fname)
            if src is None:
                n_missing_img += 1
                continue
            dst = os.path.join(IMAGE_DIR, fname)
            if not os.path.exists(dst):
                shutil.copyfile(src, dst)
            if isinstance(fmeta, dict) and fmeta.get("caption"):
                author_captions[fname] = fmeta["caption"]
            if fname not in captions:  # resumable: skip if captioned
                captions[fname] = caption_image(Path(dst), smoke=smoke)
            n_images += 1

        # ── native gold QA (kept for the optional extra evaluation) ───────────
        for qa in entry.get("qa", []):
            item = dict(qa)
            item["paper_id"] = pid
            gold_qa.append(item)

        # flush captions incrementally so an interrupted run loses nothing
        if i % 10 == 0 or i == len(paper_ids):
            json.dump(
                captions,
                open(captions_path, "w", encoding="utf-8"),
                indent=2,
                ensure_ascii=False,
            )
            json.dump(
                author_captions,
                open(author_captions_path, "w", encoding="utf-8"),
                indent=2,
                ensure_ascii=False,
            )
            print(
                f"  [{i}/{len(paper_ids)}] {pid}: "
                f"{len(entry.get('all_figures', {}))} figures, "
                f"{len(entry.get('qa', []))} native QA"
            )

        if smoke and i >= 3:
            break

    with open("spiqa_gold_qa.json", "w", encoding="utf-8") as f:
        json.dump(gold_qa, f, indent=2, ensure_ascii=False)

    print("\n=== SPIQA ingestion complete ===")
    print(
        f"Papers ingested     : {len(paper_ids) - n_skipped_text} "
        f"(~{n_pages_equiv} page-equivalents)"
    )
    print(f"Image crops         : {n_images}  -> {IMAGE_DIR}/")
    print(f"GPT-4o captions     : {len(captions)}  -> {captions_path}")
    print(
        f"Author captions     : {len(author_captions)} (sidecar, not used by pipeline)"
    )
    print(f"Native gold QA kept : {len(gold_qa)}  -> spiqa_gold_qa.json")
    if n_skipped_text:
        print(f"[warn] {n_skipped_text} papers had no extracted paragraphs")
    if n_missing_img:
        print(
            f"[warn] {n_missing_img} figure entries had no matching file in the images zip"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Ingest SPIQA test-A into the RAG pipeline."
    )
    ap.add_argument(
        "--papers",
        type=int,
        default=100,
        help="number of papers to sample (default: 100, ~1000 page-equivalents)",
    )
    ap.add_argument(
        "--probe",
        action="store_true",
        help="inspect dataset structure for one paper; write nothing",
    )
    ap.add_argument(
        "--smoke", action="store_true", help="3 papers, verbose, no captioning"
    )
    args = ap.parse_args()
    if args.probe:
        probe()
    else:
        ingest(args.papers, smoke=args.smoke)
