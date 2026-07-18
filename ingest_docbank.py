"""
DocBank ingestion: same output contract as ingestion.py (PubLayNet).

Produces:
  docbank_corpus/<page_id>.txt      - OCR-free text per page, assembled from tokens
  docbank_images/<page_id>_<N>.png  - cropped figure / table / equation image regions
  docbank_images/captions.json      - {filename: one-sentence caption}

── Why DocBank is different from PubLayNet / DocLayNet ──────────────────────
PubLayNet and DocLayNet store one annotation record per region.
DocBank stores one record per TOKEN.  Every token carries:
  row["image"]         bytes of the page JPEG (same image repeated for every token)
  row["token"]         the word string, e.g. "Abstract"
  row["bounding_box"]  [[x0, y0, x1, y1]] in the page's pixel coordinate space
  row["label"]         semantic label string (one of 13 values, see below)

So we must GROUP rows by image identity before we can do anything.  We collect
all tokens that share the same image bytes hash, then process the page once.

── DocBank label set and pipeline mapping ───────────────────────────────────

  DocBank label   -> Pipeline category    Action
  ─────────────────────────────────────────────────────────────────────────
  abstract        -> text                 text assembled from tokens
  author          -> text
  caption         -> text
  date            -> text
  equation        -> text  + image        tokens used for text; region also saved
  figure          -> figure               region saved as image + captioned
  footer          -> text
  list            -> text
  paragraph       -> text
  reference       -> text
  section         -> title
  table           -> table                tokens used for text; region also saved
  title           -> title

Because DocBank has no OCR step (the tokens ARE the text), we assemble text by
sorting tokens top-to-bottom, left-to-right within each label group and joining
them with spaces.  This replaces the pytesseract call entirely for the text track.

Pytesseract IS still used to caption image crops (figure, table, equation) so
the dependency stays the same as ingestion.py.

── How region bounding boxes are recovered ──────────────────────────────────
DocBank bboxes are per-token, not per-region.  For figure, table, and equation
labels we take the union of all token bboxes with that label on the page as a
single region box.  (DocBank documents rarely have more than one figure per page
so this works well in practice.  If needed, a DBSCAN cluster step could split
multiple figures; we skip that complexity here.)

For text labels we simply join the token strings in reading order.

── Dataset access ────────────────────────────────────────────────────────────
Dataset: maveriq/DocBank  (Apache-2.0)
Each HF row:  image (PIL bytes), token (str), bounding_box ([[x0,y0,x1,y1]]),
              color ([[r,g,b]]), font (str), label (str)

We stream the train split and group rows into pages using a running hash of the
image bytes.  Because HF streams rows in page order (all tokens of one page are
contiguous), we can flush each page as soon as the image bytes change — no need
to buffer the whole dataset.

Setup:
    pip install datasets pillow pytesseract openai python-dotenv
    (same dependencies as ingestion.py; Tesseract must be installed separately)
    Note: pytesseract is only used for captioning image crops, not for OCR of text.

Usage:
    python ingest_docbank.py --limit 1000
    python ingest_docbank.py --limit 5 --smoke
"""

import os
import io
import json
import base64
import hashlib
import argparse
from collections import defaultdict
from PIL import Image
import pytesseract          # only used for captioning image crops
from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv

# ── Windows: uncomment and set path if Tesseract is not on PATH ──────────────
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

CORPUS_DIR    = "docbank_corpus"
IMAGE_DIR     = "docbank_images"
CAPTION_MODEL = "gpt-4o"
HF_DATASET    = "maveriq/DocBank"

# ── Label → functional category mapping ──────────────────────────────────────
LABEL_TO_CATEGORY = {
    "abstract":  "text",
    "author":    "text",
    "caption":   "text",
    "date":      "text",
    "equation":  "text",    # tokens assembled as text; region also saved as image
    "figure":    "figure",  # no tokens (placeholder '##LTLine##'); region saved as image
    "footer":    "text",
    "list":      "text",
    "paragraph": "text",
    "reference": "text",
    "section":   "title",
    "table":     "table",   # tokens assembled as text; region also saved as image
    "title":     "title",
}

# Functional categories that feed the text corpus (assembled from tokens)
TEXT_CATEGORIES = {"text", "title", "table"}

# Labels whose token bboxes are merged into a region crop saved as an image
IMAGE_LABELS = {"figure", "table", "equation"}

# DocBank placeholder token inserted for non-text regions (lines, rules, etc.)
PLACEHOLDER = "##LTLine##"


def _image_hash(img_bytes: bytes) -> str:
    """Short MD5 used as the page key / filename stem."""
    return hashlib.md5(img_bytes).hexdigest()[:16]


def caption_image(crop: Image.Image, smoke: bool = False) -> str:
    """Caption a figure / table / equation crop with gpt-4o vision."""
    if smoke:
        return "(caption skipped in smoke mode)"
    buf = io.BytesIO()
    crop.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    resp = client.chat.completions.create(
        model=CAPTION_MODEL,
        temperature=0,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text":
                 "Describe this region from a scientific document in one concise sentence. "
                 "It may be a figure, chart, table, or mathematical equation. "
                 "Be specific about the content, chart type, axes, or table structure "
                 "where visible."},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
    )
    return resp.choices[0].message.content.strip()


def _union_bbox(bboxes: list[list[int]]) -> tuple[int, int, int, int]:
    """Return the bounding rectangle that contains all token bboxes.

    Each bbox is [x0, y0, x1, y1] in pixel coordinates.
    We add a small margin (4px) so the crop doesn't clip edge pixels.
    """
    xs0 = [b[0] for b in bboxes]
    ys0 = [b[1] for b in bboxes]
    xs1 = [b[2] for b in bboxes]
    ys1 = [b[3] for b in bboxes]
    margin = 4
    return (max(0, min(xs0) - margin),
            max(0, min(ys0) - margin),
            max(xs1) + margin,
            max(ys1) + margin)


def _sort_key(item):
    """Sort tokens in reading order: top-to-bottom, then left-to-right."""
    bbox = item["bbox"]
    return (bbox[1], bbox[0])   # (y0, x0)


def process_page(page_key: str,
                 img_bytes: bytes,
                 tokens: list[dict],
                 captions: dict[str, str],
                 smoke: bool = False) -> tuple[int, int]:
    """
    Process one page worth of tokens.

    tokens is a list of dicts:
        {"word": str, "bbox": [x0, y0, x1, y1], "label": str}

    Returns (n_text_regions_written, n_image_crops_written).
    """
    # Decode the page image once
    page = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    W, H = page.size

    # ── Group tokens by label ─────────────────────────────────────────────────
    label_tokens: dict[str, list[dict]] = defaultdict(list)
    for tok in tokens:
        label_tokens[tok["label"]].append(tok)

    # ── Assemble text corpus ──────────────────────────────────────────────────
    text_parts: list[str] = []
    for label, toks in label_tokens.items():
        category = LABEL_TO_CATEGORY.get(label)
        if category not in TEXT_CATEGORIES:
            continue
        sorted_toks = sorted(toks, key=_sort_key)
        words = [t["word"] for t in sorted_toks
                 if t["word"] != PLACEHOLDER and t["word"].strip()]
        if words:
            text_parts.append(" ".join(words))

    n_text = 0
    if text_parts:
        corpus_path = os.path.join(CORPUS_DIR, f"{page_key}.txt")
        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write("\n".join(text_parts))
        n_text = len(text_parts)

    # ── Crop and caption image regions ───────────────────────────────────────
    n_images = 0
    figure_count = 0
    for label in IMAGE_LABELS:
        toks = label_tokens.get(label, [])
        if not toks:
            continue
        # Collect all bboxes for this label and take their union
        bboxes = [t["bbox"] for t in toks]
        x0, y0, x1, y1 = _union_bbox(bboxes)
        # Clamp to image dimensions
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if (x1 - x0) < 8 or (y1 - y0) < 8:
            continue   # skip degenerate crops

        crop = page.crop((x0, y0, x1, y1))
        image_name = f"{page_key}_{label}{figure_count}.png"
        crop.save(os.path.join(IMAGE_DIR, image_name))
        if image_name not in captions:
            captions[image_name] = caption_image(crop, smoke=smoke)
        figure_count += 1
        n_images += 1

    if smoke:
        label_summary = {lb: len(toks) for lb, toks in label_tokens.items()}
        print(f"\n--- {page_key} ---")
        print(f"  label counts : {label_summary}")
        print(f"  text parts   : {n_text}")
        print(f"  image crops  : {n_images}")
        if text_parts:
            print(f"  first text (truncated): {text_parts[0][:120]!r}")

    return n_text, n_images


def ingest(limit: int, smoke: bool = False) -> None:
    os.makedirs(CORPUS_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR,  exist_ok=True)

    # Load existing captions so interrupted runs can resume without re-captioning.
    captions_path = os.path.join(IMAGE_DIR, "captions.json")
    captions: dict[str, str] = {}
    if os.path.exists(captions_path):
        with open(captions_path, encoding="utf-8") as f:
            captions = json.load(f)

    # Stream rows.  All tokens for one page are contiguous in the HF stream,
    # so we flush whenever the image bytes change.
    dataset = load_dataset(HF_DATASET, split="train", streaming=True)

    n_pages = n_text_total = n_images_total = 0

    current_key: str | None      = None
    current_img_bytes: bytes | None = None
    current_tokens: list[dict]   = []

    def flush():
        nonlocal n_text_total, n_images_total
        if current_key is None or not current_tokens:
            return
        n_t, n_i = process_page(current_key, current_img_bytes,
                                 current_tokens, captions, smoke=smoke)
        n_text_total   += n_t
        n_images_total += n_i

    for row in dataset:
        if n_pages >= limit:
            break

        # ── Decode image bytes (same bytes for every token on this page) ──────
        img_field  = row["image"]
        # HF Image feature can be a dict {"bytes": ..., "path": ...} or a PIL Image
        if isinstance(img_field, dict):
            img_bytes = img_field["bytes"]
        else:
            buf = io.BytesIO()
            img_field.save(buf, format="JPEG")
            img_bytes = buf.getvalue()

        page_key = _image_hash(img_bytes)

        # ── Page boundary detection ───────────────────────────────────────────
        if page_key != current_key:
            flush()
            if current_key is not None:        # a full page was just completed
                n_pages += 1
                if n_pages >= limit:
                    break
            current_key       = page_key
            current_img_bytes = img_bytes
            current_tokens    = []

        # ── Collect token ─────────────────────────────────────────────────────
        label = row["label"]
        if label not in LABEL_TO_CATEGORY:
            continue

        # bounding_box stored as [[x0, y0, x1, y1]] (list of one list)
        raw_bbox = row["bounding_box"]
        if isinstance(raw_bbox[0], (list, tuple)):
            bbox = list(raw_bbox[0])   # unwrap outer list
        else:
            bbox = list(raw_bbox)

        current_tokens.append({
            "word":  row["token"],
            "bbox":  bbox,
            "label": label,
        })

    # Flush the last page
    flush()
    if current_tokens:
        n_pages += 1

    # Persist captions
    with open(captions_path, "w", encoding="utf-8") as f:
        json.dump(captions, f, indent=2, ensure_ascii=False)

    print(f"\n=== DocBank ingestion complete ===")
    print(f"Pages processed  : {n_pages}")
    print(f"Text parts written: {n_text_total}  -> {CORPUS_DIR}/*.txt")
    print(f"Image crops       : {n_images_total} -> {IMAGE_DIR}/*.png")
    print(f"Captions written  : {len(captions)}  -> {IMAGE_DIR}/captions.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ingest DocBank into the RAG pipeline.")
    ap.add_argument("--limit", type=int, default=1000,
                    help="number of pages to process (default: 1000)")
    ap.add_argument("--smoke", action="store_true",
                    help="verbose output, few pages, skip captioning")
    args = ap.parse_args()
    ingest(args.limit, smoke=args.smoke)
