"""
DocLayNet ingestion: same output contract as ingestion.py (PubLayNet).

Produces:
  doclaynet_corpus/<page_hash>.txt      - OCR'd text regions, one file per page
  doclaynet_images/<page_hash>_<N>.png  - cropped Picture / Table images
  doclaynet_images/captions.json        - {filename: one-sentence caption}

DocLayNet has 11 label IDs.  We remap them onto the same five functional
categories used throughout the pipeline:

  DocLayNet ID  Label            -> Pipeline category
  ───────────────────────────────────────────────────
  1             Caption          -> text   (short region, OCR'd)
  2             Footnote         -> text
  3             Formula          -> text   (OCR'd as text; also saved as image)
  4             List-item        -> text
  5             Page-footer      -> text
  6             Page-header      -> text
  7             Picture          -> figure (cropped as image + captioned)
  8             Section-header   -> title
  9             Table            -> table  (OCR'd AND saved as image, like PubLayNet)
  10            Text             -> text
  11            Title            -> title

Formulas are saved as images in addition to being OCR'd because they are
visually rich and CLIP retrieval of formula crops is useful for figure questions.

Dataset: docling-project/DocLayNet-v1.2  (CDLA-Permissive 1.0)
HF columns per row:
  row["image"]                  - PIL Image (1025 × 1025 PNG)
  row["objects"]["category"]    - list of label indices (1-based, see above)
  row["objects"]["bbox"]        - list of [x, y, w, h] in the 1025px coordinate space
  row["image_id"]               - unique integer id  (used as page key)
  row["doc_category"]           - e.g. "financial_reports", "laws_and_regulations"

Optionally filter to a single doc_category with --category.

Setup:
    pip install datasets pillow pytesseract openai python-dotenv
    (same dependencies as ingestion.py; Tesseract must be installed separately)

Usage:
    python ingest_doclaynet.py --limit 1000
    python ingest_doclaynet.py --limit 1000 --category financial_reports
    python ingest_doclaynet.py --limit 5   --smoke
"""

import os
import io
import json
import base64
import argparse
from PIL import Image
import pytesseract
from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv

# ── Windows: uncomment and set path if Tesseract is not on PATH ──────────────
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

CORPUS_DIR    = "doclaynet_corpus"
IMAGE_DIR     = "doclaynet_images"
CAPTION_MODEL = "gpt-4o"
HF_DATASET    = "docling-project/DocLayNet-v1.2"

# ── Label remap: DocLayNet 1-based IDs → pipeline functional category ─────────
# Keys are the integer category values stored in row["objects"]["category"].
DOCLAYNET_LABEL = {
    1:  "Caption",
    2:  "Footnote",
    3:  "Formula",
    4:  "List-item",
    5:  "Page-footer",
    6:  "Page-header",
    7:  "Picture",
    8:  "Section-header",
    9:  "Table",
    10: "Text",
    11: "Title",
}

CATEGORY_REMAP = {
    "Caption":        "text",
    "Footnote":       "text",
    "Formula":        "text",    # also saved as image (see IMAGE_CATEGORIES)
    "List-item":      "text",
    "Page-footer":    "text",
    "Page-header":    "text",
    "Picture":        "figure",
    "Section-header": "title",
    "Table":          "table",
    "Text":           "text",
    "Title":          "title",
}

# Functional categories that feed the text corpus (OCR'd)
TEXT_CATEGORIES  = {"text", "title", "table"}

# Functional categories that are also saved as cropped images for CLIP
# "table" appears in both (same as PubLayNet); "formula" saved via its own flag below
IMAGE_CATEGORIES = {"figure", "table"}

# DocLayNet original labels that get an image crop even though their functional
# category maps to "text" (Formulas are visually distinctive for CLIP retrieval)
ALSO_SAVE_AS_IMAGE = {"Formula"}


def caption_image(crop: Image.Image, smoke: bool = False) -> str:
    """Caption a figure/table/formula crop with gpt-4o vision."""
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
                 "Describe this region from a document in one concise sentence. "
                 "It may be a figure, chart, table, mathematical formula, or diagram. "
                 "Be specific about the content, chart type, axes, or table structure "
                 "where visible."},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
    )
    return resp.choices[0].message.content.strip()


def ingest(limit: int, smoke: bool = False, category_filter: str | None = None) -> None:
    os.makedirs(CORPUS_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR,  exist_ok=True)

    # Stream the train split; DocLayNet-v1.2 is large so we take only `limit` rows.
    # trust_remote_code=False is safe; the dataset uses standard image+annotation fields.
    dataset = load_dataset(HF_DATASET, split="train", streaming=True,
                           trust_remote_code=False)

    captions: dict[str, str] = {}

    # Load existing captions so interrupted runs can be resumed without re-captioning.
    captions_path = os.path.join(IMAGE_DIR, "captions.json")
    if os.path.exists(captions_path):
        with open(captions_path, encoding="utf-8") as f:
            captions = json.load(f)

    n_pages = n_text_regions = n_images = 0

    for row in dataset:
        if n_pages >= limit:
            break

        # ── Optional domain filter ────────────────────────────────────────────
        if category_filter and row.get("doc_category") != category_filter:
            continue

        # ── Page image and key ────────────────────────────────────────────────
        page: Image.Image = row["image"]                 # already a PIL Image
        page_hash: str    = str(row["image_id"])         # unique across the dataset
        doc_category: str = row.get("doc_category", "unknown")

        # ── Annotations ───────────────────────────────────────────────────────
        # DocLayNet-v1.2 stores annotations as a dict of parallel lists under
        # row["objects"]: {"category": [...], "bbox": [...], ...}
        objects     = row["objects"]
        cat_ids     = objects["category"]    # list[int], 1-based label IDs
        bboxes      = objects["bbox"]        # list[ [x, y, w, h] ]

        page_text: list[str] = []
        figure_count = 0

        for cat_id, bbox in zip(cat_ids, bboxes):
            label    = DOCLAYNET_LABEL.get(cat_id)
            if label is None:
                continue
            category = CATEGORY_REMAP[label]

            x, y, w, h = bbox
            # Guard against degenerate boxes (can occur in a small fraction of pages)
            if w < 4 or h < 4:
                continue

            crop = page.crop((int(x), int(y), int(x + w), int(y + h)))

            # ── OCR text regions ──────────────────────────────────────────────
            if category in TEXT_CATEGORIES:
                ocr_text = pytesseract.image_to_string(crop).strip()
                if ocr_text:
                    page_text.append(ocr_text)
                    n_text_regions += 1

            # ── Save image crops ──────────────────────────────────────────────
            save_as_image = (category in IMAGE_CATEGORIES) or (label in ALSO_SAVE_AS_IMAGE)
            if save_as_image:
                image_name = f"{page_hash}_{label.lower().replace('-', '_')}{figure_count}.png"
                crop.save(os.path.join(IMAGE_DIR, image_name))
                if image_name not in captions:          # skip if already captioned
                    captions[image_name] = caption_image(crop, smoke=smoke)
                figure_count += 1
                n_images += 1

        # ── Write text corpus file ────────────────────────────────────────────
        # Prefix the doc_category so multi-hop questions can span sub-domains.
        if page_text:
            corpus_path = os.path.join(CORPUS_DIR, f"{page_hash}.txt")
            with open(corpus_path, "w", encoding="utf-8") as f:
                f.write(f"[doc_category: {doc_category}]\n")
                f.write("\n".join(page_text))

        n_pages += 1

        if smoke:
            print(f"\n--- {page_hash} ({doc_category}) ---")
            print(f"  annotations: {len(cat_ids)}, "
                  f"text regions OCR'd: {len(page_text)}, "
                  f"image crops: {figure_count}")
            if page_text:
                print(f"  first text (truncated): {page_text[0][:120]!r}")

    # ── Persist captions (incremental: safe to re-run) ────────────────────────
    with open(captions_path, "w", encoding="utf-8") as f:
        json.dump(captions, f, indent=2, ensure_ascii=False)

    print(f"\n=== DocLayNet ingestion complete ===")
    print(f"Pages processed : {n_pages}")
    print(f"Text regions    : {n_text_regions}  -> {CORPUS_DIR}/*.txt")
    print(f"Image crops     : {n_images}         -> {IMAGE_DIR}/*.png")
    print(f"Captions written: {len(captions)}    -> {IMAGE_DIR}/captions.json")
    if category_filter:
        print(f"Domain filter   : {category_filter}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ingest DocLayNet into the RAG pipeline.")
    ap.add_argument("--limit",    type=int,  default=1000,
                    help="number of pages to process (default: 1000)")
    ap.add_argument("--category", type=str,  default=None,
                    help="filter to a single doc_category, e.g. financial_reports | "
                         "laws_and_regulations | scientific_articles | "
                         "government_tenders | manuals | patents")
    ap.add_argument("--smoke",    action="store_true",
                    help="verbose output, few pages, skip captioning")
    args = ap.parse_args()
    ingest(args.limit, smoke=args.smoke, category_filter=args.category)
