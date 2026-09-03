"""
PubLayNet ingestion: turns dataset pages into the inputs the RAG pipeline consumes.

Takes a slice of the small-publaynet-wds dataset and converts each page into:
  publaynet_corpus/<KEY>.txt          - OCR'd text of the page's text/title/list/table regions
  publaynet_images/<KEY>_figN.png     - cropped figure/table images (for CLIP retrieval)
  publaynet_images/captions.json      - {image_filename: one-sentence caption}

PubLayNet region categories: 1=text 2=title 3=list 4=table 5=figure. Text/title/list/
table regions are OCR'd (Tesseract); figure and table regions are also cropped as images
and captioned with gpt-4o vision (so a table appears in both the text and the image set).

Setup:
    pip install datasets pillow pytesseract openai python-dotenv
    Install the Tesseract binary separately (on Windows, the UB-Mannheim build), and
    set its path below if it is not on PATH.
    The dataset may require: huggingface-cli login

Usage:
    python ingest_publaynet.py --limit 100          # process 100 pages
    python ingest_publaynet.py --limit 3 --smoke    # 3 pages, verbose, no captioning
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

# On Windows, Tesseract is usually not on PATH; point pytesseract at the binary.

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

CORPUS_DIR = "publaynet_corpus"
IMAGE_DIR = "publaynet_images"
CAPTION_MODEL = "gpt-4o"

CATEGORY = {1: "text", 2: "title", 3: "list", 4: "table", 5: "figure"}
TEXT_CATEGORIES = {"text", "title", "list", "table"}   # OCR'd into the corpus
IMAGE_CATEGORIES = {"figure", "table"}                 # cropped as images (table is in both)

DATASET_BASE = "https://huggingface.co/datasets/lhoestq/small-publaynet-wds/resolve/main/publaynet-train-{i:06d}.tar"


def load_slice(limit, shards=1):
    """Streams up to `limit` pages from the dataset (avoids downloading it all)."""
    urls = [DATASET_BASE.format(i=i) for i in range(shards)]
    dataset = load_dataset("webdataset", data_files={"train": urls},
                           split="train", streaming=True)
    for i, row in enumerate(dataset):
        if i >= limit:
            break
        yield row


def caption_image(crop, smoke=False):
    """Captions a figure/table crop with gpt-4o vision (skipped in smoke mode)."""
    if smoke:
        return "(caption skipped in smoke mode)"
    buf = io.BytesIO()
    crop.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    resp = client.chat.completions.create(
        model=CAPTION_MODEL, temperature=0,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text":
                 "Describe this figure or table from a scientific paper in one concise "
                 "sentence, focusing on what it shows. Be specific about chart type, "
                 "axes, or table contents if visible."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
    )
    return resp.choices[0].message.content.strip()


def ingest(limit, smoke=False):
    os.makedirs(CORPUS_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    captions = {}
    n_pages = n_text_regions = n_images = 0

    for row in load_slice(limit):
        key = row["__key__"]
        page = row["png"]
        annotations = row["json"]["annotations"]
        n_pages += 1

        page_text = []
        figure_count = 0

        for ann in annotations:
            category = CATEGORY.get(ann.get("category_id"))
            if category is None:
                continue
            x, y, w, h = ann["bbox"]
            crop = page.crop((int(x), int(y), int(x + w), int(y + h)))

            if category in TEXT_CATEGORIES:
                ocr_text = pytesseract.image_to_string(crop).strip()
                if ocr_text:
                    page_text.append(ocr_text)
                    n_text_regions += 1

            if category in IMAGE_CATEGORIES:
                image_name = f"{key}_{category}{figure_count}.png"
                crop.save(os.path.join(IMAGE_DIR, image_name))
                captions[image_name] = caption_image(crop, smoke=smoke)
                figure_count += 1
                n_images += 1

        # one .txt per page, holding all its OCR'd text regions
        if page_text:
            with open(os.path.join(CORPUS_DIR, f"{key}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(page_text))

        if smoke:
            print(f"\n--- {key} ---")
            print(f"  regions: {len(annotations)}, text regions OCR'd: {len(page_text)}, "
                  f"images: {figure_count}")
            if page_text:
                print(f"  first text (truncated): {page_text[0][:120]!r}")

    with open(os.path.join(IMAGE_DIR, "captions.json"), "w", encoding="utf-8") as f:
        json.dump(captions, f, indent=2, ensure_ascii=False)

    print(f"\n=== Ingestion complete ===")
    print(f"Pages processed : {n_pages}")
    print(f"Text regions    : {n_text_regions}  -> {CORPUS_DIR}/*.txt")
    print(f"Image crops     : {n_images}        -> {IMAGE_DIR}/*.png")
    print(f"Captions written: {len(captions)}   -> {IMAGE_DIR}/captions.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100, help="number of pages to process")
    ap.add_argument("--smoke", action="store_true", help="verbose, few pages, skip captioning")
    args = ap.parse_args()
    ingest(args.limit, smoke=args.smoke)