"""
PubLayNet ingestion — upgraded for 1,000-page runs.

Produces (same contract as before, downstream pipeline unchanged):
  publaynet_corpus/<KEY>.txt       – OCR'd text regions, one file per page
  publaynet_images/<KEY>_*.png     – cropped figure / table images
  publaynet_images/captions.json   – {filename: caption}

Upgrades over the original 100-page version
───────────────────────────────────────────
1. Shard auto-discovery.  load_dataset() validates every URL up-front, so a
   single non-existent shard raises FileNotFoundError and nothing loads.
   The script now HEAD-probes shard URLs and streams only those that exist,
   so it can't crash on a bad shard count and adapts if the dataset grows.
   (HF's own docs for this dataset use range(4) — four shards.)

2. Checkpoint / resume.  Progress is written to
   publaynet_images/progress.json after every page.  If the run crashes
   (network blip, rate-limit, power loss) restart with the same command and
   it skips already-processed pages automatically.

3. Retry + backoff on GPT-4o caption calls.  Up to 5 retries with
   exponential backoff handles transient 429 / 5xx errors without crashing
   the whole run.

4. Progress bar.  Prints a one-line status every 10 pages so you can see
   the run is alive during the multi-hour captioning phase.

Setup (unchanged):
    pip install datasets pillow pytesseract openai python-dotenv
    Install Tesseract separately; on Windows set TESSERACT_CMD below.

Usage:
    python ingest_publaynet.py --limit 1000          # full paper run
    python ingest_publaynet.py --limit 5   --smoke   # quick test, no captions
    python ingest_publaynet.py --limit 1000          # safe to re-run; resumes
    python ingest_publaynet.py --limit 1000 --shards 2   # cap shards if wanted
"""

import os
import io
import json
import time
import base64
import argparse
import requests
from PIL import Image
import pytesseract
from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv

# ── Windows: set path to Tesseract binary if not on PATH ─────────────────────
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

CORPUS_DIR    = "publaynet_corpus"
IMAGE_DIR     = "publaynet_images"
CAPTION_MODEL = "gpt-4o"
DATASET_BASE  = (
    "https://huggingface.co/datasets/lhoestq/small-publaynet-wds"
    "/resolve/main/publaynet-train-{i:06d}.tar"
)

CATEGORY = {1: "text", 2: "title", 3: "list", 4: "table", 5: "figure"}
TEXT_CATEGORIES  = {"text", "title", "list", "table"}
IMAGE_CATEGORIES = {"figure", "table"}


# ── Retry wrapper ─────────────────────────────────────────────────────────────
def caption_image(crop: Image.Image, smoke: bool = False,
                  max_retries: int = 5) -> str:
    """Caption a crop with GPT-4o vision.  Retries on 429 / 5xx with backoff."""
    if smoke:
        return "(caption skipped in smoke mode)"
    buf = io.BytesIO()
    crop.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=CAPTION_MODEL,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text":
                         "Describe this figure or table from a scientific paper in one "
                         "concise sentence, focusing on what it shows. Be specific about "
                         "chart type, axes, or table contents if visible."},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            wait = 2 ** attempt          # 1 s, 2 s, 4 s, 8 s, 16 s
            print(f"  [caption] attempt {attempt + 1} failed ({e}); "
                  f"retrying in {wait}s …")
            time.sleep(wait)

    print("  [caption] all retries exhausted — using placeholder.")
    return "(caption failed)"


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def _progress_path() -> str:
    return os.path.join(IMAGE_DIR, "progress.json")

def _load_progress() -> dict:
    """Returns {done: set[key], n_text: int, n_images: int}."""
    p = _progress_path()
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
        return {"done": set(raw["done"]),
                "n_text": raw["n_text"],
                "n_images": raw["n_images"]}
    return {"done": set(), "n_text": 0, "n_images": 0}

def _save_progress(progress: dict) -> None:
    with open(_progress_path(), "w", encoding="utf-8") as f:
        json.dump({"done":     list(progress["done"]),
                   "n_text":   progress["n_text"],
                   "n_images": progress["n_images"]}, f)


# ── Dataset loader ────────────────────────────────────────────────────────────
def discover_shards(max_probe: int = 12) -> list:
    """
    Return the URLs of shards that actually exist.

    Why this exists: load_dataset() validates every URL up-front, so a single
    missing shard raises FileNotFoundError and nothing loads at all. Probing
    first means the script self-discovers the real shard count instead of us
    hard-coding a guess — and it picks up new shards automatically if the
    dataset ever grows.

    HF's own docs for this dataset use range(4), so 4 is the expected answer;
    max_probe=12 just leaves headroom.
    """
    urls = []
    for i in range(max_probe):
        url = DATASET_BASE.format(i=i)
        try:
            # HEAD is enough to know it's there; don't download the tar.
            r = requests.head(url, allow_redirects=True, timeout=10)
            if r.status_code == 200:
                urls.append(url)
            else:
                break        # shards are contiguous; first miss ends the run
        except requests.RequestException:
            break
    return urls


def load_stream(limit: int, shards: int | None = None):
    """
    Stream up to `limit` pages.

    shards=None  → auto-discover how many exist (recommended)
    shards=N     → use the first N, still filtered to those that resolve
    """
    urls = discover_shards()
    if not urls:
        raise RuntimeError(
            "No shards found. Check your connection, or whether "
            f"{DATASET_BASE.format(i=0)} still exists."
        )
    if shards is not None:
        urls = urls[:shards]

    print(f"Found {len(urls)} shard(s); streaming from them.")

    dataset = load_dataset(
        "webdataset",
        data_files={"train": urls},
        split="train",
        streaming=True,
    )
    seen = 0
    for row in dataset:
        if seen >= limit:
            break
        yield row
        seen += 1


# ── Main ingestion ────────────────────────────────────────────────────────────
def ingest(limit: int, shards: int | None = None, smoke: bool = False) -> None:
    os.makedirs(CORPUS_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR,  exist_ok=True)

    # Load captions + progress so interrupted runs resume cleanly.
    captions_path = os.path.join(IMAGE_DIR, "captions.json")
    captions: dict[str, str] = {}
    if os.path.exists(captions_path):
        with open(captions_path, encoding="utf-8") as f:
            captions = json.load(f)

    progress = _load_progress()
    already_done = len(progress["done"])
    if already_done:
        print(f"Resuming: {already_done} pages already processed, "
              f"{limit - already_done} remaining.")

    n_pages    = already_done          # total including prior runs
    n_new      = 0                     # pages processed this run
    n_text     = progress["n_text"]
    n_images   = progress["n_images"]

    for row in load_stream(limit, shards=shards):
        key = row["__key__"]

        # Skip pages already done in a previous (interrupted) run.
        if key in progress["done"]:
            continue

        page        = row["png"]
        annotations = row["json"]["annotations"]

        page_text    = []
        figure_count = 0

        for ann in annotations:
            category = CATEGORY.get(ann.get("category_id"))
            if category is None:
                continue
            x, y, w, h = ann["bbox"]
            # Guard against degenerate boxes.
            if w < 4 or h < 4:
                continue
            crop = page.crop((int(x), int(y), int(x + w), int(y + h)))

            if category in TEXT_CATEGORIES:
                ocr_text = pytesseract.image_to_string(crop).strip()
                if ocr_text:
                    page_text.append(ocr_text)
                    n_text += 1

            if category in IMAGE_CATEGORIES:
                image_name = f"{key}_{category}{figure_count}.png"
                crop.save(os.path.join(IMAGE_DIR, image_name))
                if image_name not in captions:
                    captions[image_name] = caption_image(crop, smoke=smoke)
                figure_count += 1
                n_images += 1

        # Write text corpus file.
        if page_text:
            with open(os.path.join(CORPUS_DIR, f"{key}.txt"),
                      "w", encoding="utf-8") as f:
                f.write("\n".join(page_text))

        # Mark page as done and checkpoint.
        progress["done"].add(key)
        progress["n_text"]   = n_text
        progress["n_images"] = n_images
        n_pages += 1
        n_new   += 1

        # Persist captions + progress every page (safe to interrupt).
        with open(captions_path, "w", encoding="utf-8") as f:
            json.dump(captions, f, indent=2, ensure_ascii=False)
        _save_progress(progress)

        if smoke:
            print(f"\n--- {key} ---")
            print(f"  regions: {len(annotations)}, "
                  f"text regions OCR'd: {len(page_text)}, "
                  f"images: {figure_count}")
            if page_text:
                print(f"  first text (truncated): {page_text[0][:120]!r}")
        elif n_new % 10 == 0:
            print(f"  [{n_pages}/{limit}] pages done  "
                  f"| text regions: {n_text}  | image crops: {n_images}")

    print(f"\n=== PubLayNet ingestion complete ===")
    print(f"Pages processed (this run) : {n_new}")
    print(f"Pages processed (total)    : {n_pages}")
    print(f"Text regions               : {n_text}   -> {CORPUS_DIR}/*.txt")
    print(f"Image crops                : {n_images}  -> {IMAGE_DIR}/*.png")
    print(f"Captions                   : {len(captions)} -> {captions_path}")

    # The shards may simply not hold as many pages as requested. Say so plainly
    # rather than letting a short corpus pass unnoticed into the experiment.
    if n_pages < limit and not smoke:
        print(f"\n  ! Requested {limit} pages but the dataset yielded {n_pages}.")
        print(f"    small-publaynet-wds is a subset and may not contain "
              f"{limit} pages.")
        print(f"    Options: proceed with {n_pages} (fine if close to target), "
              f"or switch")
        print(f"    to the full PubLayNet source for more shards.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Ingest PubLayNet into the RAG pipeline (1,000-page version).")
    ap.add_argument("--limit",  type=int, default=1000,
                    help="total pages to ingest (default: 1000)")
    ap.add_argument("--shards", type=int, default=None,
                    help="cap the number of shards used. Default: auto-discover "
                         "all that exist (HF docs indicate 4 for this dataset)")
    ap.add_argument("--smoke",  action="store_true",
                    help="verbose, 5-page test run, skips captioning")
    args = ap.parse_args()
    ingest(args.limit, shards=args.shards, smoke=args.smoke)