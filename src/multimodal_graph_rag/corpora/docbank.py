"""DocBank ingestion: token-level annotations assembled into page text and crops.

DocBank stores one record per token, so rows are grouped by page image before
processing. Text is assembled from tokens in reading order (no OCR); figure,
table, and equation regions are the union of their token boxes.

Produces
    <corpus_dir>/<page_hash>.txt
    <image_dir>/<page_hash>_<label>N.png
    <image_dir>/captions.json
"""

from __future__ import annotations

import hashlib
import io
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from PIL import Image

from ..clients import ModelClient
from .captioning import CaptionStore

HF_DATASET = "maveriq/DocBank"
LABEL_TO_CATEGORY = {
    "abstract": "text",
    "author": "text",
    "caption": "text",
    "date": "text",
    "equation": "text",
    "figure": "figure",
    "footer": "text",
    "list": "text",
    "paragraph": "text",
    "reference": "text",
    "section": "title",
    "table": "table",
    "title": "title",
}
TEXT_CATEGORIES = {"text", "title", "table"}
IMAGE_LABELS = ("figure", "table", "equation")
PLACEHOLDER = "##LTLine##"
BBOX_MARGIN = 4
MIN_CROP_SIDE = 8


@dataclass(frozen=True)
class IngestReport:
    pages: int
    text_parts: int
    image_crops: int
    captions: int


def image_hash(image_bytes: bytes) -> str:
    return hashlib.md5(image_bytes).hexdigest()[:16]


def union_bbox(boxes: list[list[int]]) -> tuple[int, int, int, int]:
    return (
        max(0, min(b[0] for b in boxes) - BBOX_MARGIN),
        max(0, min(b[1] for b in boxes) - BBOX_MARGIN),
        max(b[2] for b in boxes) + BBOX_MARGIN,
        max(b[3] for b in boxes) + BBOX_MARGIN,
    )


def process_page(
    page_key: str,
    image_bytes: bytes,
    tokens: list[dict],
    *,
    corpus: Path,
    images: Path,
    captions: CaptionStore,
    client: ModelClient | None,
) -> tuple[int, int]:
    page = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = page.size
    by_label: dict[str, list[dict]] = defaultdict(list)
    for token in tokens:
        by_label[token["label"]].append(token)

    text_parts: list[str] = []
    for label, group in by_label.items():
        if LABEL_TO_CATEGORY.get(label) not in TEXT_CATEGORIES:
            continue
        ordered = sorted(group, key=lambda t: (t["bbox"][1], t["bbox"][0]))
        words = [
            t["word"] for t in ordered if t["word"] != PLACEHOLDER and t["word"].strip()
        ]
        if words:
            text_parts.append(" ".join(words))
    if text_parts:
        (corpus / f"{page_key}.txt").write_text("\n".join(text_parts), encoding="utf-8")

    crops = 0
    for label in IMAGE_LABELS:
        group = by_label.get(label, [])
        if not group:
            continue
        x0, y0, x1, y1 = union_bbox([t["bbox"] for t in group])
        x1, y1 = min(width, x1), min(height, y1)
        if (x1 - x0) < MIN_CROP_SIDE or (y1 - y0) < MIN_CROP_SIDE:
            continue
        name = f"{page_key}_{label}{crops}.png"
        crop_path = images / name
        page.crop((x0, y0, x1, y1)).save(crop_path)
        captions.ensure(client, name, crop_path)
        crops += 1
    return len(text_parts), crops


def ingest(
    *,
    corpus_dir: str | Path,
    image_dir: str | Path,
    client: ModelClient | None,
    limit: int = 1000,
    smoke: bool = False,
) -> IngestReport:
    corpus = Path(corpus_dir)
    images = Path(image_dir)
    corpus.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    captions = CaptionStore(images)
    dataset = load_dataset(HF_DATASET, split="train", streaming=True)

    pages = text_total = crops_total = 0
    current_key: str | None = None
    current_bytes: bytes | None = None
    current_tokens: list[dict] = []

    def flush() -> None:
        nonlocal text_total, crops_total
        if current_key is None or not current_tokens:
            return
        n_text, n_crops = process_page(
            current_key,
            current_bytes,
            current_tokens,
            corpus=corpus,
            images=images,
            captions=captions,
            client=client,
        )
        text_total += n_text
        crops_total += n_crops
        captions.save()
        if smoke:
            print(f"--- {current_key}: {n_text} text parts, {n_crops} crops")

    for row in dataset:
        if pages >= limit:
            break
        field = row["image"]
        if isinstance(field, dict):
            image_bytes = field["bytes"]
        else:
            buffer = io.BytesIO()
            field.save(buffer, format="JPEG")
            image_bytes = buffer.getvalue()
        page_key = image_hash(image_bytes)
        if page_key != current_key:
            flush()
            if current_key is not None:
                pages += 1
                if pages >= limit:
                    break
            current_key, current_bytes, current_tokens = page_key, image_bytes, []
        if row["label"] not in LABEL_TO_CATEGORY:
            continue
        raw_bbox = row["bounding_box"]
        bbox = (
            list(raw_bbox[0])
            if isinstance(raw_bbox[0], (list, tuple))
            else list(raw_bbox)
        )
        current_tokens.append(
            {"word": row["token"], "bbox": bbox, "label": row["label"]}
        )
    flush()
    if current_tokens:
        pages += 1
    captions.save()
    return IngestReport(
        pages=pages,
        text_parts=text_total,
        image_crops=crops_total,
        captions=len(captions),
    )
