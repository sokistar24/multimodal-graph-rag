"""DocLayNet ingestion: OCR'd page text plus cropped, captioned visual regions.

DocLayNet's eleven labels are remapped onto the pipeline's functional
categories. Formulas are OCR'd as text and also saved as crops because they
are visually distinctive for CLIP retrieval.

Produces
    <corpus_dir>/<image_id>.txt
    <image_dir>/<image_id>_<label>N.png
    <image_dir>/captions.json
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset

from ..clients import ModelClient
from .captioning import CaptionStore, configure_tesseract, ocr_text

HF_DATASET = "docling-project/DocLayNet-v1.2"
DOCLAYNET_LABEL = {
    1: "Caption",
    2: "Footnote",
    3: "Formula",
    4: "List-item",
    5: "Page-footer",
    6: "Page-header",
    7: "Picture",
    8: "Section-header",
    9: "Table",
    10: "Text",
    11: "Title",
}
CATEGORY_REMAP = {
    "Caption": "text",
    "Footnote": "text",
    "Formula": "text",
    "List-item": "text",
    "Page-footer": "text",
    "Page-header": "text",
    "Picture": "figure",
    "Section-header": "title",
    "Table": "table",
    "Text": "text",
    "Title": "title",
}
TEXT_CATEGORIES = {"text", "title", "table"}
IMAGE_CATEGORIES = {"figure", "table"}
ALSO_SAVE_AS_IMAGE = {"Formula"}
MIN_REGION_SIDE = 4


@dataclass(frozen=True)
class IngestReport:
    pages: int
    text_regions: int
    image_crops: int
    captions: int


def ingest(
    *,
    corpus_dir: str | Path,
    image_dir: str | Path,
    client: ModelClient | None,
    limit: int = 1000,
    category_filter: str | None = None,
    smoke: bool = False,
) -> IngestReport:
    configure_tesseract()
    corpus = Path(corpus_dir)
    images = Path(image_dir)
    corpus.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    captions = CaptionStore(images)
    dataset = load_dataset(HF_DATASET, split="train", streaming=True)

    pages = text_regions = crops = 0
    for row in dataset:
        if pages >= limit:
            break
        if category_filter and row.get("doc_category") != category_filter:
            continue
        page = row["image"]
        page_key = str(row["image_id"])
        doc_category = row.get("doc_category", "unknown")
        page_text: list[str] = []
        figure_count = 0
        for category_id, bbox in zip(
            row["objects"]["category"], row["objects"]["bbox"], strict=True
        ):
            label = DOCLAYNET_LABEL.get(category_id)
            if label is None:
                continue
            category = CATEGORY_REMAP[label]
            x, y, w, h = bbox
            if w < MIN_REGION_SIDE or h < MIN_REGION_SIDE:
                continue
            crop = page.crop((int(x), int(y), int(x + w), int(y + h)))
            if category in TEXT_CATEGORIES:
                text = ocr_text(crop)
                if text:
                    page_text.append(text)
                    text_regions += 1
            if category in IMAGE_CATEGORIES or label in ALSO_SAVE_AS_IMAGE:
                name = f"{page_key}_{label.lower().replace('-', '_')}{figure_count}.png"
                crop_path = images / name
                crop.save(crop_path)
                captions.ensure(client, name, crop_path)
                figure_count += 1
                crops += 1
        if page_text:
            (corpus / f"{page_key}.txt").write_text(
                f"[doc_category: {doc_category}]\n" + "\n".join(page_text),
                encoding="utf-8",
            )
        pages += 1
        captions.save()
        if smoke:
            print(
                f"--- {page_key} ({doc_category}): {len(page_text)} text regions, {figure_count} crops"
            )
    return IngestReport(
        pages=pages,
        text_regions=text_regions,
        image_crops=crops,
        captions=len(captions),
    )
