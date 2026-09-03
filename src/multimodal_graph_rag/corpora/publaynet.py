"""PubLayNet ingestion: OCR'd page text plus cropped, captioned figures and tables.

Produces
    <corpus_dir>/<key>.txt            OCR text of the page's text regions
    <image_dir>/<key>_<category>N.png cropped figure and table regions
    <image_dir>/captions.json         one-sentence captions
    <image_dir>/progress.json         checkpoint for resumable runs
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import requests
from datasets import load_dataset

from ..clients import ModelClient
from ..errors import ConfigurationError
from .captioning import CaptionStore, configure_tesseract, ocr_text

DATASET_BASE = (
    "https://huggingface.co/datasets/lhoestq/small-publaynet-wds"
    "/resolve/main/publaynet-train-{index:06d}.tar"
)
CATEGORY = {1: "text", 2: "title", 3: "list", 4: "table", 5: "figure"}
TEXT_CATEGORIES = {"text", "title", "list", "table"}
IMAGE_CATEGORIES = {"figure", "table"}
MIN_REGION_SIDE = 4


@dataclass(frozen=True)
class IngestReport:
    pages_processed: int
    pages_total: int
    text_regions: int
    image_crops: int
    captions: int


def discover_shards(max_probe: int = 12, timeout: float = 10.0) -> list[str]:
    """URLs of the shards that exist; shards are contiguous so stop at the first miss."""
    urls = []
    for index in range(max_probe):
        url = DATASET_BASE.format(index=index)
        response = requests.head(url, allow_redirects=True, timeout=timeout)
        if response.status_code != 200:
            break
        urls.append(url)
    if not urls:
        raise ConfigurationError(f"no PubLayNet shards found at {DATASET_BASE}")
    return urls


def stream_pages(limit: int, shards: int | None = None) -> Iterator[dict]:
    urls = discover_shards()
    if shards is not None:
        urls = urls[:shards]
    dataset = load_dataset(
        "webdataset", data_files={"train": urls}, split="train", streaming=True
    )
    for seen, row in enumerate(dataset):
        if seen >= limit:
            break
        yield row


class Progress:
    def __init__(self, image_dir: Path) -> None:
        self.path = image_dir / "progress.json"
        self.done: set[str] = set()
        self.text_regions = 0
        self.image_crops = 0
        if self.path.is_file():
            with self.path.open(encoding="utf-8") as stream:
                raw = json.load(stream)
            self.done = set(raw["done"])
            self.text_regions = int(raw["n_text"])
            self.image_crops = int(raw["n_images"])

    def save(self) -> None:
        with self.path.open("w", encoding="utf-8") as stream:
            json.dump(
                {
                    "done": sorted(self.done),
                    "n_text": self.text_regions,
                    "n_images": self.image_crops,
                },
                stream,
            )


def ingest(
    *,
    corpus_dir: str | Path,
    image_dir: str | Path,
    client: ModelClient | None,
    limit: int = 1000,
    shards: int | None = None,
    smoke: bool = False,
) -> IngestReport:
    """Stream pages, OCR text regions, crop and caption visual regions.

    ``client=None`` (smoke mode) skips captioning entirely.
    """
    configure_tesseract()
    corpus = Path(corpus_dir)
    images = Path(image_dir)
    corpus.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    captions = CaptionStore(images)
    progress = Progress(images)
    processed = 0

    for row in stream_pages(limit, shards):
        key = row["__key__"]
        if key in progress.done:
            continue
        page = row["png"]
        page_text: list[str] = []
        figure_count = 0
        for annotation in row["json"]["annotations"]:
            category = CATEGORY.get(annotation.get("category_id"))
            if category is None:
                continue
            x, y, w, h = annotation["bbox"]
            if w < MIN_REGION_SIDE or h < MIN_REGION_SIDE:
                continue
            crop = page.crop((int(x), int(y), int(x + w), int(y + h)))
            if category in TEXT_CATEGORIES:
                text = ocr_text(crop)
                if text:
                    page_text.append(text)
                    progress.text_regions += 1
            if category in IMAGE_CATEGORIES:
                name = f"{key}_{category}{figure_count}.png"
                crop_path = images / name
                crop.save(crop_path)
                captions.ensure(client, name, crop_path)
                figure_count += 1
                progress.image_crops += 1
        if page_text:
            (corpus / f"{key}.txt").write_text("\n".join(page_text), encoding="utf-8")
        progress.done.add(key)
        processed += 1
        captions.save()
        progress.save()
        if smoke:
            print(f"--- {key}: {len(page_text)} text regions, {figure_count} crops")
        elif processed % 10 == 0:
            print(
                f"  [{len(progress.done)}/{limit}] pages | text regions {progress.text_regions} | crops {progress.image_crops}"
            )

    report = IngestReport(
        pages_processed=processed,
        pages_total=len(progress.done),
        text_regions=progress.text_regions,
        image_crops=progress.image_crops,
        captions=len(captions),
    )
    if report.pages_total < limit and not smoke:
        print(f"requested {limit} pages but the dataset yielded {report.pages_total}")
    return report
