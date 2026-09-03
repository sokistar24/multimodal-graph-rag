"""SPIQA test-A ingestion: clean TeX-derived paper text and pre-extracted figures.

Produces
    <corpus_dir>/<paper_id>.txt        extracted paragraphs, one file per paper
    <image_dir>/<figure file>          figure and table images copied from SPIQA
    <image_dir>/captions.json          pipeline captions (same model as other corpora)
    <image_dir>/author_captions.json   original author captions (analysis only)
    <gold_qa_path>                     SPIQA's native QA for the sampled papers

The provenance unit is the paper, not the page, and there is no OCR: this is
the favourable text condition for the graph hypothesis.
"""

from __future__ import annotations

import json
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

from ..clients import ModelClient
from ..errors import ConfigurationError
from .captioning import CaptionStore

REPO_ID = "google/spiqa"
META_FILE = "test-A/SPIQA_testA.json"
IMAGES_ZIP = "test-A/SPIQA_testA_Images.zip"
PARAGRAPHS_ZIP = "SPIQA_train_val_test-A_extracted_paragraphs.zip"
DEFAULT_SEED = 42


@dataclass(frozen=True)
class IngestReport:
    papers: int
    papers_without_text: int
    images: int
    missing_images: int
    captions: int
    gold_qa: int


def fetch(filename: str, download_dir: Path) -> Path:
    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=str(download_dir),
        )
    )


def ensure_unzipped(zip_path: Path, target: Path) -> Path:
    if not target.exists() or not any(target.iterdir()):
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(target)
    return target


def index_files(root: Path) -> dict[str, Path]:
    return {path.name: path for path in root.rglob("*") if path.is_file()}


def load_paragraphs(paper_id: str, files: dict[str, Path]) -> str | None:
    """Paper text from either the ``.txt`` or the ``.json`` paragraph layout."""
    for suffix in (".txt", ".json"):
        path = files.get(paper_id + suffix)
        if path is None:
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        if suffix == ".txt":
            return raw.strip()
        data = json.loads(raw)
        if isinstance(data, dict):
            data = data.get("paragraphs") or data.get("text") or list(data.values())
        if isinstance(data, list):
            parts = [x if isinstance(x, str) else json.dumps(x) for x in data]
            return "\n\n".join(s.strip() for s in parts if s and s.strip())
        return str(data)
    return None


def ingest(
    *,
    corpus_dir: str | Path,
    image_dir: str | Path,
    download_dir: str | Path,
    gold_qa_path: str | Path,
    client: ModelClient | None,
    papers: int = 100,
    seed: int = DEFAULT_SEED,
    smoke: bool = False,
) -> IngestReport:
    corpus = Path(corpus_dir)
    images = Path(image_dir)
    downloads = Path(download_dir)
    corpus.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)

    with fetch(META_FILE, downloads).open(encoding="utf-8") as stream:
        meta = json.load(stream)
    paragraph_files = index_files(
        ensure_unzipped(fetch(PARAGRAPHS_ZIP, downloads), downloads / "paragraphs")
    )
    image_files = index_files(
        ensure_unzipped(fetch(IMAGES_ZIP, downloads), downloads / "images")
    )

    paper_ids = sorted(meta)
    random.Random(seed).shuffle(paper_ids)
    paper_ids = paper_ids[: 3 if smoke else papers]
    if not paper_ids:
        raise ConfigurationError("no SPIQA papers selected")

    captions = CaptionStore(images)
    author_captions_path = images / "author_captions.json"
    author_captions: dict[str, str] = {}
    if author_captions_path.is_file():
        with author_captions_path.open(encoding="utf-8") as stream:
            author_captions = json.load(stream)

    gold_qa: list[dict] = []
    without_text = n_images = missing_images = 0
    for number, paper_id in enumerate(paper_ids, 1):
        entry = meta[paper_id]
        text = load_paragraphs(paper_id, paragraph_files)
        if not text:
            without_text += 1
            print(f"[warn] no paragraphs for {paper_id}; paper skipped")
            continue
        (corpus / f"{paper_id}.txt").write_text(text, encoding="utf-8")
        for name, figure_meta in entry.get("all_figures", {}).items():
            source = image_files.get(name)
            if source is None:
                missing_images += 1
                continue
            destination = images / name
            if not destination.exists():
                shutil.copyfile(source, destination)
            if isinstance(figure_meta, dict) and figure_meta.get("caption"):
                author_captions[name] = figure_meta["caption"]
            captions.ensure(client, name, destination)
            n_images += 1
        for qa in entry.get("qa", []):
            gold_qa.append({**qa, "paper_id": paper_id})
        captions.save()
        with author_captions_path.open("w", encoding="utf-8") as stream:
            json.dump(author_captions, stream, indent=2, ensure_ascii=False)
        print(
            f"  [{number}/{len(paper_ids)}] {paper_id}: {len(entry.get('all_figures', {}))} figures"
        )

    gold_path = Path(gold_qa_path)
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    with gold_path.open("w", encoding="utf-8") as stream:
        json.dump(gold_qa, stream, indent=2, ensure_ascii=False)
    return IngestReport(
        papers=len(paper_ids) - without_text,
        papers_without_text=without_text,
        images=n_images,
        missing_images=missing_images,
        captions=len(captions),
        gold_qa=len(gold_qa),
    )
