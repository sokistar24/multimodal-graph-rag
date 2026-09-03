"""Captioning and OCR helpers shared by every ingester.

Captions are produced by the same vision model for every corpus so the caption
channel has identical semantics across PubLayNet, SPIQA, DocBank, and
DocLayNet. In smoke mode nothing is captioned and nothing is written in its
place; an uncaptioned crop is rejected later by the image index rather than
carried through as a placeholder string.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytesseract
from PIL import Image

from ..clients import CAPTION_MODEL, ModelClient

CAPTION_PROMPT = (
    "Describe this region from a document in one concise sentence. It may be a "
    "figure, chart, table, or diagram. Be specific about the content, chart "
    "type, axes, or table values where visible."
)
TESSERACT_ENV = "TESSERACT_CMD"


def configure_tesseract() -> None:
    """Honour ``TESSERACT_CMD`` when the binary is not on ``PATH``."""
    command = os.environ.get(TESSERACT_ENV, "").strip()
    if command:
        pytesseract.pytesseract.tesseract_cmd = command


def ocr_text(crop: Image.Image) -> str:
    return pytesseract.image_to_string(crop).strip()


def caption_image(
    client: ModelClient, path: str | Path, *, model: str = CAPTION_MODEL
) -> str:
    result = client.call(model, user=CAPTION_PROMPT, images=[path], max_tokens=200)
    return result.text


class CaptionStore:
    """``captions.json`` beside the crops, written after every change."""

    def __init__(self, image_dir: str | Path) -> None:
        self.path = Path(image_dir) / "captions.json"
        self.captions: dict[str, str] = {}
        if self.path.is_file():
            with self.path.open(encoding="utf-8") as stream:
                self.captions = {str(k): str(v) for k, v in json.load(stream).items()}

    def __contains__(self, name: str) -> bool:
        return name in self.captions

    def __len__(self) -> int:
        return len(self.captions)

    def ensure(self, client: ModelClient | None, name: str, path: Path) -> bool:
        """Caption ``name`` unless cached; returns whether a call was made."""
        if name in self.captions or client is None:
            return False
        self.captions[name] = caption_image(client, path)
        return True

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as stream:
            json.dump(self.captions, stream, indent=2, ensure_ascii=False)
