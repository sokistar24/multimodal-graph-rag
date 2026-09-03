"""CLIP image retrieval: a text question retrieves figure or table crops.

Images and queries are embedded into CLIP's shared space so that visual
similarity, not caption text, decides which crop is retrieved. The image index
is cached per image directory, keyed by a content fingerprint of the crops.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import open_clip
import torch
from PIL import Image

from ..errors import CacheError, ConfigurationError, MissingEvidenceError

CACHE_FORMAT_VERSION = 1
CLIP_ARCHITECTURE = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class ImageHit:
    name: str
    caption: str
    score: float
    rank: int


@dataclass(frozen=True)
class ImageIndexMetadata:
    image_dir: str
    fingerprint: str
    retriever: str
    image_count: int
    dimension: int
    format_version: int = CACHE_FORMAT_VERSION


def image_files(image_dir: str | Path) -> list[Path]:
    directory = Path(image_dir)
    if not directory.is_dir():
        raise ConfigurationError(f"image directory does not exist: {directory}")
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise ConfigurationError(f"image directory holds no images: {directory}")
    return files


def image_fingerprint(image_dir: str | Path) -> str:
    digest = hashlib.sha256()
    for path in image_files(image_dir):
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}".encode())
    return digest.hexdigest()[:16]


def load_captions(image_dir: str | Path) -> dict[str, str]:
    path = Path(image_dir) / "captions.json"
    if not path.is_file():
        raise MissingEvidenceError(f"captions file does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        captions = json.load(stream)
    if not isinstance(captions, dict):
        raise ConfigurationError(f"captions file has an unexpected layout: {path}")
    return {str(k): str(v) for k, v in captions.items()}


class ClipEncoder:
    """Lazily loaded CLIP model shared by image and text encoding."""

    def __init__(
        self, architecture: str = CLIP_ARCHITECTURE, pretrained: str = CLIP_PRETRAINED
    ) -> None:
        self.architecture = architecture
        self.pretrained = pretrained
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    @property
    def retriever_name(self) -> str:
        return f"clip-{self.architecture.lower()}-{self.pretrained}"

    def _load(self) -> None:
        if self._model is None:
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.architecture, pretrained=self.pretrained
            )
            model.eval()
            self._model = model
            self._preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer(self.architecture)

    def encode_image(self, path: str | Path) -> np.ndarray:
        self._load()
        image = self._preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            vector = self._model.encode_image(image)
        return vector[0].cpu().numpy().astype("float32")

    def encode_text(self, text: str) -> np.ndarray:
        self._load()
        tokens = self._tokenizer([text])
        with torch.no_grad():
            vector = self._model.encode_text(tokens)
        return vector[0].cpu().numpy().astype("float32")


class ClipImageIndex:
    """Inner-product index over normalised CLIP image embeddings."""

    def __init__(
        self,
        vectors: np.ndarray,
        names: Sequence[str],
        captions: dict[str, str],
        metadata: ImageIndexMetadata,
        encoder: ClipEncoder,
    ) -> None:
        if vectors.shape[0] != len(names):
            raise ConfigurationError("image vectors and names must align")
        missing = [name for name in names if name not in captions]
        if missing:
            raise MissingEvidenceError(
                f"{len(missing)} indexed images have no caption, e.g. {missing[:3]}"
            )
        self.names = list(names)
        self.captions = captions
        self.metadata = metadata
        self.encoder = encoder
        self.vectors = np.ascontiguousarray(vectors, dtype="float32")
        faiss.normalize_L2(self.vectors)
        self.index = faiss.IndexFlatIP(self.vectors.shape[1])
        self.index.add(self.vectors)

    @property
    def size(self) -> int:
        return len(self.names)

    def retrieve(self, query: str, k: int) -> list[ImageHit]:
        if k < 1:
            raise ConfigurationError("k must be at least 1")
        query_vector = self.encoder.encode_text(query).reshape(1, -1)
        faiss.normalize_L2(query_vector)
        scores, indices = self.index.search(query_vector, min(k, self.size))
        hits = []
        for rank, (score, position) in enumerate(
            zip(scores[0], indices[0], strict=True), 1
        ):
            if position < 0:
                continue
            name = self.names[position]
            hits.append(
                ImageHit(
                    name=name,
                    caption=self.captions[name],
                    score=float(score),
                    rank=rank,
                )
            )
        return hits

    @staticmethod
    def cache_paths(
        cache_dir: str | Path, image_dir: str | Path, fingerprint: str
    ) -> tuple[Path, Path]:
        stem = f"image_index_{Path(image_dir).name}_{fingerprint}"
        base = Path(cache_dir) / "image_index"
        return base / f"{stem}.npz", base / f"{stem}.json"

    def save(self, vectors_path: Path, metadata_path: Path) -> None:
        vectors_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            vectors_path, vectors=self.vectors, names=np.asarray(self.names).astype(str)
        )
        with metadata_path.open("w", encoding="utf-8") as stream:
            json.dump(self.metadata.__dict__, stream, indent=2)

    @classmethod
    def build(
        cls,
        image_dir: str | Path,
        *,
        cache_dir: str | Path,
        encoder: ClipEncoder | None = None,
        rebuild: bool = False,
    ) -> ClipImageIndex:
        """Load the cached image index for this directory, or embed and cache it."""
        encoder = encoder or ClipEncoder()
        captions = load_captions(image_dir)
        fingerprint = image_fingerprint(image_dir)
        vectors_path, metadata_path = cls.cache_paths(cache_dir, image_dir, fingerprint)
        if not rebuild and vectors_path.exists():
            try:
                with metadata_path.open(encoding="utf-8") as stream:
                    metadata = ImageIndexMetadata(**json.load(stream))
                with np.load(vectors_path, allow_pickle=False) as payload:
                    vectors = payload["vectors"]
                    names = payload["names"].tolist()
            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
                json.JSONDecodeError,
            ) as exc:
                raise CacheError(
                    f"image index cache is unreadable: {vectors_path}"
                ) from exc
            if (
                metadata.format_version != CACHE_FORMAT_VERSION
                or metadata.retriever != encoder.retriever_name
            ):
                raise CacheError(
                    f"image index cache {vectors_path} was built with a different "
                    "retriever or format; pass rebuild=True"
                )
            if len(names) != metadata.image_count:
                raise CacheError(
                    f"image index cache disagrees with its metadata: {vectors_path}"
                )
            return cls(vectors, names, captions, metadata, encoder)

        paths = image_files(image_dir)
        vectors = np.stack([encoder.encode_image(path) for path in paths])
        metadata = ImageIndexMetadata(
            image_dir=str(image_dir),
            fingerprint=fingerprint,
            retriever=encoder.retriever_name,
            image_count=len(paths),
            dimension=int(vectors.shape[1]),
        )
        index = cls(vectors, [path.name for path in paths], captions, metadata, encoder)
        index.save(vectors_path, metadata_path)
        return index
