"""OCR engines with optional dependencies loaded lazily."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Optional, Sequence

from .preprocessing import enhance_for_ocr, image_to_bytes, load_image


Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class OCRResult:
    """Text and metadata returned by an OCR engine."""

    text: str
    engine: str
    confidence: Optional[float] = None
    boxes: tuple[Box, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "engine": self.engine,
            "confidence": self.confidence,
            "boxes": [list(box) for box in self.boxes],
            "metadata": dict(self.metadata),
        }


class BaseOCREngine:
    """Base class for OCR engines."""

    name = "base"

    def read(self, image: str | Path | Any) -> OCRResult:  # pragma: no cover - interface
        raise NotImplementedError


class TesseractOCR(BaseOCREngine):
    """OCR through pytesseract.

    No executable path is hardcoded. If needed, pass ``tesseract_cmd`` or set
    the ``TESSERACT_CMD`` environment variable.
    """

    name = "tesseract"

    def __init__(
        self,
        languages: str = "eng",
        config: str = "--oem 3 --psm 6",
        tesseract_cmd: Optional[str] = None,
        preprocess: bool = True,
        preprocessing_options: Optional[dict[str, Any]] = None,
    ) -> None:
        self.languages = languages
        self.config = config
        self.tesseract_cmd = tesseract_cmd or os.getenv("TESSERACT_CMD")
        self.preprocess = preprocess
        self.preprocessing_options = preprocessing_options or {}

    def read(self, image: str | Path | Any) -> OCRResult:
        try:
            import pytesseract
            from pytesseract import Output
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install Tesseract support with `pip install visionparse[ocr]`.") from exc

        if self.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

        prepared = (
            enhance_for_ocr(image, **self.preprocessing_options)
            if self.preprocess
            else load_image(image)
        )
        text = pytesseract.image_to_string(prepared, lang=self.languages, config=self.config)

        boxes: list[Box] = []
        confidences: list[float] = []
        try:
            data = pytesseract.image_to_data(
                prepared,
                lang=self.languages,
                config=self.config,
                output_type=Output.DICT,
            )
            for i, word in enumerate(data.get("text", [])):
                if not str(word).strip():
                    continue
                confidence = _safe_float(data.get("conf", ["-1"])[i])
                if confidence >= 0:
                    confidences.append(confidence)
                left = int(data["left"][i])
                top = int(data["top"][i])
                width = int(data["width"][i])
                height = int(data["height"][i])
                boxes.append((left, top, left + width, top + height))
        except Exception:
            # OCR text is more important than box metadata. Keep the read stable.
            pass

        return OCRResult(
            text=text.strip(),
            engine=self.name,
            confidence=mean(confidences) if confidences else None,
            boxes=tuple(boxes),
            metadata={"languages": self.languages, "config": self.config},
        )


class EasyOCR(BaseOCREngine):
    """OCR through EasyOCR."""

    name = "easyocr"

    def __init__(self, languages: Sequence[str] = ("en",), gpu: bool = False, **reader_kwargs: Any) -> None:
        self.languages = tuple(languages)
        self.gpu = gpu
        self.reader_kwargs = reader_kwargs
        self._reader: Any = None

    def _get_reader(self) -> Any:
        if self._reader is None:
            try:
                import easyocr
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError("Install EasyOCR support with `pip install visionparse[easyocr]`.") from exc
            self._reader = easyocr.Reader(list(self.languages), gpu=self.gpu, **self.reader_kwargs)
        return self._reader

    def read(self, image: str | Path | Any) -> OCRResult:
        reader = self._get_reader()
        source: Any
        if isinstance(image, (str, Path)):
            source = str(image)
        else:
            try:
                import numpy as np
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError("EasyOCR image objects require NumPy.") from exc
            source = np.array(load_image(image))

        detections = reader.readtext(source, detail=1)
        words: list[str] = []
        boxes: list[Box] = []
        confidences: list[float] = []

        for bbox, text, confidence in detections:
            words.append(str(text))
            confidences.append(float(confidence))
            xs = [int(point[0]) for point in bbox]
            ys = [int(point[1]) for point in bbox]
            boxes.append((min(xs), min(ys), max(xs), max(ys)))

        return OCRResult(
            text=" ".join(words).strip(),
            engine=self.name,
            confidence=mean(confidences) if confidences else None,
            boxes=tuple(boxes),
            metadata={"languages": self.languages},
        )


class KerasOCR(BaseOCREngine):
    """OCR through keras-ocr."""

    name = "keras_ocr"

    def __init__(self, **pipeline_kwargs: Any) -> None:
        self.pipeline_kwargs = pipeline_kwargs
        self._pipeline: Any = None

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            try:
                import keras_ocr
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError("Install Keras OCR support with `pip install visionparse[keras]`.") from exc
            self._pipeline = keras_ocr.pipeline.Pipeline(**self.pipeline_kwargs)
        return self._pipeline

    def read(self, image: str | Path | Any) -> OCRResult:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Keras OCR support requires NumPy.") from exc

        pipeline = self._get_pipeline()
        prepared = np.array(load_image(image))
        predictions = pipeline.recognize([prepared])[0]

        words: list[str] = []
        boxes: list[Box] = []
        for word, bbox in predictions:
            words.append(str(word))
            xs = [int(point[0]) for point in bbox]
            ys = [int(point[1]) for point in bbox]
            boxes.append((min(xs), min(ys), max(xs), max(ys)))

        return OCRResult(text=" ".join(words).strip(), engine=self.name, boxes=tuple(boxes))


class GoogleVisionOCR(BaseOCREngine):
    """OCR through Google Cloud Vision.

    Credentials are provided by Application Default Credentials or by passing a
    local service-account JSON path at runtime. The path is never hardcoded.
    """

    name = "google_vision"

    def __init__(self, credentials_path: Optional[str | Path] = None) -> None:
        self.credentials_path = Path(credentials_path) if credentials_path else None
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            from google.cloud import vision
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install Google Vision support with `pip install visionparse[google]`.") from exc

        if self.credentials_path:
            try:
                from google.oauth2 import service_account
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError("Google service-account support is missing.") from exc
            credentials = service_account.Credentials.from_service_account_file(
                str(self.credentials_path)
            )
            self._client = vision.ImageAnnotatorClient(credentials=credentials)
        else:
            self._client = vision.ImageAnnotatorClient()
        return self._client

    def read(self, image: str | Path | Any) -> OCRResult:
        try:
            from google.cloud import vision
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install Google Vision support with `pip install visionparse[google]`.") from exc

        client = self._get_client()
        prepared = load_image(image)
        response = client.text_detection(image=vision.Image(content=image_to_bytes(prepared, "PNG")))
        if getattr(response, "error", None) and response.error.message:
            raise RuntimeError(response.error.message)

        annotations = list(response.text_annotations or [])
        text = annotations[0].description.strip() if annotations else ""
        boxes: list[Box] = []
        for annotation in annotations[1:]:
            vertices = annotation.bounding_poly.vertices
            xs = [vertex.x for vertex in vertices]
            ys = [vertex.y for vertex in vertices]
            boxes.append((min(xs), min(ys), max(xs), max(ys)))

        return OCRResult(text=text, engine=self.name, boxes=tuple(boxes))


def get_ocr_engine(engine: str | BaseOCREngine = "tesseract", **kwargs: Any) -> BaseOCREngine:
    """Create an OCR engine by name."""

    if isinstance(engine, BaseOCREngine):
        return engine

    key = engine.lower().replace("-", "_")
    if key in {"tesseract", "pytesseract"}:
        return TesseractOCR(**kwargs)
    if key == "easyocr":
        return EasyOCR(**kwargs)
    if key in {"keras", "keras_ocr"}:
        return KerasOCR(**kwargs)
    if key in {"google", "google_vision", "vision"}:
        return GoogleVisionOCR(**kwargs)
    raise ValueError(f"Unknown OCR engine: {engine}")


def read_text(image: str | Path | Any, engine: str | BaseOCREngine = "tesseract", **kwargs: Any) -> str:
    """Read text from an image and return only the text."""

    return get_ocr_engine(engine, **kwargs).read(image).text


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0

