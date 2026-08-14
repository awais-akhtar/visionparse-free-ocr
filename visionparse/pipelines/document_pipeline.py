"""End-to-end document parsing pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from visionparse.extraction.prices import PriceMention, extract_prices
from visionparse.extraction.structured_text import MenuItem, extract_menu_items
from visionparse.ocr.engine import BaseOCREngine, OCRResult, get_ocr_engine
from visionparse.ocr.preprocessing import Box, crop_regions, load_image


@dataclass(frozen=True)
class RegionResult:
    """OCR result for one region of an image."""

    box: Optional[Box]
    text: str
    layout_text: Optional[str] = None
    prices: tuple[PriceMention, ...] = ()
    ocr: Optional[OCRResult] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "box": list(self.box) if self.box else None,
            "text": self.text,
            "layout_text": self.layout_text,
            "prices": [price.to_dict() for price in self.prices],
            "ocr": self.ocr.to_dict() if self.ocr else None,
        }


@dataclass(frozen=True)
class PipelineResult:
    """Full output from a document parse."""

    image: str
    text: str
    layout_text: Optional[str] = None
    prices: tuple[PriceMention, ...] = ()
    items: tuple[MenuItem, ...] = ()
    regions: tuple[RegionResult, ...] = ()
    detections: tuple[Any, ...] = ()
    engine: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "text": self.text,
            "layout_text": self.layout_text,
            "prices": [price.to_dict() for price in self.prices],
            "items": [item.to_dict() for item in self.items],
            "regions": [region.to_dict() for region in self.regions],
            "detections": [
                detection.to_dict() if hasattr(detection, "to_dict") else detection
                for detection in self.detections
            ],
            "engine": self.engine,
            "metadata": dict(self.metadata),
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)


class DocumentPipeline:
    """Run detection, OCR, price extraction, and light structuring."""

    def __init__(
        self,
        *,
        ocr_engine: str | BaseOCREngine = "tesseract",
        detector: Optional[Any] = None,
        crop_padding: int = 0,
        default_currency: Optional[str] = None,
        allow_plain_number_prices: bool = False,
        ocr_options: Optional[dict[str, Any]] = None,
    ) -> None:
        self.ocr_engine = (
            ocr_engine
            if isinstance(ocr_engine, BaseOCREngine)
            else get_ocr_engine(ocr_engine, **(ocr_options or {}))
        )
        self.detector = detector
        self.crop_padding = crop_padding
        self.default_currency = default_currency
        self.allow_plain_number_prices = allow_plain_number_prices

    def run(
        self,
        image: str | Path,
        *,
        boxes: Optional[Sequence[Box]] = None,
        yolo_model: Optional[str | Path] = None,
        yolo_confidence: float = 0.25,
    ) -> PipelineResult:
        """Parse one image document."""

        image_path = Path(image)
        detections: tuple[Any, ...] = ()
        prepared_image: Any = None

        if boxes is None:
            detector = self.detector
            if detector is None and yolo_model is not None:
                from visionparse.detection.yolo import YoloDetector

                detector = YoloDetector(yolo_model, confidence=yolo_confidence)
            if detector is not None:
                detections = tuple(detector.detect(image_path))
                boxes = [detection.box for detection in detections]

        regions: list[RegionResult] = []
        if boxes:
            prepared_image = load_image(image_path)
            crops = crop_regions(prepared_image, boxes, padding=self.crop_padding)
            for box, crop in zip(boxes, crops):
                ocr = self.ocr_engine.read(crop)
                prices = tuple(
                    extract_prices(
                        ocr.text,
                        default_currency=self.default_currency,
                        allow_plain_numbers=self.allow_plain_number_prices,
                    )
                )
                regions.append(
                    RegionResult(
                        box=box,
                        text=ocr.text,
                        layout_text=ocr.layout_text,
                        prices=prices,
                        ocr=ocr,
                    )
                )
        else:
            ocr = self.ocr_engine.read(image_path)
            prices = tuple(
                extract_prices(
                    ocr.text,
                    default_currency=self.default_currency,
                    allow_plain_numbers=self.allow_plain_number_prices,
                )
            )
            regions.append(
                RegionResult(
                    box=None,
                    text=ocr.text,
                    layout_text=ocr.layout_text,
                    prices=prices,
                    ocr=ocr,
                )
            )

        combined_text = "\n\n".join(region.text for region in regions if region.text).strip()
        combined_layout_text = "\n\n".join(
            region.layout_text or region.text for region in regions if region.layout_text or region.text
        ).strip()
        prices = tuple(
            extract_prices(
                combined_text,
                default_currency=self.default_currency,
                allow_plain_numbers=self.allow_plain_number_prices,
            )
        )
        items = tuple(
            extract_menu_items(
                combined_text,
                default_currency=self.default_currency,
                allow_plain_number_prices=self.allow_plain_number_prices,
            )
        )

        return PipelineResult(
            image=str(image_path),
            text=combined_text,
            layout_text=combined_layout_text,
            prices=prices,
            items=items,
            regions=tuple(regions),
            detections=detections,
            engine=getattr(self.ocr_engine, "name", None),
            metadata={"region_count": len(regions)},
        )


def parse_document(image: str | Path, **kwargs: Any) -> PipelineResult:
    """Convenience wrapper around :class:`DocumentPipeline`."""

    pipeline_options = {
        key: kwargs.pop(key)
        for key in list(kwargs)
        if key
        in {
            "ocr_engine",
            "detector",
            "crop_padding",
            "default_currency",
            "allow_plain_number_prices",
            "ocr_options",
        }
    }
    return DocumentPipeline(**pipeline_options).run(image, **kwargs)
