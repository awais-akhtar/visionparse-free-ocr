"""Crop detected regions with YOLO, then OCR each region.

Usage:
    python examples/yolo_then_ocr.py path/to/menu.jpg --model models/best.pt
"""

from __future__ import annotations

import argparse

from visionparse.detection.yolo import YoloDetector
from visionparse.pipelines.document_pipeline import DocumentPipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--model", required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--lang", default="eng")
    args = parser.parse_args()

    detector = YoloDetector(args.model, confidence=args.confidence)
    pipeline = DocumentPipeline(
        ocr_engine="tesseract",
        detector=detector,
        ocr_options={"languages": args.lang},
    )
    result = pipeline.run(args.image)

    print(result.layout_text or result.text)
    print()
    print("Prices:")
    for price in result.prices:
        print(f"- {price.raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

