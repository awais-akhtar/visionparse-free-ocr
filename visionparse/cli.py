"""Command-line interface for VisionParse."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .extraction.prices import extract_prices
from .extraction.structured_text import extract_menu_items
from .ocr.engine import get_ocr_engine
from .pipelines.document_pipeline import DocumentPipeline


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 0
    return int(args.handler(args) or 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visionparse",
        description="OCR, YOLO detection, price extraction, and document parsing.",
    )
    subparsers = parser.add_subparsers(dest="command")

    ocr = subparsers.add_parser("ocr", help="Read text from an image.")
    ocr.add_argument("image", help="Path to an image.")
    ocr.add_argument("--engine", default="tesseract", help="tesseract, easyocr, keras, google")
    ocr.add_argument("--lang", default="eng", help="OCR language(s), for example eng or eng+ara.")
    ocr.add_argument("--tesseract-cmd", default=None, help="Optional path to the Tesseract binary.")
    ocr.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    ocr.set_defaults(handler=cmd_ocr)

    prices = subparsers.add_parser("prices", help="Extract prices from text.")
    prices.add_argument("text", nargs="*", help="Text to parse. Reads stdin if omitted.")
    prices.add_argument("--file", type=Path, help="Read text from a file.")
    prices.add_argument("--default-currency", default=None)
    prices.add_argument("--allow-plain-numbers", action="store_true")
    prices.add_argument("--pretty", action="store_true")
    prices.set_defaults(handler=cmd_prices)

    parse = subparsers.add_parser("parse", help="Run the document pipeline on an image.")
    parse.add_argument("image", help="Path to an image.")
    parse.add_argument("--engine", default="tesseract", help="tesseract, easyocr, keras, google")
    parse.add_argument("--lang", default="eng")
    parse.add_argument("--tesseract-cmd", default=None)
    parse.add_argument("--yolo-model", default=None, help="Optional YOLO model path.")
    parse.add_argument("--yolo-confidence", type=float, default=0.25)
    parse.add_argument("--default-currency", default=None)
    parse.add_argument("--allow-plain-numbers", action="store_true")
    parse.add_argument("--pretty", action="store_true")
    parse.set_defaults(handler=cmd_parse)

    detect = subparsers.add_parser("detect", help="Run YOLO detection on an image.")
    detect.add_argument("image", help="Path to an image.")
    detect.add_argument("--model", required=True, help="YOLO model path or model name.")
    detect.add_argument("--confidence", type=float, default=0.25)
    detect.add_argument("--output", type=Path, help="Optional annotated image path.")
    detect.add_argument("--pretty", action="store_true")
    detect.set_defaults(handler=cmd_detect)

    return parser


def cmd_ocr(args: argparse.Namespace) -> int:
    engine_kwargs: dict[str, Any] = {}
    if args.engine.lower() in {"tesseract", "pytesseract"}:
        engine_kwargs = {"languages": args.lang, "tesseract_cmd": args.tesseract_cmd}
    engine = get_ocr_engine(args.engine, **engine_kwargs)
    result = engine.read(args.image)
    _write_json(result.to_dict(), pretty=args.pretty)
    return 0


def cmd_prices(args: argparse.Namespace) -> int:
    text = _read_text_argument(args)
    prices = extract_prices(
        text,
        default_currency=args.default_currency,
        allow_plain_numbers=args.allow_plain_numbers,
    )
    _write_json([price.to_dict() for price in prices], pretty=args.pretty)
    return 0


def cmd_parse(args: argparse.Namespace) -> int:
    ocr_options: dict[str, Any] = {}
    if args.engine.lower() in {"tesseract", "pytesseract"}:
        ocr_options = {"languages": args.lang, "tesseract_cmd": args.tesseract_cmd}

    pipeline = DocumentPipeline(
        ocr_engine=args.engine,
        default_currency=args.default_currency,
        allow_plain_number_prices=args.allow_plain_numbers,
        ocr_options=ocr_options,
    )
    result = pipeline.run(
        args.image,
        yolo_model=args.yolo_model,
        yolo_confidence=args.yolo_confidence,
    )
    _write_json(result.to_dict(), pretty=args.pretty)
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    from .detection.yolo import YoloDetector, draw_detections

    detector = YoloDetector(args.model, confidence=args.confidence)
    detections = detector.detect(args.image)
    payload: dict[str, Any] = {"detections": [detection.to_dict() for detection in detections]}
    if args.output:
        output = draw_detections(args.image, detections, args.output)
        payload["output"] = str(output)
    _write_json(payload, pretty=args.pretty)
    return 0


def _read_text_argument(args: argparse.Namespace) -> str:
    if args.file:
        return args.file.read_text(encoding="utf-8")
    if args.text:
        return " ".join(args.text)
    return sys.stdin.read()


def _write_json(payload: Any, *, pretty: bool = False) -> None:
    indent = 2 if pretty else None
    print(json.dumps(payload, indent=indent, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())

