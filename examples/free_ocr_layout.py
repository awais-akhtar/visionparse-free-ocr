"""Free OCR with layout-aware text reconstruction.

Usage:
    python examples/free_ocr_layout.py path/to/menu.jpg
"""

from __future__ import annotations

import argparse

from visionparse.ocr.engine import TesseractOCR


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--lang", default="eng")
    parser.add_argument("--tesseract-cmd", default=None)
    args = parser.parse_args()

    ocr = TesseractOCR(languages=args.lang, tesseract_cmd=args.tesseract_cmd)
    result = ocr.read(args.image)

    print("Raw OCR")
    print("-------")
    print(result.text)
    print()
    print("Layout-aware OCR")
    print("----------------")
    print(result.layout_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

