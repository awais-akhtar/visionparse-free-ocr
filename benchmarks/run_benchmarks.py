"""Small benchmark runner for the public repo.

The default benchmark uses text fixtures so it can run anywhere. If you pass an
image folder and have Tesseract installed, it will also run a free-OCR smoke
benchmark over local images.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visionparse.extraction.prices import extract_prices
from visionparse.extraction.structured_text import extract_menu_items


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, help="Optional folder of local images.")
    args = parser.parse_args()

    run_text_fixture()

    if args.images:
        run_image_smoke(args.images)

    return 0


def run_text_fixture() -> None:
    text = (ROOT / "fixtures" / "menu_ocr_noisy.txt").read_text(encoding="utf-8")
    expected = json.loads((ROOT / "fixtures" / "menu_expected_items.json").read_text())

    start = perf_counter()
    prices = extract_prices(text, allow_plain_numbers=False)
    items = extract_menu_items(text)
    elapsed = perf_counter() - start

    expected_names = {row["name"] for row in expected}
    found_names = {item.name for item in items}

    print("Text fixture benchmark")
    print("----------------------")
    print(f"prices_found: {len(prices)}")
    print(f"items_found: {len(items)}")
    print(f"expected_items: {len(expected)}")
    print(f"matched_item_names: {len(expected_names & found_names)}")
    print(f"elapsed_seconds: {elapsed:.4f}")
    print()


def run_image_smoke(folder: Path) -> None:
    try:
        from visionparse.ocr.engine import TesseractOCR
    except ImportError as exc:
        print(f"Skipping image OCR benchmark: {exc}")
        return

    images = sorted(
        path
        for path in folder.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )
    if not images:
        print(f"No images found in {folder}")
        return

    ocr = TesseractOCR(languages="eng", config="--oem 3 --psm 6")
    print("Local image OCR smoke benchmark")
    print("-------------------------------")

    for image in images:
        start = perf_counter()
        try:
            result = ocr.read(image)
        except Exception as exc:
            print(f"{image.name}: skipped ({exc})")
            continue
        elapsed = perf_counter() - start
        print(
            f"{image.name}: chars={len(result.text)} prices="
            f"{len(extract_prices(result.text))} seconds={elapsed:.2f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
