# Research notes and project outcomes

VisionParse came from a practical OCR research workflow: take menu/document images, find the important regions, OCR them with free or local tools, then recover the structure that normal OCR tends to flatten.

## Problem

Basic OCR is often enough to get words, but not enough to get usable data. Menu and receipt images add a few extra problems:

- prices sit in separate columns from item names;
- OCR engines can read visually separate regions in the wrong order;
- decorative images and backgrounds create noise;
- model APIs can become expensive when every experiment depends on paid OCR or an LLM;
- hardcoded local paths and API keys make experiments difficult to publish safely.

## Approach

The project explored a layered pipeline:

1. Preprocess the image with free/local tooling.
2. Localize text using bounding boxes instead of relying only on raw text.
3. Group words into visual lines and blocks.
4. Preserve the approximate layout in plain text.
5. Extract prices with deterministic rules.
6. Optionally use YOLO to crop menu/document regions before OCR.
7. Optionally use an LLM only after the cheap/local pass has done most of the work.

That keeps the default workflow free-first, with paid/cloud tools as optional upgrades.

## Findings

- Text localization matters more than aggressive image filters once the image is readable. Grouping word boxes into lines and blocks keeps item names and prices closer to their original visual relationship.
- Cropping document sections before OCR reduces background noise and helps the OCR engine focus on the useful region.
- Tesseract remains a good baseline for free OCR when paired with sensible preprocessing and layout grouping.
- EasyOCR and Keras OCR are useful alternatives, especially when Tesseract struggles with stylized fonts, but they add heavier dependencies.
- LLM cleanup is most reliable when it receives localized, layout-aware OCR text instead of one flattened word stream.
- Model weights, generated images, raw notebooks, and service-account files should stay outside the package. The package should load those assets when the user provides them.

## Current package outcome

The public package now focuses on the reusable parts of the research:

- `visionparse.ocr.preprocessing` for image preparation and crops;
- `visionparse.ocr.localization` for word boxes, lines, blocks, and aligned text;
- `visionparse.ocr.engine` for free/local OCR plus optional cloud engines;
- `visionparse.detection.yolo` for fine-tuned YOLO model loading;
- `visionparse.extraction.prices` for price parsing;
- `visionparse.extraction.structured_text` for menu-like structured output;
- `visionparse.pipelines.document_pipeline` for the full path.

## What is intentionally not published

The original workspace contained generated images, OCR output files, notebooks, and a Google service-account JSON. Those are useful during research but unsafe or noisy in a public package. They are kept out of git.

The legacy code also references fine-tuned YOLO weights named like `best (1).pt` and `best (2).pt`, but those files were not present in the workspace when the package was prepared. VisionParse supports loading those weights when you provide them locally.

