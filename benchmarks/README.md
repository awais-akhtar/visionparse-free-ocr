# Benchmarks

These benchmarks are intentionally lightweight. They measure the pieces that can be tested without paid services:

- price extraction from noisy OCR text;
- menu item extraction;
- layout reconstruction from localized OCR tokens.

For image OCR benchmarks, point the runner at your own image folder:

```bash
python benchmarks/run_benchmarks.py --images .visionparse_private_legacy
```

If Tesseract is installed, the runner will OCR the images and report rough counts. If it is not installed, the text fixtures still run.

## Research benchmark idea

A practical benchmark set for this project should track:

- raw OCR text quality;
- price recall;
- item/price alignment accuracy;
- layout preservation after token grouping;
- effect of YOLO region cropping vs whole-image OCR.

The most important outcome is not just “more text extracted”; it is “item names and prices stay aligned enough to become structured data.”

