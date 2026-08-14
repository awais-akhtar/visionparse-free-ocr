# VisionParse

VisionParse is a small, practical toolkit for turning messy image-based documents into useful text and data. It wraps the pieces that usually end up scattered across notebooks: OCR, image preprocessing, YOLO/object detection, price extraction, and optional LLM cleanup.

It started life as a set of computer-vision experiments. This package gives those ideas a proper home: import-safe modules, a CLI, tests, PyPI metadata, and GitHub Actions publishing.

The heart of the project is still research-minded: use free/local OCR first, keep text localized with bounding boxes, preserve the page/menu layout as much as possible, and only bring in heavier YOLO or LLM tools when they genuinely help.

```bash
pip install visionparse-free-ocr
```

The PyPI distribution is named `visionparse-free-ocr`. The Python import stays short:

```python
import visionparse
```

## What it does

- Runs OCR with Tesseract, EasyOCR, Keras OCR, or Google Vision.
- Preprocesses images before OCR: resize, grayscale, denoise, threshold, contrast, crop.
- Runs YOLO detections and returns clean bounding boxes.
- Groups localized OCR words into lines and blocks so aligned text stays aligned.
- Extracts prices from noisy OCR text.
- Turns menu-like OCR into lightweight structured items.
- Optionally asks an LLM/LangChain flow to clean up the structure.
- Provides one document pipeline and one CLI so the pieces fit together.

## Installation

The base install is intentionally light:

```bash
pip install visionparse-free-ocr
```

For OCR with Tesseract:

```bash
pip install "visionparse-free-ocr[ocr]"
```

You still need the Tesseract system binary installed. On Windows, install Tesseract and either add it to `PATH` or pass the path when you create the engine.

For YOLO detection:

```bash
pip install "visionparse-free-ocr[yolo]"
```

For the full kitchen sink:

```bash
pip install "visionparse-free-ocr[all]"
```

Extras are split this way because object detection, EasyOCR, Keras OCR, and Google Vision pull in heavier dependencies. Most projects do not need all of them at once.

## Quick start

### Extract prices from text

```python
from visionparse import extract_prices

text = "Chicken Biryani £8.99\nFamily Platter 24.50\nMango Lassi Rs. 450"

for price in extract_prices(text):
    print(price.raw, price.amount, price.currency)
```

### Parse menu-like OCR text

```python
from visionparse import extract_menu_items

ocr_text = """
Starters
Samosa £3.50
Chicken Pakora £5.99

Mains
Lamb Karahi £12.95
"""

items = extract_menu_items(ocr_text)

for item in items:
    print(item.name, item.prices, item.category)
```

### OCR an image with Tesseract

```python
from visionparse.ocr.engine import TesseractOCR

ocr = TesseractOCR(
    languages="eng",
    config="--oem 3 --psm 6",
    # tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)

result = ocr.read("menu.jpg")
print(result.text)
```

### Run the document pipeline

```python
from visionparse.pipelines.document_pipeline import DocumentPipeline

pipeline = DocumentPipeline(ocr_engine="tesseract")
result = pipeline.run("menu.jpg")

print(result.text)
print(result.layout_text)  # layout-preserving text when OCR boxes are available
print([price.raw for price in result.prices])
print([item.to_dict() for item in result.items])
```

### Preserve layout from localized OCR

```python
from visionparse.ocr.localization import TextToken, group_tokens_into_lines, render_aligned_text

tokens = [
    TextToken("Burger", (10, 10, 70, 25)),
    TextToken("£7.99", (180, 10, 230, 25)),
    TextToken("Fries", (10, 45, 55, 60)),
    TextToken("£2.50", (180, 45, 230, 60)),
]

lines = group_tokens_into_lines(tokens)
print(render_aligned_text(lines, char_width=10))
```

### Use YOLO regions before OCR

```python
from visionparse.detection.yolo import YoloDetector
from visionparse.pipelines.document_pipeline import DocumentPipeline

detector = YoloDetector("models/menu-sections.pt", confidence=0.25)
pipeline = DocumentPipeline(ocr_engine="tesseract", detector=detector)

result = pipeline.run("menu.jpg")

for region in result.regions:
    print(region.box, region.text[:120])
```

Model weights are not bundled. Keep them outside the package or in `visionparse/models/` locally, but do not commit them.

The original research code referenced fine-tuned YOLO weights such as `best (1).pt` and `best (2).pt`. Those binary files were not present in this workspace when the public package was prepared. The repo does include the safe model reference assets that were present, including `yolov3.cfg`, `coco.names`, and COCO/TensorFlow config files. See `docs/model-card.md`.

Darknet/OpenCV YOLO is also supported:

```python
from visionparse.detection.yolo import OpenCVDarknetYoloDetector

detector = OpenCVDarknetYoloDetector(
    weights_path="models/yolov3.weights",
    # config_path and names_path default to the packaged yolov3.cfg/coco.names
)

detections = detector.detect("menu.jpg")
```

## Command line

After installation, the `visionparse` command is available.

OCR:

```bash
visionparse ocr menu.jpg --engine tesseract --lang eng --pretty
```

Extract prices from a string:

```bash
visionparse prices "Burger £7.99 Fries 2.50" --pretty
```

Extract prices from a file:

```bash
visionparse prices --file ocr-output.txt --pretty
```

Run the full parser:

```bash
visionparse parse menu.jpg --engine tesseract --pretty
```

Run the parser with YOLO regions:

```bash
visionparse parse menu.jpg --engine tesseract --yolo-model models/menu-sections.pt --pretty
```

Run YOLO only:

```bash
visionparse detect menu.jpg --model models/menu-sections.pt --pretty
```

Run Darknet YOLO with the packaged config/labels and your local weights:

```bash
visionparse detect menu.jpg --backend darknet --model models/yolov3.weights --pretty
```

Save an annotated detection image:

```bash
visionparse detect menu.jpg --model models/menu-sections.pt --output annotated.jpg
```

## OCR engines

### Tesseract

Good default when you want a local, lightweight OCR engine. Install the Python extra and the system binary:

```bash
pip install "visionparse-free-ocr[ocr]"
```

```python
from visionparse.ocr.engine import TesseractOCR

ocr = TesseractOCR(languages="eng+ara", config="--oem 3 --psm 6")
print(ocr.read("receipt.jpg").text)
```

If Tesseract is installed in a custom location:

```python
ocr = TesseractOCR(tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe")
```

You can also set:

```bash
set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### EasyOCR

```bash
pip install "visionparse-free-ocr[easyocr]"
```

```python
from visionparse.ocr.engine import EasyOCR

ocr = EasyOCR(languages=("en",))
result = ocr.read("shop-sign.jpg")
```

### Keras OCR

```bash
pip install "visionparse-free-ocr[keras]"
```

```python
from visionparse.ocr.engine import KerasOCR

ocr = KerasOCR()
result = ocr.read("menu.jpg")
```

### Google Vision

```bash
pip install "visionparse-free-ocr[google]"
```

Use Application Default Credentials, or pass a service-account file at runtime. Do not commit the JSON file.

```python
from visionparse.ocr.engine import GoogleVisionOCR

ocr = GoogleVisionOCR(credentials_path="local-only-service-account.json")
print(ocr.read("invoice.jpg").text)
```

## LLM/LangChain cleanup

The regular parser is deterministic and does not need an API key. If you want LLM cleanup, install the LLM extra and use an environment variable:

```bash
pip install "visionparse-free-ocr[llm]"
set OPENAI_API_KEY=your-key-here
```

```python
from visionparse.extraction.structured_text import structure_with_llm

cleaned = structure_with_llm(raw_ocr_text, model="gpt-4o-mini")
print(cleaned)
```

No OpenAI key is stored in the package. The code reads from `OPENAI_API_KEY` at runtime.

## Research notes, examples, and benchmarks

The repo includes:

- `docs/research.md` — project findings and outcomes from the OCR/layout experiments.
- `docs/model-card.md` — how the fine-tuned YOLO model should be handled.
- `examples/` — free OCR and YOLO+OCR usage scripts.
- `benchmarks/` — lightweight text/layout benchmarks plus an optional local image OCR runner.

The public package does not commit the old generated images, notebooks, OCR outputs, or service-account files. If you want to benchmark the legacy images locally, keep them in `.visionparse_private_legacy/` or another local folder:

```bash
python benchmarks/run_benchmarks.py --images .visionparse_private_legacy
```

## Package layout

```text
visionparse/
├── detection/
│   └── yolo.py
├── ocr/
│   ├── engine.py
│   └── preprocessing.py
├── extraction/
│   ├── prices.py
│   └── structured_text.py
├── pipelines/
│   └── document_pipeline.py
├── models/
├── cli.py
└── tests/
```

## Publishing to PyPI from GitHub Actions

This repo includes `.github/workflows/publish.yml`.

To publish:

1. Create a PyPI API token.
2. Add it to the GitHub repository secrets as `PYPI_API_TOKEN`.
3. Push a version tag:

```bash
git tag v0.1.1
git push origin v0.1.1
```

The workflow builds the source distribution and wheel, checks them with Twine, and publishes to PyPI using the secret.

You can also run the publish workflow manually from GitHub Actions.

## Security notes

This package should not contain:

- OpenAI keys
- AWS keys
- Google service-account JSON files
- YOLO weights
- generated OCR output files
- test images or notebook outputs

Use environment variables or local-only files instead:

```bash
set OPENAI_API_KEY=...
set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\service-account.json
```

The `.gitignore` is set up to keep the common mistakes out of the repo.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m build
twine check dist/*
```

The tests avoid heavyweight OCR/model dependencies. They check the parser, price extraction, and import safety first; model-specific tests can be added later with fixtures.

## License

MIT.
