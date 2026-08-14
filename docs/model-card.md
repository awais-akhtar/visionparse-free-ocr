# Fine-tuned YOLO model card

VisionParse is designed to work with the YOLO detector work from the original research project. The repository now ships the small safe reference files that were present in the workspace, including `yolov3.cfg` and `coco.names`.

## Why the weights are not in git

YOLO weight files are usually large binary artifacts. They can also contain licensing constraints depending on training data and base model. For that reason, the package ships the loader, config, and labels, while you provide the actual `.pt` or `.weights` file locally or through a release/artifact system.

At packaging time, this workspace included:

- `yolov3.cfg`
- `coco.names`
- `mscoco_label_map.pbtxt`
- `ssd_mobilenet_v2_coco.pbtxt`

The workspace did not include:

- `best (1).pt`
- `best (2).pt`
- `yolov3.weights`
- `frozen_inference_graph.pb`

## Expected model behavior

The legacy experiments used YOLO to find useful document/menu regions before OCR. A typical detector should return boxes for regions such as:

- menu sections;
- text panels;
- price/item blocks;
- document regions that should be cropped before OCR.

## How to use your trained model

```python
from visionparse.detection.yolo import YoloDetector
from visionparse.pipelines.document_pipeline import DocumentPipeline

detector = YoloDetector("models/best.pt", confidence=0.25)
pipeline = DocumentPipeline(ocr_engine="tesseract", detector=detector)

result = pipeline.run("examples/assets/menu.jpg")
print(result.layout_text)
```

Or from the CLI:

```bash
visionparse parse menu.jpg --engine tesseract --yolo-model models/best.pt --pretty
```

For the Darknet/OpenCV path:

```bash
visionparse detect menu.jpg \
  --backend darknet \
  --model models/yolov3.weights \
  --config visionparse/models/yolov3.cfg \
  --names visionparse/models/coco.names \
  --pretty
```

If you omit `--config` and `--names`, VisionParse uses the packaged `yolov3.cfg` and `coco.names`.

## Recommended release workflow

If the model can be shared publicly, publish it as a GitHub Release asset rather than committing it to the repo:

```text
VisionParse/
├── visionparse/
├── docs/
├── examples/
└── models/
    └── README.md
```

Then users can download the model into `models/best.pt`.

If the model is private, keep it in local storage and pass the path at runtime.
