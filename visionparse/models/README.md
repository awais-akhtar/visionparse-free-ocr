# Model assets

This folder contains small model reference files:

- `yolov3.cfg`
- `coco.names`
- `mscoco_label_map.pbtxt`
- `ssd_mobilenet_v2_coco.pbtxt`

Large binary weights are not committed:

- `.pt`
- `.pth`
- `.weights`
- `.onnx`
- `.pb`
- `.tflite`

The legacy scripts reference fine-tuned weights named like `best (1).pt` and `best (2).pt`, and TensorFlow examples reference `frozen_inference_graph.pb`. Those files are not present in this workspace.

Example with Darknet YOLO:

```bash
visionparse detect menu.jpg --backend darknet --model models/yolov3.weights
```

Example with Ultralytics/fine-tuned `.pt` weights:

```bash
visionparse detect menu.jpg --backend ultralytics --model models/best.pt
```
