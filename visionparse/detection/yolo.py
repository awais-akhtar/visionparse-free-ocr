"""YOLO object-detection helpers.

The module is safe to import without Ultralytics installed. The dependency is
loaded only when a detector is created.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Iterable, Optional


Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class Detection:
    """One object detection in xyxy format."""

    label: str
    confidence: float
    box: Box
    class_id: Optional[int] = None

    @property
    def xyxy(self) -> Box:
        return self.box

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "box": list(self.box),
            "class_id": self.class_id,
        }


class YoloDetector:
    """Small wrapper around Ultralytics YOLO.

    Parameters
    ----------
    model_path:
        Local model path or a model name understood by Ultralytics, for example
        ``"yolov8n.pt"`` or ``"models/menu-sections.pt"``.
    confidence:
        Minimum confidence threshold.
    provider:
        ``"auto"`` tries ``ultralytics`` first, then ``ultralyticsplus`` for
        older experiments.
    """

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.25,
        iou: Optional[float] = None,
        device: Optional[str] = None,
        image_size: Optional[int] = None,
        provider: str = "auto",
        **predict_kwargs: Any,
    ) -> None:
        self.model_path = str(model_path)
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self.image_size = image_size
        self.predict_kwargs = predict_kwargs

        yolo_cls = _load_yolo_class(provider)
        self.model = yolo_cls(self.model_path)
        self._apply_overrides()

    def _apply_overrides(self) -> None:
        overrides = getattr(self.model, "overrides", None)
        if isinstance(overrides, dict):
            overrides["conf"] = self.confidence
            if self.iou is not None:
                overrides["iou"] = self.iou
            if self.device is not None:
                overrides["device"] = self.device
            if self.image_size is not None:
                overrides["imgsz"] = self.image_size

    def detect(self, image: str | Path | Any, **kwargs: Any) -> list[Detection]:
        """Run YOLO and return simple dataclass detections."""

        predict_options = dict(self.predict_kwargs)
        predict_options.update(kwargs)
        predict_options.setdefault("conf", self.confidence)
        if self.iou is not None:
            predict_options.setdefault("iou", self.iou)
        if self.device is not None:
            predict_options.setdefault("device", self.device)
        if self.image_size is not None:
            predict_options.setdefault("imgsz", self.image_size)

        results = self.model.predict(str(image), **predict_options)
        if not results:
            return []

        result = results[0]
        boxes_obj = getattr(result, "boxes", None)
        if boxes_obj is None:
            return []

        boxes = _as_rows(getattr(boxes_obj, "xyxy", []))
        confidences = _as_flat_list(getattr(boxes_obj, "conf", []))
        classes = _as_flat_list(getattr(boxes_obj, "cls", []))
        names = getattr(result, "names", None) or getattr(self.model, "names", {})

        detections: list[Detection] = []
        for index, box in enumerate(boxes):
            class_id = _safe_int(classes[index]) if index < len(classes) else None
            label = _label_for(names, class_id)
            confidence = _safe_float(confidences[index]) if index < len(confidences) else 0.0
            xyxy = tuple(int(round(float(value))) for value in box[:4])
            detections.append(
                Detection(
                    label=label,
                    confidence=confidence,
                    box=xyxy,  # type: ignore[arg-type]
                    class_id=class_id,
                )
            )

        return detections


class OpenCVDarknetYoloDetector:
    """YOLO detector for Darknet config/weights through OpenCV DNN.

    This supports the older research assets in the project, such as
    ``yolov3.cfg`` and ``coco.names``. A Darknet config and class-name file can
    be shipped safely, but the binary ``.weights`` file must be supplied by the
    user at runtime.
    """

    def __init__(
        self,
        weights_path: str | Path,
        config_path: str | Path | None = None,
        names_path: str | Path | None = None,
        confidence: float = 0.5,
        nms_threshold: float = 0.4,
        input_size: tuple[int, int] = (416, 416),
        scale: float = 1 / 255.0,
        swap_rb: bool = True,
        crop: bool = False,
    ) -> None:
        self.weights_path = str(weights_path)
        self.config_path = str(config_path or packaged_model_path("yolov3.cfg"))
        self.names_path = str(names_path or packaged_model_path("coco.names"))
        self.confidence = confidence
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.scale = scale
        self.swap_rb = swap_rb
        self.crop = crop
        self.names = load_class_names(self.names_path)

        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install Darknet YOLO support with `pip install visionparse[darknet]`.") from exc

        if not Path(self.config_path).exists():
            raise FileNotFoundError(f"YOLO config not found: {self.config_path}")
        if not Path(self.weights_path).exists():
            raise FileNotFoundError(
                f"YOLO weights not found: {self.weights_path}. "
                "The package includes config/labels, but trained weights must be supplied locally."
            )

        self.cv2 = cv2
        self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)

    def detect(self, image: str | Path | Any) -> list[Detection]:
        """Run Darknet YOLO through OpenCV DNN."""

        cv2 = self.cv2
        if isinstance(image, (str, Path)):
            frame = cv2.imread(str(image))
            if frame is None:
                raise ValueError(f"Could not read image: {image}")
        else:
            frame = image

        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=self.scale,
            size=self.input_size,
            mean=(0, 0, 0),
            swapRB=self.swap_rb,
            crop=self.crop,
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self._output_layer_names())

        boxes: list[list[int]] = []
        confidences: list[float] = []
        class_ids: list[int] = []

        for output in outputs:
            for row in output:
                if len(row) <= 5:
                    continue
                objectness = float(row[4])
                scores = row[5:]
                class_id = int(scores.argmax())
                class_score = float(scores[class_id])
                confidence = objectness * class_score
                if confidence < self.confidence:
                    continue

                center_x = int(row[0] * width)
                center_y = int(row[1] * height)
                box_width = int(row[2] * width)
                box_height = int(row[3] * height)
                left = int(center_x - box_width / 2)
                top = int(center_y - box_height / 2)

                boxes.append([left, top, box_width, box_height])
                confidences.append(confidence)
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.nms_threshold)
        selected = _flatten_indices(indices)
        detections: list[Detection] = []

        for index in selected:
            left, top, box_width, box_height = boxes[index]
            x1 = max(0, left)
            y1 = max(0, top)
            x2 = min(width, left + box_width)
            y2 = min(height, top + box_height)
            class_id = class_ids[index]
            label = self.names[class_id] if 0 <= class_id < len(self.names) else str(class_id)
            detections.append(
                Detection(
                    label=label,
                    confidence=confidences[index],
                    box=(x1, y1, x2, y2),
                    class_id=class_id,
                )
            )

        return detections

    def _output_layer_names(self) -> list[str]:
        layer_names = self.net.getLayerNames()
        unconnected = self.net.getUnconnectedOutLayers()
        indices = _flatten_indices(unconnected)
        return [layer_names[index - 1] for index in indices]


DarknetYoloDetector = OpenCVDarknetYoloDetector


def detect_boxes(
    image: str | Path,
    model_path: str | Path,
    confidence: float = 0.25,
    **kwargs: Any,
) -> list[Box]:
    """Convenience function that returns only bounding boxes."""

    detector = YoloDetector(model_path=model_path, confidence=confidence, **kwargs)
    return [detection.box for detection in detector.detect(image)]


def draw_detections(
    image_path: str | Path,
    detections: Iterable[Detection],
    output_path: str | Path,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> Path:
    """Draw detection boxes onto an image and save it."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install OpenCV with `pip install visionparse[opencv]`.") from exc

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    for detection in detections:
        x1, y1, x2, y2 = detection.box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        if detection.label:
            cv2.putText(
                image,
                f"{detection.label} {detection.confidence:.2f}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output), image)
    if not ok:
        raise ValueError(f"Could not write image: {output}")
    return output


def _load_yolo_class(provider: str) -> Any:
    providers = [provider]
    if provider == "auto":
        providers = ["ultralytics", "ultralyticsplus"]

    errors: list[str] = []
    for name in providers:
        try:
            if name == "ultralytics":
                from ultralytics import YOLO

                return YOLO
            if name == "ultralyticsplus":
                from ultralyticsplus import YOLO

                return YOLO
        except ImportError as exc:
            errors.append(f"{name}: {exc}")

    raise ImportError(
        "YOLO support is optional. Install it with `pip install visionparse[yolo]` "
        "or install `ultralyticsplus` if you need the older provider. "
        f"Import attempts: {'; '.join(errors)}"
    )


def packaged_model_path(filename: str) -> Path:
    """Return the path to a model reference file packaged with VisionParse."""

    return Path(str(files("visionparse.models").joinpath(filename)))


def available_packaged_assets() -> dict[str, str]:
    """List small model/config assets bundled with the package."""

    root = files("visionparse.models")
    assets: dict[str, str] = {}
    for name in ("yolov3.cfg", "coco.names", "mscoco_label_map.pbtxt", "ssd_mobilenet_v2_coco.pbtxt"):
        resource = root.joinpath(name)
        if resource.is_file():
            assets[name] = str(Path(str(resource)))
    return assets


def load_class_names(path: str | Path) -> list[str]:
    """Read class labels from a Darknet ``.names`` file."""

    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _tensor_to_python(value: Any) -> Any:
    for method in ("detach", "cpu"):
        if hasattr(value, method):
            value = getattr(value, method)()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _as_rows(value: Any) -> list[list[float]]:
    value = _tensor_to_python(value)
    if value is None:
        return []
    return [list(row) for row in value]


def _as_flat_list(value: Any) -> list[Any]:
    value = _tensor_to_python(value)
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    return list(value)


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _label_for(names: Any, class_id: Optional[int]) -> str:
    if class_id is None:
        return ""
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _flatten_indices(indices: Any) -> list[int]:
    indices = _tensor_to_python(indices)
    if indices is None:
        return []
    result: list[int] = []
    for item in indices:
        if isinstance(item, (list, tuple)):
            if item:
                result.append(int(item[0]))
        else:
            result.append(int(item))
    return result
