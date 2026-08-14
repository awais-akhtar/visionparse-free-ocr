"""YOLO object-detection helpers.

The module is safe to import without Ultralytics installed. The dependency is
loaded only when a detector is created.
"""

from __future__ import annotations

from dataclasses import dataclass
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

