"""Image preprocessing helpers for OCR."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Optional


Box = tuple[int, int, int, int]


def load_image(image: str | Path | Any, mode: str = "RGB") -> Any:
    """Load an image path or normalize a PIL image."""

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency installed by package
        raise ImportError("Pillow is required for image loading.") from exc

    if isinstance(image, (str, Path)):
        return Image.open(image).convert(mode)
    if hasattr(image, "convert"):
        return image.convert(mode)
    raise TypeError("Expected an image path or a PIL Image object.")


def resize_image(image: Any, scale: float = 1.5, max_side: Optional[int] = None) -> Any:
    """Resize an image by scale, optionally limiting its largest side."""

    if scale <= 0:
        raise ValueError("scale must be greater than 0")

    width, height = image.size
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    if max_side and max(new_width, new_height) > max_side:
        ratio = max_side / max(new_width, new_height)
        new_width = max(1, int(new_width * ratio))
        new_height = max(1, int(new_height * ratio))

    return image.resize((new_width, new_height))


def to_grayscale(image: Any) -> Any:
    """Convert an image to grayscale."""

    return image.convert("L")


def threshold_image(image: Any, threshold: int = 180) -> Any:
    """Convert a grayscale image to a black-and-white image."""

    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be between 0 and 255")
    gray = image.convert("L")
    return gray.point(lambda pixel: 255 if pixel >= threshold else 0)


def enhance_for_ocr(
    image: str | Path | Any,
    scale: float = 1.5,
    grayscale: bool = True,
    threshold: Optional[int] = None,
    denoise: bool = True,
    contrast: float = 1.6,
    sharpen: bool = False,
    max_side: Optional[int] = None,
) -> Any:
    """Apply a sensible OCR preprocessing pass.

    The defaults are deliberately mild: resize, grayscale, median denoise, and
    contrast. Thresholding is available, but not forced, because many menus and
    receipts lose detail when binarized too aggressively.
    """

    try:
        from PIL import ImageEnhance, ImageFilter
    except ImportError as exc:  # pragma: no cover - dependency installed by package
        raise ImportError("Pillow is required for preprocessing.") from exc

    prepared = load_image(image)
    prepared = resize_image(prepared, scale=scale, max_side=max_side)

    if grayscale:
        prepared = prepared.convert("L")
    if denoise:
        prepared = prepared.filter(ImageFilter.MedianFilter(size=3))
    if contrast and contrast != 1:
        prepared = ImageEnhance.Contrast(prepared).enhance(contrast)
    if sharpen:
        prepared = prepared.filter(ImageFilter.SHARPEN)
    if threshold is not None:
        prepared = threshold_image(prepared, threshold=threshold)

    return prepared


def crop_image(image: str | Path | Any, box: Box, padding: int = 0) -> Any:
    """Crop a PIL image using an xyxy box."""

    prepared = load_image(image)
    x1, y1, x2, y2 = box
    if padding:
        x1 -= padding
        y1 -= padding
        x2 += padding
        y2 += padding

    width, height = prepared.size
    clipped = (
        max(0, int(x1)),
        max(0, int(y1)),
        min(width, int(x2)),
        min(height, int(y2)),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError(f"Invalid crop box after clipping: {box}")

    return prepared.crop(clipped)


def crop_regions(image: str | Path | Any, boxes: Iterable[Box], padding: int = 0) -> list[Any]:
    """Crop several regions from one image."""

    prepared = load_image(image)
    return [crop_image(prepared, box, padding=padding) for box in boxes]


def image_to_bytes(image: Any, format: str = "PNG") -> bytes:
    """Encode a PIL image to bytes."""

    buffer = BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def opencv_preprocess(
    image: str | Path | Any,
    scale: float = 1.5,
    blur_kernel: int = 5,
    threshold: Optional[int] = None,
) -> Any:
    """OpenCV-style preprocessing kept for users coming from CV notebooks."""

    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install OpenCV with `pip install visionparse[opencv]`.") from exc

    if isinstance(image, (str, Path)):
        array = cv2.imread(str(image))
        if array is None:
            raise ValueError(f"Could not read image: {image}")
    else:
        array = cv2.cvtColor(np.array(load_image(image)), cv2.COLOR_RGB2BGR)

    if scale != 1:
        array = cv2.resize(array, (0, 0), fx=scale, fy=scale)
    gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    if blur_kernel and blur_kernel > 1:
        kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        gray = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    if threshold is not None:
        _, gray = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return gray

