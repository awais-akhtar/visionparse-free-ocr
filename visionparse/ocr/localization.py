"""Layout-aware OCR helpers.

OCR engines are good at finding words; downstream extraction usually needs the
reading order, columns, and visual grouping. These helpers turn word boxes into
lines/blocks and can render a plain-text view that keeps the original layout
much closer than a simple ``" ".join(words)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Iterable, Optional, Sequence


Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class TextToken:
    """A localized OCR token."""

    text: str
    box: Box
    confidence: Optional[float] = None

    @property
    def left(self) -> int:
        return self.box[0]

    @property
    def top(self) -> int:
        return self.box[1]

    @property
    def right(self) -> int:
        return self.box[2]

    @property
    def bottom(self) -> int:
        return self.box[3]

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def y_center(self) -> float:
        return self.top + self.height / 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "box": list(self.box),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class TextLine:
    """One visual line of localized text."""

    tokens: tuple[TextToken, ...]

    @property
    def text(self) -> str:
        return " ".join(token.text for token in self.tokens).strip()

    @property
    def box(self) -> Box:
        return union_boxes(token.box for token in self.tokens)

    @property
    def top(self) -> int:
        return self.box[1]

    @property
    def height(self) -> int:
        return max(1, self.box[3] - self.box[1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "box": list(self.box),
            "tokens": [token.to_dict() for token in self.tokens],
        }


@dataclass(frozen=True)
class TextBlock:
    """A group of nearby text lines."""

    lines: tuple[TextLine, ...]

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines).strip()

    @property
    def box(self) -> Box:
        return union_boxes(line.box for line in self.lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "box": list(self.box),
            "lines": [line.to_dict() for line in self.lines],
        }


def group_tokens_into_lines(
    tokens: Sequence[TextToken],
    *,
    y_tolerance: Optional[float] = None,
    min_y_tolerance: float = 4.0,
) -> list[TextLine]:
    """Group localized tokens into visual lines.

    The grouping is deliberately geometry-first. It works well with Tesseract
    ``image_to_data`` output and with EasyOCR/Keras OCR boxes converted to
    ``TextToken`` objects.
    """

    clean_tokens = [token for token in tokens if token.text.strip()]
    if not clean_tokens:
        return []

    heights = [token.height for token in clean_tokens if token.height > 0]
    tolerance = y_tolerance
    if tolerance is None:
        tolerance = max(min_y_tolerance, median(heights) * 0.55 if heights else min_y_tolerance)

    rough_lines: list[list[TextToken]] = []
    line_centers: list[float] = []

    for token in sorted(clean_tokens, key=lambda item: (item.y_center, item.left)):
        matched_index: Optional[int] = None
        for index, center in enumerate(line_centers):
            if abs(token.y_center - center) <= tolerance:
                matched_index = index
                break

        if matched_index is None:
            rough_lines.append([token])
            line_centers.append(token.y_center)
        else:
            rough_lines[matched_index].append(token)
            line_centers[matched_index] = median(t.y_center for t in rough_lines[matched_index])

    lines = [TextLine(tokens=tuple(sorted(line, key=lambda item: item.left))) for line in rough_lines]
    return sorted(lines, key=lambda line: (line.top, line.box[0]))


def group_lines_into_blocks(
    lines: Sequence[TextLine],
    *,
    vertical_gap_threshold: Optional[float] = None,
    horizontal_overlap_threshold: float = 0.1,
) -> list[TextBlock]:
    """Group lines into nearby text blocks.

    This is useful for menus and receipts where headings, item columns, and
    price columns should be kept together before extraction.
    """

    if not lines:
        return []

    sorted_lines = sorted(lines, key=lambda line: (line.top, line.box[0]))
    heights = [line.height for line in sorted_lines]
    max_gap = vertical_gap_threshold
    if max_gap is None:
        max_gap = max(8.0, median(heights) * 1.8)

    blocks: list[list[TextLine]] = [[sorted_lines[0]]]

    for line in sorted_lines[1:]:
        previous = blocks[-1][-1]
        gap = line.box[1] - previous.box[3]
        overlap = horizontal_overlap_ratio(previous.box, line.box)
        if gap <= max_gap and overlap >= horizontal_overlap_threshold:
            blocks[-1].append(line)
        else:
            blocks.append([line])

    return [TextBlock(lines=tuple(block)) for block in blocks]


def render_aligned_text(
    lines: Sequence[TextLine],
    *,
    char_width: Optional[float] = None,
    min_gap: int = 1,
) -> str:
    """Render OCR lines into monospaced, layout-preserving text."""

    if not lines:
        return ""

    all_tokens = [token for line in lines for token in line.tokens]
    if char_width is None:
        char_width = estimate_char_width(all_tokens)

    left_edge = min(token.left for token in all_tokens)
    rendered: list[str] = []

    for line in sorted(lines, key=lambda item: (item.top, item.box[0])):
        parts: list[str] = []
        cursor = 0
        for token in line.tokens:
            column = max(0, int(round((token.left - left_edge) / char_width)))
            if column <= cursor:
                gap = min_gap
            else:
                gap = max(min_gap, column - cursor)
            parts.append(" " * gap if parts else " " * max(0, column))
            parts.append(token.text)
            cursor = column + len(token.text)
        rendered.append("".join(parts).rstrip())

    return "\n".join(rendered).strip()


def align_tokens(
    tokens: Sequence[TextToken],
    *,
    y_tolerance: Optional[float] = None,
    char_width: Optional[float] = None,
) -> str:
    """Convenience function: tokens in, layout-aware text out."""

    return render_aligned_text(
        group_tokens_into_lines(tokens, y_tolerance=y_tolerance),
        char_width=char_width,
    )


def tokens_from_dicts(rows: Iterable[dict[str, Any]]) -> list[TextToken]:
    """Build tokens from dictionaries containing text and box values."""

    tokens: list[TextToken] = []
    for row in rows:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        box_value = row.get("box")
        if not box_value or len(box_value) != 4:
            continue
        confidence = row.get("confidence")
        tokens.append(
            TextToken(
                text=text,
                box=tuple(int(value) for value in box_value),  # type: ignore[arg-type]
                confidence=float(confidence) if confidence is not None else None,
            )
        )
    return tokens


def union_boxes(boxes: Iterable[Box]) -> Box:
    """Return one box covering every input box."""

    box_list = list(boxes)
    if not box_list:
        return (0, 0, 0, 0)
    return (
        min(box[0] for box in box_list),
        min(box[1] for box in box_list),
        max(box[2] for box in box_list),
        max(box[3] for box in box_list),
    )


def horizontal_overlap_ratio(first: Box, second: Box) -> float:
    """Return horizontal overlap divided by the smaller box width."""

    overlap = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    smaller_width = max(1, min(first[2] - first[0], second[2] - second[0]))
    return overlap / smaller_width


def estimate_char_width(tokens: Sequence[TextToken]) -> float:
    """Estimate a monospace character width from token boxes."""

    widths = [
        token.width / max(1, len(token.text))
        for token in tokens
        if token.width > 0 and token.text.strip()
    ]
    if not widths:
        return 8.0
    return max(1.0, median(widths))

