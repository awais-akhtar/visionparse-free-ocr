"""Turn OCR text into lightweight structured data."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .prices import PriceMention, extract_prices


@dataclass(frozen=True)
class MenuItem:
    """A simple item parsed from menu-like text."""

    name: str
    prices: tuple[PriceMention, ...]
    category: Optional[str] = None
    raw_line: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "prices": [price.to_dict() for price in self.prices],
            "raw_line": self.raw_line,
        }


@dataclass(frozen=True)
class StructuredDocument:
    """Structured representation of OCR text."""

    text: str
    lines: tuple[str, ...]
    prices: tuple[PriceMention, ...] = ()
    items: tuple[MenuItem, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "lines": list(self.lines),
            "prices": [price.to_dict() for price in self.prices],
            "items": [item.to_dict() for item in self.items],
            "metadata": dict(self.metadata),
        }


def clean_text(text: str) -> str:
    """Clean common OCR spacing and encoding noise without being too clever."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = text.replace("Â£", "£").replace("â‚¬", "€")
    text = text.replace("â€“", "–").replace("â€”", "—")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_lines(text: str, *, keep_blank: bool = False) -> list[str]:
    """Split cleaned text into lines."""

    lines = [line.strip() for line in clean_text(text).splitlines()]
    if keep_blank:
        return lines
    return [line for line in lines if line]


def extract_menu_items(
    text: str,
    *,
    default_currency: Optional[str] = None,
    allow_plain_number_prices: bool = False,
) -> list[MenuItem]:
    """Extract menu-like items by pairing each line with its prices."""

    items: list[MenuItem] = []
    current_category: Optional[str] = None

    for line in split_lines(text):
        prices = extract_prices(
            line,
            default_currency=default_currency,
            allow_plain_numbers=allow_plain_number_prices,
        )

        if not prices:
            if _looks_like_category(line):
                current_category = line.rstrip(":").strip()
            continue

        name = _remove_price_spans(line, prices)
        if not name:
            continue

        items.append(
            MenuItem(
                name=name,
                prices=tuple(prices),
                category=current_category,
                raw_line=line,
            )
        )

    return items


def structure_text(
    text: str,
    *,
    default_currency: Optional[str] = None,
    allow_plain_number_prices: bool = False,
) -> StructuredDocument:
    """Build a deterministic structure from OCR text."""

    cleaned = clean_text(text)
    lines = tuple(split_lines(cleaned))
    prices = tuple(
        extract_prices(
            cleaned,
            default_currency=default_currency,
            allow_plain_numbers=allow_plain_number_prices,
        )
    )
    items = tuple(
        extract_menu_items(
            cleaned,
            default_currency=default_currency,
            allow_plain_number_prices=allow_plain_number_prices,
        )
    )
    return StructuredDocument(text=cleaned, lines=lines, prices=prices, items=items)


def structure_with_llm(
    text: str,
    *,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    temperature: float = 0,
) -> str:
    """Use LangChain/OpenAI to clean OCR text into structured Markdown.

    The API key is read from the argument or ``OPENAI_API_KEY``. Nothing is
    hardcoded.
    """

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("Set OPENAI_API_KEY or pass api_key before using LLM cleanup.")

    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install LLM support with `pip install visionparse[llm]`.") from exc

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You clean OCR output. Keep only information supported by the text. "
                "Return tidy Markdown with item names, categories, and prices where possible.",
            ),
            ("user", "OCR text:\n\n{text}"),
        ]
    )
    llm = ChatOpenAI(model=model, temperature=temperature, api_key=key)
    response = (prompt | llm).invoke({"text": clean_text(text)})
    return str(response.content).strip()


def _remove_price_spans(line: str, prices: list[PriceMention]) -> str:
    result = line
    line_offset_prices: list[tuple[int, int]] = []
    for price in prices:
        local_start = line.find(price.raw)
        if local_start >= 0:
            line_offset_prices.append((local_start, local_start + len(price.raw)))

    for start, end in sorted(line_offset_prices, reverse=True):
        result = result[:start] + " " + result[end:]

    result = re.sub(r"[\s.\-–—:|]+$", "", result)
    result = re.sub(r"^[\s.\-–—:|]+", "", result)
    return re.sub(r"\s{2,}", " ", result).strip()


def _looks_like_category(line: str) -> bool:
    if not line:
        return False
    if len(line) > 50:
        return False
    if line.endswith(":"):
        return True
    words = line.split()
    return 1 <= len(words) <= 4 and not any(char.isdigit() for char in line)

