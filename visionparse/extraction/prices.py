"""Price extraction from noisy OCR text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Optional


CURRENCY_PATTERN = r"(?:[$£€¥₹]|USD|GBP|EUR|PKR|Rs\.?|INR|AED|SAR|CAD|AUD)"
AMOUNT_PATTERN = r"(?:\d{1,3}(?:[,\s]\d{3})+|\d+)(?:[.,]\d{1,2})?"
PRICE_PATTERN = re.compile(
    rf"(?P<prefix>{CURRENCY_PATTERN})?\s*"
    rf"(?P<amount>{AMOUNT_PATTERN})"
    rf"\s*(?P<suffix>{CURRENCY_PATTERN})?(?!\s*\d)"
    rf"(?P<trailing>\s*/-)?",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class PriceMention:
    """A price found in text."""

    raw: str
    amount: Decimal
    currency: Optional[str] = None
    start: int = 0
    end: int = 0
    line: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "amount": str(self.amount),
            "currency": self.currency,
            "start": self.start,
            "end": self.end,
            "line": self.line,
        }


def extract_prices(
    text: str,
    *,
    default_currency: Optional[str] = None,
    allow_plain_numbers: bool = False,
    min_amount: Decimal | int | str = Decimal("0"),
    max_amount: Optional[Decimal | int | str] = None,
) -> list[PriceMention]:
    """Find prices in OCR text.

    By default, a match needs a currency symbol/code, a decimal separator, or
    the common ``/-`` suffix. Set ``allow_plain_numbers=True`` for menus that
    print prices as plain integers.
    """

    minimum = _to_decimal(min_amount)
    maximum = _to_decimal(max_amount) if max_amount is not None else None
    mentions: list[PriceMention] = []

    for line_start, line in _iter_lines_with_offsets(text):
        for match in PRICE_PATTERN.finditer(line):
            prefix = _clean_currency(match.group("prefix"))
            suffix = _clean_currency(match.group("suffix"))
            currency = prefix or suffix or default_currency
            trailing = bool(match.group("trailing"))
            amount_text = match.group("amount")
            raw = match.group(0).strip()

            if not raw:
                continue
            if not _looks_like_price(amount_text, currency, trailing, allow_plain_numbers):
                continue

            amount = normalize_amount(amount_text)
            if amount is None:
                continue
            if amount < minimum:
                continue
            if maximum is not None and amount > maximum:
                continue

            start = line_start + match.start()
            end = line_start + match.end()
            mentions.append(
                PriceMention(
                    raw=raw,
                    amount=amount,
                    currency=currency,
                    start=start,
                    end=end,
                    line=line.strip() or None,
                )
            )

    return mentions


def normalize_amount(amount: str) -> Optional[Decimal]:
    """Normalize amount text to Decimal."""

    cleaned = amount.strip().replace(" ", "")
    if not cleaned:
        return None

    if "," in cleaned and "." not in cleaned:
        pieces = cleaned.split(",")
        if len(pieces[-1]) in {1, 2} and not all(len(piece) == 3 for piece in pieces[1:]):
            cleaned = "".join(pieces[:-1]) + "." + pieces[-1]
        else:
            cleaned = cleaned.replace(",", "")
    else:
        cleaned = cleaned.replace(",", "")

    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def prices_by_line(text: str, **kwargs: Any) -> list[tuple[str, list[PriceMention]]]:
    """Return every line with prices found on that line."""

    rows: list[tuple[str, list[PriceMention]]] = []
    for line in text.splitlines():
        prices = extract_prices(line, **kwargs)
        if prices:
            rows.append((line, prices))
    return rows


def strip_prices(line: str, prices: Iterable[PriceMention]) -> str:
    """Remove price mentions from a single line."""

    result = line
    for price in sorted(prices, key=lambda item: item.start, reverse=True):
        if price.line != line.strip() and price.line != line:
            result = result.replace(price.raw, " ")
        else:
            local_start = max(0, price.start)
            local_end = max(local_start, price.end)
            if local_end <= len(result):
                result = result[:local_start] + " " + result[local_end:]
            else:
                result = result.replace(price.raw, " ")
    result = re.sub(r"[\s.\-–—:|]+$", "", result)
    result = re.sub(r"^[\s.\-–—:|]+", "", result)
    return re.sub(r"\s{2,}", " ", result).strip()


def _iter_lines_with_offsets(text: str) -> Iterable[tuple[int, str]]:
    offset = 0
    for line in text.splitlines(keepends=True):
        clean_line = line.rstrip("\r\n")
        yield offset, clean_line
        offset += len(line)
    if text and not text.endswith(("\n", "\r")) and "\n" not in text:
        return


def _looks_like_price(
    amount: str,
    currency: Optional[str],
    trailing: bool,
    allow_plain_numbers: bool,
) -> bool:
    if currency or trailing:
        return True
    if "." in amount or "," in amount:
        return True
    return allow_plain_numbers


def _clean_currency(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = value.strip()
    aliases = {"rs": "Rs.", "rs.": "Rs."}
    return aliases.get(normalized.lower(), normalized.upper() if len(normalized) == 3 else normalized)


def _to_decimal(value: Decimal | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))
