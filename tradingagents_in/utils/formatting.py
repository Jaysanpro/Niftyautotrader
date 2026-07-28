"""INR-aware number formatting (lakh/crore) and small text helpers."""

from __future__ import annotations

import math

CRORE = 1_00_00_000
LAKH = 1_00_000


def inr(value: float | int | None, decimals: int = 2) -> str:
    """Format a rupee amount using the Indian lakh/crore convention."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    sign = "-" if value < 0 else ""
    v = abs(float(value))
    if v >= CRORE:
        return f"{sign}Rs {v / CRORE:,.{decimals}f} Cr"
    if v >= LAKH:
        return f"{sign}Rs {v / LAKH:,.{decimals}f} L"
    return f"{sign}Rs {v:,.{decimals}f}"


def inr_plain(value: float | int | None, decimals: int = 2) -> str:
    """Rupee amount with Indian digit grouping (12,34,567.89)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    sign = "-" if value < 0 else ""
    v = abs(float(value))
    whole, frac = divmod(round(v, decimals), 1)
    digits = str(int(whole))
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        digits = ",".join(parts + [tail])
    frac_str = f"{frac:.{decimals}f}"[2:] if decimals else ""
    return f"{sign}Rs {digits}" + (f".{frac_str}" if decimals else "")


def pct(value: float | None, decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:+.{decimals}f}%"


def num(value: float | None, decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:,.{decimals}f}"


def truncate(text: str, limit: int = 280) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def bullet_list(items: list[str], limit: int | None = None) -> str:
    chosen = items[:limit] if limit else items
    return "\n".join(f"- {item}" for item in chosen) if chosen else "- (none)"


def section(title: str, body: str) -> str:
    return f"### {title}\n\n{body.strip()}\n"
