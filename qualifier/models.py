"""Normalizes raw, messy company records into a consistent shape.

The source dataset stores nested objects (`address`, `primary_naics`,
`secondary_naics`) inconsistently: sometimes as real JSON objects,
sometimes as Python dict repr's stuffed into a string (e.g.
`"{'country_code': 'ro', ...}"`). Every field is also allowed to be
missing. This module is the single place that absorbs that mess so the
rest of the pipeline can assume clean, typed data.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Optional


def _flex_parse(value: Any) -> Any:
    """Best-effort parse of a value that may be a dict, a JSON string,
    or a Python-repr string, returning the original value if none apply."""
    if value is None or isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        for parser in (ast.literal_eval,):
            try:
                return parser(value)
            except (ValueError, SyntaxError):
                pass
    return value


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


@dataclass
class NaicsCode:
    code: Optional[str] = None
    label: Optional[str] = None
    share: Optional[float] = None

    @classmethod
    def from_raw(cls, raw: Any) -> Optional["NaicsCode"]:
        parsed = _flex_parse(raw)
        if not isinstance(parsed, dict):
            return None
        return cls(
            code=str(parsed.get("code")) if parsed.get("code") is not None else None,
            label=parsed.get("label"),
            share=_to_float(parsed.get("share")),
        )


@dataclass
class Address:
    country_code: Optional[str] = None
    region_name: Optional[str] = None
    town: Optional[str] = None
    raw_text: Optional[str] = None

    @classmethod
    def from_raw(cls, raw: Any) -> Optional["Address"]:
        if raw is None:
            return None
        parsed = _flex_parse(raw)
        if isinstance(parsed, dict):
            cc = parsed.get("country_code")
            return cls(
                country_code=cc.lower() if isinstance(cc, str) else None,
                region_name=parsed.get("region_name"),
                town=parsed.get("town"),
                raw_text=None,
            )
        if isinstance(parsed, str):
            # Free-text fallback, e.g. "Munich, Germany"
            return cls(raw_text=parsed)
        return None


@dataclass
class Company:
    raw: dict = field(repr=False)
    website: Optional[str] = None
    operational_name: Optional[str] = None
    year_founded: Optional[int] = None
    employee_count: Optional[int] = None
    revenue: Optional[float] = None
    is_public: Optional[bool] = None
    description: str = ""
    business_model: list = field(default_factory=list)
    target_markets: list = field(default_factory=list)
    core_offerings: list = field(default_factory=list)
    address: Optional[Address] = None
    primary_naics: Optional[NaicsCode] = None
    secondary_naics: list = field(default_factory=list)

    @classmethod
    def from_raw(cls, raw: dict) -> "Company":
        secondary = _flex_parse(raw.get("secondary_naics"))
        secondary_list = []
        for item in _as_list(secondary):
            code = NaicsCode.from_raw(item)
            if code:
                secondary_list.append(code)
        return cls(
            raw=raw,
            website=raw.get("website"),
            operational_name=raw.get("operational_name"),
            year_founded=_to_int(raw.get("year_founded")),
            employee_count=_to_int(raw.get("employee_count")),
            revenue=_to_float(raw.get("revenue")),
            is_public=raw.get("is_public") if isinstance(raw.get("is_public"), bool) else None,
            description=raw.get("description") or "",
            business_model=_as_list(raw.get("business_model")),
            target_markets=_as_list(raw.get("target_markets")),
            core_offerings=_as_list(raw.get("core_offerings")),
            address=Address.from_raw(raw.get("address")),
            primary_naics=NaicsCode.from_raw(raw.get("primary_naics")),
            secondary_naics=secondary_list,
        )

    @property
    def country_code(self) -> Optional[str]:
        return self.address.country_code if self.address else None

    @property
    def display_name(self) -> str:
        return self.operational_name or self.website or "Unknown company"

    def text_blob(self) -> str:
        """Composite free text used for keyword overlap and embedding
        similarity. Weighted by repetition, not by TF-IDF weights, so
        core_offerings/description dominate over incidental fields."""
        parts = [
            self.display_name,
            self.description,
            self.description,  # description carries the most semantic signal
            " ".join(self.core_offerings),
            " ".join(self.core_offerings),
            " ".join(self.target_markets),
            " ".join(self.business_model),
        ]
        if self.primary_naics and self.primary_naics.label:
            parts.append(self.primary_naics.label)
        for sec in self.secondary_naics:
            if sec.label:
                parts.append(sec.label)
        return " ".join(p for p in parts if p)
