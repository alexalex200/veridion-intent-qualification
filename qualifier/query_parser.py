"""Turns a free-text query into structured constraints, without an LLM.

Two kinds of signal are extracted:
1. Hard, checkable constraints (country/region, employee count, revenue,
   founding year, public/private) via regex + gazetteer lookup.
2. Soft "what industry/role is this about" signal via substring matching
   against the concept ontology in taxonomy.py, which also produces
   query-expansion terms for the embedding stage.

Anything not recognized by a rule falls back to raw keyword/embedding
matching, so the system degrades gracefully on queries outside the
hand-authored taxonomy rather than failing closed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .taxonomy import COUNTRY_NAME_TO_CODE, REGION_ALIASES, REGION_GROUPS, CONCEPTS

_NUM = r"[\d][\d,\.]*"


def _num(text: str) -> float:
    return float(text.replace(",", ""))


def _scale(text: str, multiplier: Optional[str]) -> float:
    value = _num(text)
    if not multiplier:
        return value
    m = multiplier.lower().rstrip(".")
    if m in ("k", "thousand"):
        return value * 1_000
    if m in ("m", "million", "mm"):
        return value * 1_000_000
    if m in ("b", "bn", "billion"):
        return value * 1_000_000_000
    return value


@dataclass
class ParsedQuery:
    raw_query: str
    country_codes: set = field(default_factory=set)   # empty = no region constraint
    region_specified: bool = False
    is_public: Optional[bool] = None
    employee_min: Optional[float] = None
    employee_max: Optional[float] = None
    revenue_min: Optional[float] = None
    revenue_max: Optional[float] = None
    founded_min: Optional[int] = None
    founded_max: Optional[int] = None
    matched_concepts: list = field(default_factory=list)
    expansion_terms: list = field(default_factory=list)
    naics_prefixes: set = field(default_factory=set)
    keywords: set = field(default_factory=set)

    def expanded_text(self) -> str:
        return self.raw_query + " " + " ".join(self.expansion_terms)


def _parse_regions(query_lower: str, parsed: ParsedQuery) -> None:
    codes: set = set()
    found = False
    for alias, canonical in REGION_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", query_lower):
            codes |= REGION_GROUPS[canonical]
            found = True
    for group_name, group_codes in REGION_GROUPS.items():
        if re.search(rf"\b{re.escape(group_name)}\b", query_lower):
            codes |= group_codes
            found = True
    for name, code in COUNTRY_NAME_TO_CODE.items():
        if re.search(rf"\b{re.escape(name)}\b", query_lower):
            codes.add(code)
            found = True
    parsed.country_codes = codes
    parsed.region_specified = found


def _parse_numeric_constraints(query_lower: str, parsed: ParsedQuery) -> None:
    # Employee count: "more than 1,000 employees", "fewer than 200 employees",
    # "1000+ employees"
    m = re.search(rf"(more than|over|above|at least|>\s*)\s*({_NUM})\+?\s*employees", query_lower)
    if m:
        parsed.employee_min = _num(m.group(2))
    m = re.search(rf"(fewer than|less than|under|below|<\s*)\s*({_NUM})\s*employees", query_lower)
    if m:
        parsed.employee_max = _num(m.group(2))
    m = re.search(rf"({_NUM})\+\s*employees", query_lower)
    if m and parsed.employee_min is None:
        parsed.employee_min = _num(m.group(1))

    # Revenue: "revenue over $50 million", "over $50m in revenue"
    m = re.search(
        rf"(?:revenue[s]?\s+)?(?:of\s+)?(more than|over|above|at least|>\s*)\s*\$?\s*({_NUM})\s*"
        rf"(million|billion|thousand|m|bn|k)?\s*(?:in\s+)?(?:revenue)?",
        query_lower,
    )
    if m and ("revenue" in query_lower or "$" in query_lower):
        parsed.revenue_min = _scale(m.group(2), m.group(3))
    m = re.search(
        rf"(?:revenue[s]?\s+)?(?:of\s+)?(less than|under|below|<\s*)\s*\$?\s*({_NUM})\s*"
        rf"(million|billion|thousand|m|bn|k)?\s*(?:in\s+)?(?:revenue)?",
        query_lower,
    )
    if m and ("revenue" in query_lower or "$" in query_lower):
        parsed.revenue_max = _scale(m.group(2), m.group(3))

    # Founding year: "founded after 2018", "founded before 2015", "since 2020"
    m = re.search(r"founded\s+after\s+(\d{4})|since\s+(\d{4})", query_lower)
    if m:
        parsed.founded_min = int(m.group(1) or m.group(2))
    m = re.search(r"founded\s+before\s+(\d{4})", query_lower)
    if m:
        parsed.founded_max = int(m.group(1))

    # Public / private
    if re.search(r"\bpublic(ly traded)?\b", query_lower):
        parsed.is_public = True
    elif re.search(r"\bprivate(ly held)?\b", query_lower):
        parsed.is_public = False


def _parse_concepts(query_lower: str, parsed: ParsedQuery) -> None:
    for concept_key, concept in CONCEPTS.items():
        for alias in concept["aliases"]:
            if alias in query_lower:
                parsed.matched_concepts.append(concept_key)
                parsed.naics_prefixes |= set(concept["naics_prefixes"])
                parsed.keywords |= set(concept["keywords"])
                parsed.expansion_terms.extend(concept["expansion_terms"])
                break

    # Always include the query's own significant words as keywords too,
    # so queries outside the taxonomy still get keyword-overlap signal.
    stopwords = {
        "the", "a", "an", "in", "of", "for", "with", "and", "or", "companies",
        "company", "that", "could", "than", "more", "fewer", "less", "over",
        "under", "employees", "revenue", "founded", "after", "before", "public",
        "private", "using", "similar", "platforms", "to",
    }
    words = re.findall(r"[a-z][a-z\-]{2,}", query_lower)
    parsed.keywords |= {w for w in words if w not in stopwords}


def parse_query(query: str) -> ParsedQuery:
    parsed = ParsedQuery(raw_query=query)
    query_lower = query.lower()
    _parse_regions(query_lower, parsed)
    _parse_numeric_constraints(query_lower, parsed)
    _parse_concepts(query_lower, parsed)
    return parsed
