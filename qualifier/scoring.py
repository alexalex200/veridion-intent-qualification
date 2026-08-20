"""Combines three cheap, independent relevance signals into one score:

- naics_score: does the company's own industry classification match the
  concept(s) detected in the query? Strongest signal when available,
  because it's assigned by the data provider, not inferred from prose.
- keyword_score: literal keyword overlap between the query's terms
  (+ concept keywords) and the company's text blob. Catches exact
  vocabulary matches embeddings can dilute.
- embedding_score: cosine similarity between the (expansion-enriched)
  query and the company's text blob. Catches paraphrase / semantic
  matches keywords miss.

Weighted blend rather than any single signal, because each fails
differently: NAICS is coarse and sometimes absent from the labelled
data set; keyword overlap misses paraphrases; embeddings alone conflate
"topically similar" with "actually relevant" (the exact failure mode
the assignment calls out for Baseline B).

Corroboration discount: testing surfaced a consistent false-positive
pattern - companies that share generic business vocabulary with a query
("distribution", "supply chain", "transportation") without their own
industry classification supporting it at all (an oil refiner, a gas
utility, a forklift manufacturer all describe "distribution" somewhere,
none of them are logistics companies). When a query has an industry
expectation (naics_prefixes) and a company's classification flatly
disagrees (naics_score == 0), keyword overlap alone is weaker evidence -
it's corroborated by nothing structured - so it's discounted. This is the
retrieval analogue of a general finding in anomaly/relevance detection:
a single uncorroborated signal should carry less weight than one multiple
independent signals agree on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .models import Company
from .query_parser import ParsedQuery

W_EMBEDDING = 0.45
W_NAICS = 0.30
W_KEYWORD = 0.25
UNVERIFIED_PENALTY = 0.04
NAICS_MISMATCH_KEYWORD_PENALTY = 0.5


@dataclass
class ScoreBreakdown:
    embedding_score: float
    naics_score: float
    keyword_score: float
    final_score: float


def _naics_score(company: Company, parsed: ParsedQuery) -> float:
    """Graduated, not binary: some NAICS codes are a strong, specific fit
    for a concept (e.g. 5112 "Software Publishers" for "software company"),
    others are broader/adjacent (5415x "Computer Systems Design Services" -
    real, but describes IT-services firms as much as software product
    companies). A strong match earns full credit; a weak-only match earns
    partial credit rather than being indistinguishable from a perfect one."""
    if not parsed.naics_prefixes:
        return 0.0
    best = 0.0

    def tier(code: str) -> float:
        if any(code.startswith(p) for p in parsed.strong_naics_prefixes):
            return 1.0
        if any(code.startswith(p) for p in parsed.naics_prefixes):
            return 0.5
        return 0.0

    if company.primary_naics and company.primary_naics.code:
        best = max(best, tier(company.primary_naics.code))
    for sec in company.secondary_naics:
        if sec.code:
            best = max(best, tier(sec.code) * 0.6)
    return best


def _keyword_score(company: Company, parsed: ParsedQuery) -> float:
    if not parsed.keywords:
        return 0.0
    blob = company.text_blob().lower()
    hits = sum(1 for kw in parsed.keywords if kw in blob)
    return min(1.0, hits / max(3, len(parsed.keywords) * 0.5))


def compute_score(
    company: Company,
    parsed: ParsedQuery,
    embedding_similarity: float,
    unverified_count: int,
) -> ScoreBreakdown:
    naics = _naics_score(company, parsed)
    keyword = _keyword_score(company, parsed)
    embedding = max(0.0, float(embedding_similarity))

    if parsed.naics_prefixes and naics == 0.0:
        keyword *= NAICS_MISMATCH_KEYWORD_PENALTY

    relevance = W_EMBEDDING * embedding + W_NAICS * naics + W_KEYWORD * keyword
    final = relevance - UNVERIFIED_PENALTY * unverified_count
    final = max(0.0, min(1.0, final))

    return ScoreBreakdown(
        embedding_score=embedding, naics_score=naics, keyword_score=keyword, final_score=final
    )
