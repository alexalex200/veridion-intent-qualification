"""Orchestrates the full qualification flow:

  raw records -> Company objects -> [fit embedder once over full corpus]
  query -> ParsedQuery -> for each company: hard gates -> score -> rank

Design intent: every per-company step here is a vectorized array op or a
dict/set lookup - nothing calls out to a network or a model per company.
The only "fit" cost (building the TF-IDF matrix, or optionally encoding
with sentence-transformers) happens once per dataset load, not once per
query, so repeated queries against the same company set are effectively
free after the first one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional

from .embeddings import build_embedder
from .filters import evaluate_gates
from .models import Company
from .query_parser import parse_query
from .scoring import compute_score


@dataclass
class QualificationResult:
    company: Company
    final_score: float
    embedding_score: float
    naics_score: float
    keyword_score: float
    gate_passed: bool
    hard_fail_reasons: list
    unverified: list
    matched_concepts: list

    def to_dict(self) -> dict:
        c = self.company
        return {
            "operational_name": c.display_name,
            "website": c.website,
            "country": c.country_code,
            "final_score": round(self.final_score, 4),
            "score_breakdown": {
                "embedding": round(self.embedding_score, 4),
                "naics": round(self.naics_score, 4),
                "keyword": round(self.keyword_score, 4),
            },
            "gate_passed": self.gate_passed,
            "hard_fail_reasons": self.hard_fail_reasons,
            "unverified_fields": self.unverified,
            "matched_concepts": self.matched_concepts,
        }


class QualificationPipeline:
    def __init__(self, raw_companies: List[dict]):
        self.companies = [Company.from_raw(r) for r in raw_companies]
        corpus = [c.text_blob() for c in self.companies]
        self.embedder = build_embedder(corpus)

    def qualify(
        self, query: str, top_k: int = 25, min_score: float = 0.12
    ) -> List[QualificationResult]:
        parsed = parse_query(query)
        embedding_scores = self.embedder.score(parsed.expanded_text())

        results: List[QualificationResult] = []
        for company, emb_score in zip(self.companies, embedding_scores):
            gate = evaluate_gates(company, parsed)
            if not gate.passed:
                continue  # hard-fail rejection: cheap, no scoring needed
            breakdown = compute_score(
                company, parsed, emb_score, unverified_count=len(gate.unverified)
            )
            if breakdown.final_score < min_score:
                continue
            results.append(
                QualificationResult(
                    company=company,
                    final_score=breakdown.final_score,
                    embedding_score=breakdown.embedding_score,
                    naics_score=breakdown.naics_score,
                    keyword_score=breakdown.keyword_score,
                    gate_passed=gate.passed,
                    hard_fail_reasons=gate.hard_fail_reasons,
                    unverified=gate.unverified,
                    matched_concepts=parsed.matched_concepts,
                )
            )

        results.sort(key=lambda r: r.final_score, reverse=True)
        return results[:top_k]
