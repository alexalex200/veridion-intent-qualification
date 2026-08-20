"""Orchestrates the full qualification flow:

  raw records -> Company objects -> [fit embedder once over full corpus]
  query -> ParsedQuery -> for each company: hard gates -> score -> rank
                                                              |
                                                   [optional] LLM verification
                                                   only for the borderline band

Design intent: every per-company step in the rule+embedding stage is a
vectorized array op or a dict/set lookup - nothing calls out to a network
or a model per company. The only "fit" cost (building the TF-IDF matrix,
or optionally encoding with sentence-transformers) happens once per
dataset load, not once per query, so repeated queries against the same
company set are effectively free after the first one.

The optional LLM stage (qualifier/llm_verifier.py, backed by a local
Ollama model) is only invoked on companies whose rule+embedding score
lands in an ambiguous middle band - clearly-qualifying and clearly-
non-qualifying companies never reach it. This keeps the expensive step
bounded to a small slice of candidates per query instead of the full
candidate set, unlike the "LLM per company" baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .embeddings import build_embedder
from .filters import evaluate_gates
from .llm_verifier import OllamaVerifier
from .models import Company
from .query_parser import parse_query
from .scoring import compute_score

LLM_MATCH_BOOST = 0.15
LLM_REJECT_MULTIPLIER = 0.4


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
    llm_verdict: Optional[dict] = None

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
            "llm_verdict": self.llm_verdict,
        }


class QualificationPipeline:
    def __init__(
        self,
        raw_companies: List[dict],
        llm_verifier: Optional[OllamaVerifier] = None,
    ):
        self.companies = [Company.from_raw(r) for r in raw_companies]
        corpus = [c.text_blob() for c in self.companies]
        self.embedder = build_embedder(corpus)
        self.llm_verifier = llm_verifier

    def qualify(
        self,
        query: str,
        top_k: int = 25,
        min_score: float = 0.12,
        use_llm: bool = True,
        borderline_high: float = 0.45,
        llm_votes: int = 1,
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

        # Sort BEFORE verifying and only spend LLM calls on a bounded window
        # around the cutoff (standard cascade-retrieval practice: the
        # expensive stage only ever sees a fixed-size shortlist, not every
        # gate-passing company - for a broad query that could be hundreds).
        results.sort(key=lambda r: r.final_score, reverse=True)
        candidate_window = max(top_k * 3, 30)

        if use_llm and self.llm_verifier and self.llm_verifier.available():
            for result in results[:candidate_window]:
                # A high total score isn't enough to skip verification on
                # its own if NAICS only weakly corroborates it (score < 1.0
                # - see scoring.py's graduated NAICS credit). Observed case:
                # Globant (IT-services, naics=0.5) cleared borderline_high
                # on keyword+embedding alone and skipped verification
                # entirely, remaining unrejected while near-identical
                # naics=0.5 peers (Fujitsu, Capgemini...) - which scored
                # lower and DID reach the LLM - were correctly rejected by
                # it. Only a naics=1.0 (industry-confirmed) match is
                # trusted to skip the check outright.
                if result.final_score >= borderline_high and result.naics_score >= 1.0:
                    continue  # confident AND industry-confirmed, no need to spend a call
                verdict = self.llm_verifier.verify(
                    query, result.company.summary_for_llm(), votes=llm_votes
                )
                result.llm_verdict = {
                    "match": verdict.match, "reason": verdict.reason, "votes": verdict.votes
                }
                if verdict.match is True:
                    result.final_score = min(1.0, result.final_score + LLM_MATCH_BOOST)
                elif verdict.match is False:
                    result.final_score *= LLM_REJECT_MULTIPLIER
                # verdict.match is None (verifier call failed): leave score untouched

        # An explicit LLM rejection excludes a company outright, regardless
        # of how high its rule-based score was. The multiplier above still
        # runs (useful if a caller wants to see the demoted score), but is
        # not sufficient on its own: a company with a high pre-verification
        # score (Globant: keyword=1.0, embedding=0.15, naics=0.5 -> 0.47)
        # survives even a 0.6x demotion, which would leave it "qualified"
        # while its own llm_verdict says match=False - a contradiction a
        # caller trusting the qualified list wouldn't see unless they
        # inspected every verdict by hand.
        results = [
            r for r in results
            if r.final_score >= min_score
            and not (r.llm_verdict and r.llm_verdict["match"] is False)
        ]
        results.sort(key=lambda r: r.final_score, reverse=True)
        return results[:top_k]
