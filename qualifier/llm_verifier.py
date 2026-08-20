"""Optional LLM verification stage, backed by a local Ollama model.

This is deliberately NOT the "LLM per company" baseline the assignment
warns against: it is only ever called on the slice of companies whose
rule+embedding score falls in an ambiguous middle band (see
qualifier/pipeline.py). Companies that clearly qualify or clearly don't
are decided by the cheap stages alone. Because the model runs locally via
Ollama, calls carry no per-token API cost - only latency - so the
"expensive" half of Baseline A's problem doesn't apply; the "slow" half
is mitigated by only invoking it on a small, already-narrowed candidate
set.

The "inconsistent" half is real even at temperature=0: verifying the same
company/query/prompt twice produced different verdicts in testing (see
WRITEUP.md). This matches published findings on LLM-as-judge reliability
- temperature=0 reduces but does not eliminate judge inconsistency, and
multi-trial majority voting is the standard mitigation (diminishing
returns past a handful of trials, since votes aren't independent). That's
implemented here as an opt-in `votes` parameter: with votes=1 (default),
a single temperature=0 call is used, prioritizing speed; with votes>1,
the model is sampled at moderate temperature multiple times and the
majority verdict wins, trading latency for reliability. Because this
only ever runs on the borderline band, even votes=3-5 stays cheap
relative to Baseline A's per-company LLM calls.

Uses only the stdlib (urllib) to talk to Ollama's local REST API, so this
file adds no new pip dependency - if Ollama isn't running, `available()`
returns False and the pipeline falls back to rule+embedding scoring only.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VerifyResult:
    match: Optional[bool]  # None = verifier call failed; treat as "no opinion"
    reason: str
    votes: dict = field(default_factory=dict)  # e.g. {"true": 2, "false": 1}, only when votes>1


class OllamaVerifier:
    def __init__(
        self,
        model: str = "llama3.2:3b",
        host: str = "http://localhost:11434",
        timeout: float = 30.0,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._available: Optional[bool] = None

    def available(self) -> bool:
        if self._available is None:
            try:
                req = urllib.request.Request(f"{self.host}/api/tags")
                with urllib.request.urlopen(req, timeout=3) as resp:
                    self._available = resp.status == 200
            except (urllib.error.URLError, OSError):
                self._available = False
        return self._available

    def _build_prompt(self, query: str, company_summary: str) -> str:
        # Three things shaped this prompt, each fixing a specific observed
        # failure rather than a hypothetical one:
        # (1) explicitly telling the model to reject companies whose CORE
        #     business is something else even if the query's vocabulary
        #     appears in passing - without this, a 3B model agreed with
        #     almost everything (an oil refiner and a forklift maker both
        #     "matched" a logistics query).
        # (2) asking for a short "reasoning" field BEFORE "match" in the
        #     JSON, giving the model a place to work through the
        #     distinction instead of pattern-matching straight to an
        #     answer. A further pass adding explicit contrastive examples
        #     made it strict to a fault (rejecting a literal
        #     warehousing-for-hire company) - overfitting to a handful of
        #     visible test cases in the other direction, so it was dropped.
        # (3) explicitly telling the model NOT to re-litigate numeric/
        #     factual constraints (employees, revenue, founding year,
        #     public/private, country) - those are already checked by
        #     separate hard gates before a company ever reaches this
        #     prompt. Without this, the model would confidently assert
        #     constraints were satisfied for companies whose profile
        #     explicitly says "unknown" for that field (e.g. claiming a
        #     company "was founded after 2018" when Founded: unknown) -
        #     fabricating confirmation of something it had no basis to
        #     confirm, rather than reasoning only about industry/role fit.
        return (
            "You are a strict B2B analyst qualifying ONE company against a search "
            "query for a company database.\n\n"
            f"Query: {query}\n\n"
            f"Company profile:\n{company_summary}\n\n"
            "Task: decide if this company's actual BUSINESS/INDUSTRY/ROLE is what "
            "the query is asking for - not a company that merely uses, sells to, or "
            "is loosely associated with that space. Be strict: a company whose core "
            "business is something else does NOT match just because the query's "
            "vocabulary appears somewhere in its description.\n\n"
            "Important: any numeric or factual filters in the query (employee count, "
            "revenue, founding year, public/private, country) have ALREADY been "
            "checked separately before this company reached you - do not use them to "
            "justify your decision, and never assume a value marked 'unknown' in the "
            "profile satisfies the query. Base your decision ONLY on whether the "
            "company's business itself matches the query's industry/role/product "
            "intent.\n\n"
            "First think in a short \"reasoning\" field (1-2 sentences) about the "
            "business/industry/role fit only, then decide.\n"
            'Respond with ONLY compact JSON in this exact shape: '
            '{"reasoning": "<1-2 sentences>", "match": true or false}'
        )

    def _call_once(self, prompt: str, temperature: float) -> VerifyResult:
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": temperature},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            parsed = json.loads(body["response"])
            reason = parsed.get("reasoning", parsed.get("reason", ""))
            return VerifyResult(match=bool(parsed.get("match")), reason=str(reason))
        except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
            return VerifyResult(match=None, reason=f"verifier error: {exc}")

    def verify(self, query: str, company_summary: str, votes: int = 1) -> VerifyResult:
        prompt = self._build_prompt(query, company_summary)
        if votes <= 1:
            return self._call_once(prompt, temperature=0)

        samples = [self._call_once(prompt, temperature=0.4) for _ in range(votes)]
        tally = Counter(s.match for s in samples if s.match is not None)
        if not tally:
            return VerifyResult(match=None, reason="all verifier calls failed", votes={})
        winner, _ = tally.most_common(1)[0]
        # Prefer the reasoning from a sample that agrees with the winning verdict
        reason = next((s.reason for s in samples if s.match == winner), "")
        return VerifyResult(
            match=winner,
            reason=reason,
            votes={str(k).lower(): v for k, v in tally.items()},
        )
