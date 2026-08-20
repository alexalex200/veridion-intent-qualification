# Intent Qualification — Writeup

## 3.1 Approach

The system is a four-stage cascade, where cost only escalates as far as a
query actually needs:

```
query
  │
  ▼
[1] ParsedQuery         structured constraints + industry "concepts" (regex + gazetteer + ontology lookup)
  │
  ▼
[2] Hard gates           reject companies that explicitly violate a checkable constraint
  │  (vectorized: dict/set lookups, no model calls)
  ▼
[3] Scoring & rank       blend NAICS match + keyword overlap + TF-IDF embedding similarity
  │
  ▼
[4] LLM verification     (optional) a local Ollama model re-checks only the ambiguous
      (borderline band)   middle band of the ranked shortlist - not the full candidate set
  │
  ▼
ranked, explainable results
```

This shape - cheap lexical/metadata filter first, expensive semantic
step only over a bounded shortlist - is the standard architecture for
production retrieval-with-reranking systems ([two-stage retrieval
cascades](https://www.emergentmind.com/topics/two-stage-retrieval-architecture),
[LlamaIndex's retrieval+reranking
guide](https://www.llamaindex.ai/blog/using-llms-for-retrieval-and-reranking-23cf2d3a14b6)),
and it's what stages 1-4 here map onto: stage 1-2 are the sparse/metadata
filter, stage 3 is the dense/lexical scorer, stage 4 is the reranker -
just with a locally-hosted LLM as the reranker instead of a cross-encoder,
since the task ("does this company really satisfy this intent") needs
more reasoning than a cross-encoder gives.

**Query parsing (`qualifier/query_parser.py` + `qualifier/taxonomy.py`).**
A query is decomposed into two kinds of signal. *Checkable constraints* —
country/region, employee count, revenue, founding year, public/private —
are pulled out with regex and a country/region gazetteer. *Industry
intent* is detected by matching the query against a small hand-authored
ontology of ~12 "concepts" (logistics, software, pharma, fintech, EV
battery supply chain, etc.), each carrying NAICS-code prefixes, keywords,
and — crucially — **expansion terms**. The expansion terms exist because
the query often describes the *buyer's* problem ("supply packaging for a
cosmetics brand") while a matching company's profile describes the
*supplier's* business ("contract packaging", "injection molding",
"corrugated cartons"). Expanding the query before embedding it closes
that vocabulary gap without needing an LLM to bridge it at query time.
Queries whose topic isn't in the ontology still work — they fall back to
raw keyword/embedding matching over their own words — degrading gracefully
rather than failing closed.

**Hard gates (`qualifier/filters.py`).** Cheap, deterministic checks run
before any scoring: a company is rejected only when a field is *present*
and *clearly violates* the constraint (e.g., `employee_count=50` against
"more than 1,000 employees"). A missing field never causes rejection — it's
recorded as `unverified` and nudges the score down slightly rather than
zeroing the company out. This was a deliberate response to the assignment's
warning that real company data has heavy missingness (in this dataset,
`employee_count` is null for 39% of records, `revenue` for 19%,
`year_founded` for 27%) — a system that hard-rejects on missing data would
silently drop a large share of legitimate matches.

**Scoring (`qualifier/scoring.py` + `qualifier/embeddings.py`).** Gate
survivors get a blended relevance score:
`0.45·embedding + 0.30·naics + 0.25·keyword`, each in [0,1], minus a small
penalty per unverified field. This mirrors the standard hybrid-search
recipe of fusing a sparse/exact signal with a dense/semantic one (BM25 +
dense retrieval in the RAG literature; here, NAICS + keyword play the
sparse/exact role and TF-IDF cosine plays the semantic-ish role) — the two
kinds of signal fail differently, so combining them recovers cases either
one misses alone. NAICS is weighted highest of the three because it's an
industry classification assigned by the data provider rather than inferred
from prose — when present, it's the most reliable signal. Embedding
similarity is computed with TF-IDF + cosine similarity (`scikit-learn`),
fit **once** over the whole company corpus, then scored against every
company in a single vectorized call — this is what makes the difference
between "cheap" and "one call per company." An optional
`sentence-transformers` backend is wired in behind an env var
(`QUALIFIER_USE_ST=1`); it falls back to TF-IDF automatically if
unavailable. (A more principled fusion than the current weighted sum would
be [Reciprocal Rank Fusion](https://www.digitalapplied.com/blog/hybrid-search-bm25-vector-reranking-reference-2026),
which combines rank positions instead of raw scores from different
distributions - noted in 3.2 as a considered-but-deferred improvement,
since RRF makes the required per-signal score breakdown less directly
interpretable.)

**LLM verification (`qualifier/llm_verifier.py`, optional).** After
scoring and sorting, companies whose score falls in an ambiguous middle
band (`min_score` ≤ score < `borderline_high`, default 0.45) get a single
verification call to a local Ollama model (`llama3.2:3b`). Companies
already scoring above the confidence threshold, or already excluded, never
reach this stage - and the LLM is only ever run over a bounded shortlist
(`max(top_k * 3, 30)` candidates), not the full gate-passing set, which
matters at scale (see 3.4). Because the model runs locally, this carries
no per-token API cost, only latency - so it sidesteps the "expensive" half
of Baseline A's problem while still getting genuine reasoning on the
hardest cases. See 3.3 for how much this actually helped, and where it
didn't.

**Why this design over the two baselines.** Baseline A (LLM per company)
is accurate but its cost and latency scale linearly with candidate count
regardless of how easy the query is — a structured filter like "public
software companies with >1,000 employees" gets the same expensive
treatment as a genuinely ambiguous one. Baseline B (raw embedding
similarity) is fast but conflates *topical* similarity with *relevance* —
its own example failure (cosmetics companies outranking packaging
suppliers) is exactly what concept-based query expansion is built to fix.
This system keeps Baseline B's cost profile for the bulk of the work, and
spends Baseline A's accuracy only on the slice of candidates where the
cheap signals disagree with each other or are simply too weak to be
confident.

## 3.2 Tradeoffs

I optimized for **cost, speed, and interpretability** first, and added the
**LLM verification stage as a bounded, opt-in accuracy top-up** rather than
a redesign around it.

- Every score is decomposable (`embedding`, `naics`, `keyword`,
  `unverified_fields`, `hard_fail_reasons`, and now `llm_verdict` with its
  reasoning are all returned per company) — I chose explainability over a
  black-box scorer, because a ranking system that can't say *why* it
  ranked something is hard to trust or debug.
- I chose TF-IDF over a dense sentence embedding model as the default. TF-IDF
  is exact-vocabulary-sensitive and won't catch pure paraphrase, but it's
  synchronous, needs no model download, and is trivially inspectable.
- I kept weighted-sum score fusion instead of switching to Reciprocal Rank
  Fusion, even though RRF is the more robust way to combine signals that
  live on different scales. The reason is the assignment's own
  deliverable: a decomposable `score_breakdown` per company is more useful
  for debugging and trust than a marginally more robust but opaque rank
  fusion, at this candidate-pool size.
- The LLM stage is bounded twice over: by score band (only the ambiguous
  middle gets verified) and by candidate window (only the top
  `max(top_k*3, 30)` scored companies are eligible at all, computed
  **after** sorting). I initially wired the verification loop to run over
  *every* gate-passing company above `min_score`, before sorting or
  windowing — for a broad query that's not "a handful of borderline
  cases," it's most of the candidate pool, and it made a 12-query batch
  run take significantly longer than it should have. This was a real bug
  caught by watching wall-clock time, not something I reasoned my way to
  in advance — worth stating plainly, since it's exactly the kind of
  cost mistake the assignment is warning against.
- Default `llm_votes=1` (a single temperature=0 call) rather than
  majority-voting by default, even though testing showed temperature=0
  alone does *not* fully eliminate verdict inconsistency (see 3.3) and
  published work on [LLM-as-judge
  reliability](https://arxiv.org/html/2510.27106v1) recommends multi-trial
  majority voting as the standard mitigation. `--llm-votes N` is available
  as an opt-in for when reliability matters more than latency; I didn't
  make it the default because it multiplies the already-slowest stage's
  cost by N for a gain that the same research describes as high-variance
  and subject to diminishing returns past a handful of trials.
- Hard gates are intentionally lenient on missing data (see 3.1). This
  trades strictness for recall — a company that's plausibly a match but
  missing `employee_count` stays in the running rather than being dropped.

## 3.3 Error Analysis

Concrete examples pulled from actual runs against the 457-company dataset
(after deduplication, see the data-quality note below).

**Prompt engineering mattered more than expected, and the first version
failed silently.** My first verification prompt just asked "does this
company match the query?" and returned `{"match": ..., "reason": ...}`.
Run against the "Logistics companies in Romania" borderline band, it
answered `true` for nearly every company shown to it — including
`Rompetrol` (an oil refiner) and `STILL` (a forklift manufacturer), the
exact false positives the stage exists to catch. Two changes fixed it:
(1) explicitly instructing the model to reject companies whose *core*
business is something else even if the query's vocabulary appears in the
text, and (2) asking for a short `"reasoning"` field *before* `"match"` in
the JSON output, giving the model a place to work through the distinction
instead of pattern-matching straight to an answer. A further, more
aggressive version with contrastive few-shot examples overcorrected in the
other direction — it then rejected `Brasov Industrial Portfolio`, a
company whose primary NAICS is literally "General Warehousing and
Storage," reasoning that its offerings were "more focused on facility
leasing." I kept the milder, reasoning-first version rather than the
few-shot one specifically because the few-shot version was tuned against
a handful of visible failures and promptly broke on a case outside that
set — a small, direct demonstration of prompt overfitting.

**Verified: even at temperature=0, the same call can flip.** Running the
identical prompt against `Portul Constanta` (a port/harbor operator) in
two separate calls produced `match: true` once ("...making it a logistics
company by definition") and `match: false` once ("...a specific and
distinct business model from logistics") in the same session, same model,
same temperature setting. This isn't a hypothetical caveat — it's an
observed result, and it matches published findings that [temperature=0
reduces but does not eliminate LLM-judge
inconsistency](https://arxiv.org/pdf/2606.13685). This is precisely the
"Inconsistent" weakness the assignment attributes to Baseline A; routing
only a fraction of companies through the LLM reduces its *cost*, but does
not, by itself, fix this failure mode for the companies that do reach it.
`--llm-votes N` (majority voting) is the mitigation on offer, at a
latency cost.

**False positive the cheap stages alone let through, that the LLM stage
fixes inconsistently.** Query: *"Logistics companies in Romania"*.
`STILL` (forklifts), `Rompetrol` (oil refining), `Transgaz` (gas
transmission), `Romgaz` (gas extraction), and `OSCAR` (fuel wholesale) all
clear the rule+embedding score threshold via `keyword_score ≈ 0.5-0.67` —
their descriptions legitimately contain words like "distribution" and
"supply chain," just describing an adjacent role, not the query's target.
NAICS matching (naics=0.0 for all of them) keeps them below the true
freight forwarders in rank but doesn't remove them. Across different runs
of the full 12-query batch, the LLM stage's actual behavior split: in one
run it correctly demoted `STILL` and `Rompetrol` below the qualification
threshold; in the run whose output ships in `results/`, it *also* let
`OSCAR`, `Transgaz`, and `Romgaz` through with `match: true` and a
plausible-sounding rationale each time ("Transgaz's...core offerings are
directly related to the management and transmission of natural gas,
indicating it is a logistics company"). Same prompt, same model, same
temperature — different outcome depending on which specific companies
landed in that day's borderline band. This is the concrete case the
verification stage was built to fix, and also the concrete evidence that
it does so unreliably (see the temperature=0 finding above).

**Weak signal, correctly low-confidence.** Query: *"E-commerce companies
using Shopify or similar platforms"*. The dataset has no technographic
field (no "uses Shopify" signal anywhere), so the system can only proxy
via the `ecommerce_platform` concept's keywords/NAICS. Only a handful of
companies clear the score threshold at all, with top scores far below the
~0.6–0.7 seen on well-supported queries like "pharmaceutical companies in
Switzerland." The LLM stage can't invent a signal that isn't in the data
either — asking it "does this company use Shopify" when nothing in the
profile says so just produces a plausible-sounding guess, which is worse
than the rule+embedding stage's honest low score. This is a case where
*not* trusting the LLM's confident-sounding reasoning is the right call.

**Coarse-industry over-matching.** Query: *"Public software companies with
more than 1,000 employees"*. Top rule+embedding results include `Fujitsu`,
`Capgemini`, `Atos`, `SAIC`, `Genpact`, `CGI` — large public IT-services /
consulting firms, not "software companies" in the product-company sense a
reader probably intends. This traces to the `software` concept's NAICS
prefixes including "Computer Systems Design Services," which covers IT
consultancies as well as software product companies. Because these
companies score high enough (naics=1.0, keyword=1.0) to clear
`borderline_high`, they're never sent to the LLM stage at all — a reminder
that the LLM safety net only catches errors in the *ambiguous* band, not
confidently-wrong ones.

## 3.4 Scaling to 100,000 companies

- **Vector index instead of brute-force cosine.** At 100k rows a single
  TF-IDF matrix multiply is still sub-second, but I'd run gates first (as
  now) to shrink the candidate set, then only build/query an ANN index
  (FAISS/HNSW) over survivors if the corpus grows well past that.
- **Precompute and persist the embedding index** rather than refitting on
  every process start, keyed on a hash of the dataset, invalidated on
  updates.
- **Index gate fields directly** (country_code, employee_count buckets,
  is_public, NAICS prefix) — a dict-of-sets, or push filtering into
  SQLite/DuckDB `WHERE` clauses — turning gate evaluation from O(n) per
  query into effectively O(matches).
- **The LLM stage's candidate window matters even more.** The fix
  described in 3.2 (verify only `max(top_k*3, 30)` candidates, chosen
  *after* sorting) is what keeps LLM call volume flat as the dataset
  grows — at 100k companies, an unbounded "verify everything in the
  borderline band" approach could mean thousands of calls per query
  instead of dozens. The bounded-window design is what makes this
  architecture scale-safe in the first place, not an afterthought.
  I'd also batch the verification calls (Ollama supports concurrent
  requests) rather than the current sequential loop.
- **Batch/async the dense-embedding option** if `sentence-transformers` is
  enabled — encode in batches on GPU rather than one call per company.

## 3.5 Failure Modes

**Where confident-but-wrong results are most likely:**

1. **Ontology gaps.** A query about an industry with no taxonomy entry
   loses both the naics_score and the concept-keyword boost, falling back
   to raw keyword/embedding overlap, with no signal to the user that this
   happened.
2. **NAICS-classification coarseness on confidently-scored companies.**
   As in the "software companies" example, a broad NAICS bucket can pull
   in adjacent-but-wrong companies at naics_score=1.0 - and because that
   pushes them above `borderline_high`, the LLM safety net never sees
   them. The LLM stage only patches errors in the ambiguous middle; it
   does nothing for errors the cheap stages are (wrongly) confident about.
3. **LLM verdict flips on rerun.** Demonstrated directly in 3.3: the same
   company, query, and prompt produced different verdicts at
   temperature=0 across calls. A single-vote verification result should
   be read as "probably right," not "checked."
4. **Subjective/temporal qualifiers the schema can't verify.** "Fast-growing,"
   "competing with traditional banks" — there's no growth-rate or
   competitive-positioning field, so these get silently treated as
   "fintech companies in Europe." The LLM stage, if asked, may produce a
   confident-sounding rationalization for a "fast-growing" claim it has no
   actual basis to verify - an LLM's fluency here is a liability, not a
   feature, unless the prompt explicitly tells it to say "cannot verify."

**What I'd monitor in production:**

- **Score distribution per query**, to flag queries the taxonomy/data
  can't actually support (like the Shopify example) as low-confidence
  rather than presenting a top-15 at face value.
- **Gate-pass rate**, to catch regex constraint mis-parses that silently
  turn a filtered query into an unfiltered one.
- **Concept-match coverage** — log queries matching zero taxonomy
  concepts; that list is the taxonomy-extension backlog.
- **LLM verdict agreement rate under repeat sampling**, on a background
  sample of borderline companies - given the observed temperature=0
  inconsistency, this is a cheap way to detect prompt/model drift before
  it shows up as a user-visible ranking flip. This is a direct application
  of the "self-consistency" metric from the LLM-judge-reliability research
  cited above, run continuously rather than as a one-off test.
- **LLM-vs-rule agreement rate** - track how often the LLM stage's verdict
  disagrees with what the rule+embedding score alone would have surfaced.
  A sudden shift here (after a prompt change, model upgrade, or taxonomy
  edit) is an early warning sign worth alerting on.

## Data-quality note

The provided dataset contained 26 sets of byte-identical duplicate rows
(same name, website, description — e.g. `ENERCON` and `Norhybrid Renewables`
each appeared twice verbatim). Left in, these would let one company occupy
two slots in a top-K ranking. `solution.py::load_companies` deduplicates
on exact record equality before the pipeline runs (477 → 457 companies).
Nested fields (`address`, `primary_naics`) were also stored inconsistently
— sometimes as real JSON objects, sometimes as Python `repr()` strings
(e.g. `"{'country_code': 'ro', ...}"`) — handled in `qualifier/models.py`
via a flexible parser that accepts either form.

## What I'd prioritize next

1. A small labelled eval set (even ~5 companies × 12 queries, hand-judged)
   to replace eyeballing with a measurable precision/recall number per
   query, and specifically to measure the LLM stage's net effect (does it
   improve precision more than its own inconsistency costs?) rather than
   assuming it helps.
2. Concurrent/batched LLM verification calls instead of the current
   sequential loop - the biggest remaining latency cost once the candidate
   window is bounded.
3. Widening the taxonomy, since it's currently the single biggest lever on
   quality for the rule+embedding stages and the most likely thing to be
   incomplete for queries outside the 12 examples.
4. Evaluating whether Reciprocal Rank Fusion improves on the current
   weighted-sum score combination now that there's a decomposable
   `score_breakdown` already logged per company to compare against.

## Sources consulted

- [Two-Stage Retrieval Architecture — EmergentMind](https://www.emergentmind.com/topics/two-stage-retrieval-architecture)
- [LLM Retrieval and Reranking: Two-Stage RAG Guide — LlamaIndex](https://www.llamaindex.ai/blog/using-llms-for-retrieval-and-reranking-23cf2d3a14b6)
- [Rating Roulette: Self-Inconsistency in LLM-As-A-Judge Frameworks](https://arxiv.org/html/2510.27106v1)
- [The Coin Flip Judge? Reliability and Bias in LLM-as-a-Judge Evaluation](https://arxiv.org/pdf/2606.13685)
- [Hybrid Search: BM25, Vector & Reranking Reference 2026 — Digital Applied](https://www.digitalapplied.com/blog/hybrid-search-bm25-vector-reranking-reference-2026)
