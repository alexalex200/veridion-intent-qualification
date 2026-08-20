# Intent Qualification — Writeup

## 3.1 Approach

The system is a three-stage pipeline, none of whose per-company steps call an LLM:

```
query
  │
  ▼
[1] ParsedQuery      structured constraints + industry "concepts" (regex + gazetteer + ontology lookup)
  │
  ▼
[2] Hard gates        reject companies that explicitly violate a checkable constraint
  │  (vectorized: dict/set lookups, no model calls)
  ▼
[3] Scoring & rank    blend NAICS match + keyword overlap + TF-IDF embedding similarity
  │
  ▼
ranked, explainable results
```

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
penalty per unverified field. NAICS match is weighted highest of the three
independent-signal weights because it's an industry classification
assigned by the data provider rather than inferred from prose — when
present, it's the most reliable of the three. Embedding similarity is
computed with TF-IDF + cosine similarity (`scikit-learn`), fit **once**
over the whole company corpus, then scored against every company in a
single vectorized call — this is what makes the difference between "cheap"
and "one call per company." An optional `sentence-transformers` backend is
wired in behind an env var (`QUALIFIER_USE_ST=1`) for teams that want denser
semantic matching and are willing to pay for a one-time model download; it
falls back to TF-IDF automatically if unavailable, so the pipeline never
hard-depends on network access.

**Why this design over the two baselines.** Baseline A (LLM per company)
is accurate but its cost and latency scale linearly with candidate count
regardless of how easy the query is — a structured filter like "public
software companies with >1,000 employees" gets the same expensive
treatment as a genuinely ambiguous one. Baseline B (raw embedding
similarity) is fast but conflates *topical* similarity with *relevance* —
its own example failure (cosmetics companies outranking packaging
suppliers) is exactly what concept-based query expansion is built to fix.
This system keeps Baseline B's cost profile while addressing its accuracy
gap with domain rules, and reserves the option of an LLM pass only for the
sliver of cases that need it (see 3.5).

## 3.2 Tradeoffs

I optimized for **cost, speed, and interpretability**, in that order, and
accepted **accuracy on genuinely subjective queries** as the cost.

- Every score is decomposable (`embedding`, `naics`, `keyword`,
  `unverified_fields`, `hard_fail_reasons` are all returned per company) —
  I chose explainability over squeezing out a few extra points of raw
  accuracy from a black-box scorer, because a ranking system that can't say
  *why* it ranked something is hard to trust or debug.
- I chose TF-IDF over a dense sentence embedding model as the default. TF-IDF
  is exact-vocabulary-sensitive and won't catch pure paraphrase, but it's
  synchronous, needs no model download, and is trivially inspectable. The
  dense backend is available as an opt-in upgrade, not the default — for a
  477-row dataset, the accuracy delta from paraphrase-matching didn't
  seem to outweigh introducing a hard dependency (torch) into the default
  path.
- I built the ontology by hand for ~12 domains rather than deriving concepts
  automatically (e.g., from the NAICS taxonomy or via clustering). This is
  fast to reason about and easy to extend, but it means the system's
  judgment quality on a topic is only as good as the taxonomy entry for
  it — a query about a domain with no entry gets materially weaker
  signal (naics_score and keyword-concept-boost both drop out; it's left
  with raw keyword overlap + embedding similarity only).
- Hard gates are intentionally lenient on missing data (see 3.1). This
  trades strictness for recall — a company that's plausibly a match but
  missing `employee_count` stays in the running rather than being dropped,
  which is right for a candidate-surfacing tool but would be wrong for a
  system making a final purchase/contact decision without human review.

## 3.3 Error Analysis

Concrete examples pulled from actual runs against the 457-company dataset
(after deduplication, see 3.5):

**False positive — generic-vocabulary drift.** Query: *"Logistics
companies in Romania"*. `STILL` (a forklift/material-handling equipment
maker) and `Rompetrol` (an oil refiner) both qualify with
`keyword_score ≈ 0.67`, matched via the `logistics` concept. Their
descriptions use words like "distribution", "supply chain", and
"industrial products" — genuinely present in the text, but describing an
adjacent role (equipment supplier to logistics operators; industrial fuel
distribution) rather than being a logistics company. This is the keyword
signal's core failure mode: it counts term overlap, not the *role* the
term is played in. NAICS matching (which STILL and Rompetrol both fail —
naics=0.0 in the actual run) is what keeps them below the true freight
forwarders (`Brasov Industrial Portfolio`, national rail operator `CFR`,
`Portul Constanta`, all naics=1.00) in the ranking, but they aren't
filtered out entirely.

**Weak signal, correctly low-confidence.** Query: *"E-commerce companies
using Shopify or similar platforms"*. The dataset has no technographic
field (no "uses Shopify" signal anywhere), so the system can only proxy
via the `ecommerce_platform` concept's keywords/NAICS. The result: only 10
companies clear the score threshold at all, and the top score is 0.18 —
an order of magnitude below the ~0.6–0.7 top scores on well-supported
queries like "pharmaceutical companies in Switzerland". This is arguably
the system working *correctly* — it has no basis for confidence and its
scores reflect that — but it will look like a "weak result set" if a user
expects a confident top-15, and a rank-only view (without the score) would
misrepresent that as a strong match.

**Coarse-industry over-matching.** Query: *"Public software companies with
more than 1,000 employees"*. The top results include `Fujitsu`,
`Capgemini`, `Atos`, `SAIC`, `Genpact`, `CGI` — large public IT-services /
consulting firms, not "software companies" in the product-company sense
a reader probably intends. This traces to the `software` concept's NAICS
prefixes (5112, 5415) including "Computer Systems Design Services", which
covers IT consultancies as well as software product companies — NAICS
granularity doesn't distinguish "sells software" from "sells IT services
that include software." A tighter prefix set (or a `keyword_score` weight
increase against "platform"/"product" language) would sharpen this, at
the cost of recall on companies that are hybrids of both.

## 3.4 Scaling to 100,000 companies

The current design already separates a cheap gate stage from a scoring
stage specifically so it scales past the sample size tested here, but at
100k some concrete changes would matter:

- **Vector index instead of brute-force cosine.** TF-IDF/dense similarity
  is currently one matrix multiply against every company; at 100k rows
  this is still sub-second, but the ceiling on that approach is in the
  low millions before it's worth introducing an ANN index (FAISS/HNSW).
  I'd run gates first (as now) to shrink the candidate set, then only
  build/query the vector index over survivors.
- **Precompute and persist the embedding index**, rather than fitting it
  fresh on every process start (`QualificationPipeline.__init__` currently
  refits TF-IDF on load). At 100k rows, re-fitting per run is wasted work
  if the underlying data hasn't changed — I'd cache the fitted matrix
  keyed on a hash of the dataset and invalidate on data updates.
- **Index gate fields directly** (country_code, employee_count buckets,
  is_public, NAICS prefix) rather than scanning linearly, e.g. a
  dict-of-sets keyed by country_code, or loading into SQLite/DuckDB and
  pushing the hard-gate filtering into a `WHERE` clause. This turns gate
  evaluation from O(n) per query into effectively O(matches).
- **Selective LLM pass for the score band right at the qualification
  threshold**, now affordable because gating + scoring has already cut
  100k down to a low hundreds. This is the natural place to reintroduce
  Baseline A's strength (see 3.5) without its cost problem.
- **Batch/async the dense-embedding option** if `sentence-transformers` is
  enabled — encode in batches on GPU rather than one call per company.

## 3.5 Failure Modes

**Where confident-but-wrong results are most likely:**

1. **Ontology gaps.** A query about an industry with no taxonomy entry
   loses both the naics_score and the concept-keyword boost, falling back
   to raw keyword/embedding overlap. It won't error, and it will still
   return a ranked list with plausible-looking scores in the same numeric
   range — but the ranking quality silently degrades. This is the most
   dangerous case because nothing in the output signals "the system
   doesn't actually understand this domain."
2. **NAICS-classification coarseness.** As in the "software companies"
   example above, a broad NAICS bucket can pull in adjacent-but-wrong
   companies with a full naics_score=1.0, which looks like a strong
   signal in the breakdown even though the classification itself is
   too coarse for the query's intent.
3. **Keyword overlap on shared vocabulary.** Industries that share
   generic business language ("distribution," "supply chain," "solutions")
   inflate keyword_score for companies with no real relevance, particularly
   for one- or two-word concept matches.
4. **Subjective/temporal qualifiers the schema can't verify.** "Fast-growing,"
   "competing with traditional banks" — the system has no growth-rate or
   competitive-positioning field, so these queries are silently treated as
   "fintech companies in Europe," dropping the qualifier without flagging
   that it was dropped.

**What I'd monitor in production:**

- **Score distribution per query.** A query whose top score is far below
  the historical median for its concept category (like the Shopify example)
  is a signal the taxonomy or data doesn't actually support that query —
  worth surfacing to the user as low-confidence rather than presenting a
  top-15 at face value.
- **Gate-pass rate.** If hard gates reject unusually few or unusually many
  candidates in the presence of missing fields, it suggests the regex
  constraint parser mis-parsed the query (e.g., a revenue figure it
  failed to extract, silently turning a filtered query into an unfiltered
  one).
- **Concept-match coverage.** Log when a query matches zero taxonomy
  concepts — that's the clearest signal a query needs a new ontology entry.
  Over time, this list is exactly the backlog for extending the taxonomy.
- **Sampled human/LLM audits on the score band near the threshold** — the
  cases most likely to be borderline-wrong are clustered right around
  `min_score`, not at the top of the ranking, so that's where spot-checking
  effort is best spent.

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
   query — right now tuning the score weights is guesswork.
2. The selective-LLM-verification pass described in 3.4/3.5, gated to only
   the borderline score band, to recover the judgment-heavy-query accuracy
   this design intentionally leaves on the table.
3. Widening the taxonomy, since it's currently the single biggest lever on
   quality and the most likely thing to be incomplete for queries outside
   the 12 examples.
