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
from prose — when present, it's the most reliable signal.

Two refinements to that base scheme came directly out of error analysis
against real results, not from design up front:

- **NAICS is graduated, not binary.** Some codes are a precise fit for a
  concept (`513210` "Software Publishers" for "software company"), others
  are broad/adjacent (`541511`/`541512`, "Computer Systems Design
  Services"/"Custom Computer Programming Services" - real IT work, but
  covers consultancies as much as software product companies). Each
  taxonomy concept can now declare a `strong_naics_prefixes` subset
  (full 1.0 credit) versus its full `naics_prefixes` (0.5 credit) - see
  the "coarse-industry over-matching" fix in 3.3.
- **Corroboration discount.** When a query carries an industry expectation
  (`naics_prefixes` non-empty) and a company's own classification flatly
  disagrees (`naics_score == 0`), keyword overlap alone is discounted
  (×0.5) rather than trusted at face value - it's the retrieval analogue
  of a general finding in anomaly/relevance detection that an
  uncorroborated single signal should carry less weight than one multiple
  independent signals agree on. Concretely: an oil refiner, a gas utility,
  and a forklift manufacturer all mention "distribution"/"supply chain"
  somewhere in their description (real text, adjacent meaning), and none
  of them have a transportation/warehousing NAICS code - the discount
  catches this class of false positive deterministically, without relying
  on the (measurably inconsistent, see 3.3) LLM stage to catch it instead.

Embedding
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

**A second, independent prompt bug: fabricating confirmation of
unverifiable fields.** Separately from the reasoning-quality issue above,
the verification prompt originally let the model freely discuss employee
count, revenue, and founding year - fields already checked by hard gates
before a company ever reaches the LLM stage. For "Clean energy startups
founded after 2018 with fewer than 200 employees," two companies with
`employee_count: null` and `year_founded: null` in their actual profile
got `match: true` with reasoning that explicitly asserted *"it was founded
after 2018 and has fewer than 200 employees"* - fabricated, not verified,
since the model had no data to check that against. The fix: the prompt
now explicitly tells the model those fields are pre-checked and instructs
it to never assume a value marked "unknown" satisfies the query, reasoning
only about industry/role fit. Re-running the same three companies after
the fix, no response mentioned founding year or employee count at all.
This is worth flagging as its own class of LLM-stage risk, distinct from
the temperature=0 inconsistency above: the model wasn't uncertain here, it
was *confidently wrong* about something outside its actual knowledge -
grounding what the model is and isn't allowed to reason about turned out
to matter as much as strictness about the query's actual intent. With
both prompt fixes and the scoring fixes below in place, "Logistics
companies in Romania" went from 15 results (many false positives) to 4
(`Portul Constanta`, `Brasov Industrial Portfolio`, `CFR`, and one
residual borderline case, `METRO România`, a grocery wholesaler the LLM
verified as a match on the strength of its own distribution network - a
defensible but debatable call, and a reminder that "much cleaner" isn't
"perfect").

**A silent, wrong assumption about the data - caught by auditing, not
intuition.** The taxonomy was originally hand-written from general NAICS
knowledge, e.g. "software companies are NAICS 5112." Results looked
plausible (companies still ranked, scores were in a normal range) which is
exactly what made this dangerous - nothing errored. Auditing every
concept's `naics_prefixes` against the 105 distinct NAICS codes actually
present in this dataset surfaced three real bugs: (1) this dataset uses
**NAICS 2022**, which reclassified "Software Publishers" from `511210` to
`513210` - the "software" and "saas_hr" concepts had been matching *zero*
software-product companies by NAICS this entire time, silently falling
back to keyword/embedding-only scoring for what should have been their
strongest signal; (2) the EV-battery concept had `335911` where the
dataset actually uses `335910` ("Battery Manufacturing") - a one-digit
transcription error that also matched nothing; (3) `454110` for
"e-commerce platform" doesn't appear anywhere in the dataset at all - not
a bug, but a genuine absence of any reliable structured signal for that
query, addressed by leaving the prefix list empty rather than pretending
otherwise (see `qualifier/taxonomy.py`). Fixing (1) and (2) changed
real output, not just internal scores:

  | Query | Before | After |
  |---|---|---|
  | *B2B SaaS companies providing HR solutions in Europe* (top 5) | Globant, ASCIA, HRWare Consulting, Sincron HR, Pandapé (IT consultancies mixed with real HR software) | Bizneo HR, Personio, Sincron HR, Pandapé, BambooHR - all `naics=1.0`, all genuine HR software vendors |
  | *Companies that manufacture or supply critical components for EV battery production* (top score) | 0.58 (`naics=0.0` for nearly every result - the concept was running on keyword/embedding alone) | 0.72 (`naics=1.0` for the top 15 - Fengyang Pengen, CIDEcell, Altmin, Stratus Materials: genuine battery-materials manufacturers) |

**A second audit pass, prompted by manually inspecting mid-ranked results
rather than just the top of each list, found three more issues of the
same shape.** (1) The `335911`/`335910` typo from the EV-battery concept
turned out to be copy-pasted into `clean_energy` too, silently blocking
battery-storage companies there as well - same fix, same root cause,
missed the first time because I fixed the concept the query obviously
pointed at (EV battery) and didn't check whether the same mistake existed
elsewhere. (2) Scanning results with `naics=0` that still scored
reasonably well (the corroboration discount halves rather than zeroes
keyword credit, so residual false-taxonomy matches don't vanish, they
just rank lower - which makes them findable) surfaced `326199` ("All
Other Plastics Product Manufacturing") as the dominant code for
cosmetic-packaging specialists: `SZ SJ Packaging` (whose own description
says "specialized in the production of cosmetic packaging, including
custom lipstick tubes, eyeshadow palettes, skincare bottles") was scoring
`naics=0` and ranking below companies with no more real relevance than it
had, purely because a common code was missing from the taxonomy. Added as
a *weak* prefix (0.5 credit, not 1.0), since it's a broad catch-all shared
with unrelated plastics manufacturers - Crystal International's score
went from 0.39 to 0.59 after the fix. (3) The same scan surfaced a genuine
LLM failure the earlier fixes hadn't touched: `Algavo`, a marine-biomass
/ bioeconomy company with no clean-energy business at all, was scoring
`naics=0.5` (via the broad `541690` "Other Scientific and Technical
Consulting Services" code, shared with actual renewable-energy
consultancies) and got LLM-approved with the reasoning *"clean energy
startups in the marine bioeconomy space"* - a category that doesn't
exist, a non-sequitur the LLM stage was supposed to catch and didn't.
Rather than trying to prompt-engineer around one specific bad case, the
fix was upstream: `541690` was too broad to be useful signal for *any*
concept (it's shared across unrelated consulting fields), so it was
removed from `clean_energy`'s prefixes entirely instead of just
downweighted. After the fix, `Algavo` no longer appears in the top 15 at
all, and legitimate renewable-energy wind/turbine companies that used to
depend on the LLM correctly guessing their relevance (`Fred. Olsen 1848`,
`World Wide Wind`, `Verta`, `Ventum Dynamics`, `Norhybrid Renewables`) now
carry `naics=1.0` directly via a newly-added `333611` ("Turbine and
Turbine Generator Set Units Manufacturing") prefix - most of them skip
the LLM stage entirely now, which is strictly better: a deterministic
correct answer beats a probabilistic one that happened to land right.
The general lesson, stated plainly: **the taxonomy is the least tested
part of this system, and it's easy to be technically not-wrong (a real
NAICS code, a real industry description) while still building the wrong
signal.** The audit script (`naics_audit.py`-style, per 3.4/priorities)
and manually reading mid-ranked results rather than only the top-5 are
what caught these, in that order of usefulness.

**Coarse-industry over-matching - fixed in two passes, not one.** Query:
*"Public software companies with more than 1,000 employees"*. Before the
graduated-NAICS fix, the full top-15 was IT-services/consulting firms
(`Fujitsu`, `Capgemini`, `Atos`, `SAIC`, `Genpact`, `CGI`, `NTT DATA`,
`Tata Consultancy Services`, `Wipro`...), all `naics=1.0` under the old
binary scoring, all confidently above `borderline_high` and therefore
never reaching the LLM stage at all. Marking `5415x` codes as weak (0.5
credit) instead of strong dropped their scores enough to land in the
borderline band, and the LLM stage correctly rejected most of them:
*"Fujitsu's primary industry is Computer Systems Design Services, which
[does not match a software product company]"*. But `Globant` (score
0.467, same weak `naics=0.5`) still cleared `borderline_high` outright and
skipped verification entirely - on keyword+embedding alone, not because
its industry classification was any stronger than Fujitsu's. This
revealed a second bug, one level up from the taxonomy: **the "skip LLM if
confident" rule only checked total score, not what the score was made
of.** A company could reach `borderline_high` on keyword/embedding alone
and never have its industry classification checked at all - the exact
gap a corroboration-style design is supposed to close. Fixed by requiring
`naics_score >= 1.0` (industry-*confirmed*, not just industry-adjacent) in
addition to the score threshold before skipping verification
(`qualifier/pipeline.py`). A related, sharper bug surfaced by the same fix:
even after Globant correctly reached the LLM and got `match: false`, it
*still* appeared in the qualified results, because the 0.4x rejection
demotion (0.467 → 0.187) wasn't enough to push it below the 0.12 default
threshold - a company explicitly marked "does not match" remained in its
own "qualified companies" list. Demotion alone was never going to be
airtight for this case, since how much demotion is "enough" depends on
how high the pre-verification score was; an explicit `match: false`
verdict now excludes a company outright, independent of `min_score`.
This is the concrete limit of the design stated in 3.1, now narrowed: the
LLM safety net catches errors in the ambiguous band **and** in
industry-unconfirmed high scorers, but still does nothing for a company
that manages `naics=1.0` (a real, if occasionally too-coarse, industry
match) and a high score both at once.

**Weak signal, and the LLM stage makes it *worse*, not better, when
forced to guess.** Query: *"E-commerce companies using Shopify or similar
platforms"*. The dataset has no technographic field at all - confirmed by
the NAICS audit above, not assumed - so `ecommerce_platform` intentionally
carries no `naics_prefixes`. Before any LLM involvement, this leaves the
system with only weak keyword/embedding signal and correctly low scores.
Two different observed runs disagreed on what to do with that weak
signal: one run returned zero qualifying companies (every borderline
candidate got LLM-rejected, arguably the most honest possible outcome
given there's nothing to verify against); the run shipped in `results/`
returned exactly one - `Flextribe`, a packaging company - with the LLM
reasoning *"Flextribe's core business is manufacturing and selling
eco-friendly packaging"* as its justification for `match: true`, which
does not actually support matching an e-commerce/Shopify query at all.
That's a non-sequitur verdict, not a defensible edge-case judgment - the
clearest evidence in this whole exercise that when the underlying data
has no real signal, adding an LLM verification pass doesn't rescue the
query; it just adds a plausible-sounding wrapper around what is, either
way, a guess.

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
   Mitigated but not eliminated by the strong/weak NAICS split in 3.1/3.3 -
   `Globant` still clears `borderline_high` on the software query despite
   being IT-services under the same weak code as the now-correctly-
   rejected `Fujitsu`/`Capgemini`, just because its keyword+embedding
   scores happened to be high enough on their own. Whenever a company
   scores above `borderline_high`, the LLM safety net never sees it - the
   cheap stages' confidence is never independently checked.
3. **LLM verdict flips on rerun.** Demonstrated directly in 3.3: the same
   company, query, and prompt produced different verdicts at
   temperature=0 across calls. A single-vote verification result should
   be read as "probably right," not "checked."
4. **Subjective/temporal qualifiers the schema can't verify.** "Fast-growing,"
   "competing with traditional banks" — there's no growth-rate or
   competitive-positioning field, so these get silently treated as
   "fintech companies in Europe." The LLM stage, if asked, may produce a
   confident-sounding rationalization for a "fast-growing" claim it has no
   actual basis to verify.
5. **A weak-signal query can get a confidently-wrong single answer instead
   of an honestly-empty result.** The "E-commerce/Shopify" case in 3.3:
   with no real signal to work from, one run returned zero results (right
   call) and another returned one wrong one with fluent-but-nonsensical
   LLM justification. Zero results looks like a broken system to a user;
   one wrong result looks like a working one. The failure that *looks*
   worse is actually the safer one, which is a genuinely awkward property
   to design a UI around.

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
via a flexible parser that accepts either form. Less obviously: this
dataset's NAICS codes follow the **2022 revision**, not 2017 - e.g.
"Software Publishers" is `513210` here, not the more commonly-referenced
`511210`/`5112`. A taxonomy authored from general NAICS knowledge without
checking the actual codes present in the data will silently mismatch on
any concept that happens to fall in a reclassified sector - see 3.3 for
how much this mattered in practice (`qualifier/taxonomy.py` now documents
each such fix inline, and `naics_audit.py`-style validation - diffing a
taxonomy's prefixes against the dataset's actual code list - is cheap
enough that it should run automatically, not just once by hand).

## What I'd prioritize next

1. Automate the NAICS-prefix audit (3.3) as a startup check or test, not
   a one-off manual script - it caught real, silent bugs affecting 3 of
   12 concepts, and nothing about the system would have surfaced them on
   its own if I hadn't gone looking.
2. A small labelled eval set (even ~5 companies × 12 queries, hand-judged)
   to replace eyeballing with a measurable precision/recall number per
   query, and specifically to measure the LLM stage's net effect (does it
   improve precision more than its own inconsistency costs?) rather than
   assuming it helps.
3. Concurrent/batched LLM verification calls instead of the current
   sequential loop - the biggest remaining latency cost once the candidate
   window is bounded.
4. Widening the taxonomy to more domains, and applying the strong/weak
   NAICS split (3.1) to the concepts that don't have one yet - it was only
   added to the five concepts where over-matching was directly observed.
5. Evaluating whether Reciprocal Rank Fusion improves on the current
   weighted-sum score combination now that there's a decomposable
   `score_breakdown` already logged per company to compare against.

## Sources consulted

- [Two-Stage Retrieval Architecture — EmergentMind](https://www.emergentmind.com/topics/two-stage-retrieval-architecture)
- [LLM Retrieval and Reranking: Two-Stage RAG Guide — LlamaIndex](https://www.llamaindex.ai/blog/using-llms-for-retrieval-and-reranking-23cf2d3a14b6)
- [Rating Roulette: Self-Inconsistency in LLM-As-A-Judge Frameworks](https://arxiv.org/html/2510.27106v1)
- [The Coin Flip Judge? Reliability and Bias in LLM-as-a-Judge Evaluation](https://arxiv.org/pdf/2606.13685)
- [Hybrid Search: BM25, Vector & Reranking Reference 2026 — Digital Applied](https://www.digitalapplied.com/blog/hybrid-search-bm25-vector-reranking-reference-2026)
