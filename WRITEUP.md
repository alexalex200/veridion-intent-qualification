# Writeup

## 3.1 Approach

The pipeline has 4 steps:

1. **Parse the query** (`query_parser.py` + `taxonomy.py`) - pull out things like country, employee count, revenue, founding year, public/private using regex, and figure out which "industry concept" the query is about (logistics, software, pharma, fintech, etc.) using a keyword list I wrote by hand. Each concept also has a list of NAICS code prefixes and some extra words to add to the query before running it through the embedding search (because a query like "supply packaging for a cosmetics brand" doesn't use the same words a packaging company would use about itself - so I add words like "contract packaging", "injection molding" to the query text).

2. **Hard filters** (`matching.py`) - reject a company only if a field is actually known and clearly breaks a rule (e.g. employee_count is 50 but the query wants >1000). If the field is missing, I don't reject it, I just note it as unverified and take a small point off the score later. A lot of company data here has missing fields, so rejecting on missing data would throw away good matches.

3. **Scoring** (`matching.py` + `embeddings.py`) - combine 3 things: does the company's NAICS code match the concept (30%), does its text have the query's keywords (25%), and how similar is it to the query using TF-IDF + cosine similarity (45%). I picked TF-IDF instead of a real embedding model because it needs no downloads and is fast enough for this dataset size.

4. **LLM check** (`llm_check.py`, optional) - for companies whose score is in the middle (not clearly good, not clearly bad), I send a single request to a local Ollama model (llama3.2:3b) asking if the company actually fits the query. This only runs on a small slice of companies per query (top ~30-45 after sorting), not everyone, so it stays fast and free. If Ollama isn't running, this step is skipped and the app still works with just the rules+embeddings.

Basically: cheap checks first, and only spend the slow/LLM step on the companies that are actually ambiguous. This avoids sending every company to an LLM (slow and unnecessary for easy queries like "public software companies") while still getting some LLM reasoning on the hard cases.

## 3.2 Tradeoffs

I went for speed and cost over squeezing out every bit of accuracy. TF-IDF instead of a real embedding model is the main tradeoff - it won't catch things phrased very differently, but it's instant and needs no setup. Same idea with the LLM step: only 1 call per company (not multiple calls averaged together), so it's faster but a bit less consistent, since a small local model doesn't always answer the same way twice for the same input (more on that below).

I also chose to be lenient with missing data (a match with unknown employee count still gets included) instead of throwing it away, because in this dataset a huge chunk of fields are just null (employee_count missing for ~39% of rows, revenue for ~19%).

## 3.3 Error Analysis

Some things I found while checking results:

**Keyword matches without NAICS backup are risky.** For "logistics companies in Romania", companies like an oil refinery and a forklift manufacturer showed up because their description mentions "distribution" or "supply chain" - real words, wrong company. I added a rule: if the query expects a specific industry code and the company's own code doesn't match at all, I cut the keyword score in half instead of trusting it fully. That alone fixed most of these without needing the LLM step.

**The NAICS codes I picked by hand were sometimes wrong.** I checked the software/HR concepts against real data and found "Software Publishers" is coded `513210` in this dataset (NAICS 2022), not the more common `511210`/`5112` I originally used - so those concepts were matching zero companies by NAICS this whole time and running on keywords only. Same thing with the EV battery concept, I had `335911` instead of the real `335910`, a typo. Fixing both changed the actual results a lot - HR software query went from a mix of IT consulting firms and real HR tools to mostly real HR software companies, and the EV battery query's top score went from 0.58 to 0.72 with much better matches.

**The LLM isn't always consistent.** I ran the same company through the same prompt twice and got different answers once (`match: true` one time, `match: false` the next), even with temperature set to 0. So the verification step helps on average but isn't something to fully trust on a single call - a company marked `false` by the LLM is excluded from the results, but that's still a probabilistic call, not a guarantee.

**No signal, no good answer.** For "e-commerce companies using Shopify", there's just nothing in this dataset about which platform a company uses, so the system can only guess from generic e-commerce keywords. Scores stay low and results are basically unreliable for this one - which is at least honest (low confidence) instead of confidently wrong.

## 3.4 Scaling to 100,000 companies

- The gates + TF-IDF part would still work fine, but I'd cache the fitted TF-IDF matrix instead of rebuilding it every run.
- For the LLM step, I already limit it to only check the top ~30-45 candidates after sorting (not every company that scored above the minimum) - that's what keeps it from getting slow as the dataset grows, and it would matter even more at 100k rows.
- I'd index the hard-filter fields (country, employee count, etc.) instead of looping through every company - a simple SQLite table with a WHERE clause would do.
- If a real semantic embedding model was worth the setup cost at that scale, I'd batch-encode everything once instead of using TF-IDF.

## 3.5 Failure Modes

- A query about an industry I didn't add to `taxonomy.py` just falls back to keyword/embedding matching, with no indication that the system "doesn't really understand" that domain.
- A NAICS code that's too broad (covers more than one type of company) can make a wrong company score high enough to skip the LLM check entirely, since the "skip if confident" rule only fires when score is high AND NAICS is a strong match now - but there are probably still cases like this I haven't caught.
- Since the LLM step isn't perfectly consistent, running the same query twice can give slightly different results at the edges.
- Anything the query asks that isn't actually a field in the data ("fast-growing", "competing with banks") just gets ignored quietly instead of flagged as unverifiable.

## Data notes

The dataset had ~26 sets of exact duplicate rows (same name/website/description repeated), so `company_data.py` dedupes on exact match before anything else runs (477 -> 457 companies). Some fields (`address`, `primary_naics`) come as either real JSON objects or as strings that look like Python dicts, so there's a small parser that handles both.

## What I'd do next

- Add a small labeled test set (even just a handful of companies per query, marked right/wrong by hand) so I can actually measure accuracy instead of eyeballing it.
- Run the LLM check with a couple of calls averaged together instead of one, to deal with the inconsistency issue - didn't do this by default since it's slower.
- Add more industry concepts to `taxonomy.py` for queries outside the 12 examples.

## Resources I used

- Read up on how retrieval + reranking pipelines are usually structured (cheap filter first, LLM/expensive step only on a short list) - this matched what I ended up building.
- Looked into how reliable LLMs are as judges/verifiers - found that even at temperature 0 they don't always give the same answer twice, which matched what I saw when testing.
