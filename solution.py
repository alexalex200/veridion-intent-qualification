"""Entry point for the company intent-qualification system.

Usage:
    # Single query, printed to stdout
    python solution.py --query "Logistics companies in Romania"

    # All queries in a file, one JSON result file per query under results/
    python solution.py --queries-file queries.json --output-dir results

Design summary (see WRITEUP.md for full rationale):
    1. Load & normalize company records (qualifier/models.py).
    2. Fit a local TF-IDF (or optional sentence-transformers) index over
       the full corpus once (qualifier/embeddings.py).
    3. For each query: parse it into structured constraints + industry
       concepts (qualifier/query_parser.py), apply cheap hard gates
       (qualifier/filters.py), score survivors by blending NAICS +
       keyword + embedding signals (qualifier/scoring.py), and rank.
    4. Optionally, companies whose score lands in an ambiguous middle
       band get a single verification call to a local Ollama model
       (qualifier/llm_verifier.py) - not the full candidate set, and no
       external API calls. Disable with --no-llm; the pipeline also
       degrades to rule+embedding-only automatically if Ollama isn't
       running.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from qualifier.llm_verifier import OllamaVerifier
from qualifier.pipeline import QualificationPipeline


def load_companies(path: str) -> list:
    """Loads JSONL records and drops exact-duplicate rows. The source
    dataset contains byte-identical duplicate records for some companies
    (same name, website, description) - almost certainly a collection
    artifact rather than distinct entities, and left in they'd let one
    company occupy two ranked slots."""
    seen = set()
    companies = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = json.dumps(record, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            companies.append(record)
    return companies


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "query"


def run_single(
    pipeline: QualificationPipeline, query: str, top_k: int, min_score: float,
    use_llm: bool, borderline_high: float, llm_votes: int,
) -> list:
    results = pipeline.qualify(
        query, top_k=top_k, min_score=min_score, use_llm=use_llm,
        borderline_high=borderline_high, llm_votes=llm_votes,
    )
    return [r.to_dict() for r in results]


def main() -> None:
    parser = argparse.ArgumentParser(description="Company intent-qualification system")
    parser.add_argument("--data", default="data/companies.jsonl", help="Path to companies JSONL file")
    parser.add_argument("--query", help="Run a single query and print results to stdout")
    parser.add_argument("--queries-file", help="JSON file with a list of query strings")
    parser.add_argument("--output-dir", default="results", help="Where batch results are written")
    parser.add_argument("--top-k", type=int, default=25, help="Max companies to return per query")
    parser.add_argument("--min-score", type=float, default=0.12, help="Minimum final_score to qualify")
    parser.add_argument("--no-llm", action="store_true", help="Disable the Ollama verification stage")
    parser.add_argument("--llm-model", default="llama3.2:3b", help="Ollama model tag for verification")
    parser.add_argument("--llm-host", default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument(
        "--borderline-high", type=float, default=0.45,
        help="Scores at/above this skip LLM verification (already confident)",
    )
    parser.add_argument(
        "--llm-votes", type=int, default=1,
        help="Majority-vote over N sampled LLM calls per borderline company "
             "(>1 trades latency for verdict reliability; see WRITEUP.md)",
    )
    args = parser.parse_args()

    if not args.query and not args.queries_file:
        parser.error("Provide --query or --queries-file")

    raw_companies = load_companies(args.data)

    verifier = None
    if not args.no_llm:
        verifier = OllamaVerifier(model=args.llm_model, host=args.llm_host)
        if not verifier.available():
            print(f"[llm] Ollama not reachable at {args.llm_host}; "
                  f"continuing with rule+embedding scoring only.", file=sys.stderr)

    pipeline = QualificationPipeline(raw_companies, llm_verifier=verifier)
    print(f"Loaded {len(raw_companies)} companies from {args.data} "
          f"(embedding backend: {pipeline.embedder.name}, "
          f"llm verification: {'on' if verifier and verifier.available() else 'off'})", file=sys.stderr)

    if args.query:
        results = run_single(
            pipeline, args.query, args.top_k, args.min_score, not args.no_llm,
            args.borderline_high, args.llm_votes,
        )
        print(json.dumps({"query": args.query, "results": results}, indent=2))
        return

    queries = json.loads(Path(args.queries_file).read_text(encoding="utf-8"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for query in queries:
        results = run_single(
            pipeline, query, args.top_k, args.min_score, not args.no_llm,
            args.borderline_high, args.llm_votes,
        )
        out_path = out_dir / f"{slugify(query)}.json"
        out_path.write_text(
            json.dumps({"query": query, "results": results}, indent=2), encoding="utf-8"
        )
        print(f"[{len(results):3d} matches] {query}  ->  {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
