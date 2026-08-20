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
No LLM calls are made anywhere in this pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

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


def run_single(pipeline: QualificationPipeline, query: str, top_k: int, min_score: float) -> list:
    results = pipeline.qualify(query, top_k=top_k, min_score=min_score)
    return [r.to_dict() for r in results]


def main() -> None:
    parser = argparse.ArgumentParser(description="Company intent-qualification system")
    parser.add_argument("--data", default="data/companies.jsonl", help="Path to companies JSONL file")
    parser.add_argument("--query", help="Run a single query and print results to stdout")
    parser.add_argument("--queries-file", help="JSON file with a list of query strings")
    parser.add_argument("--output-dir", default="results", help="Where batch results are written")
    parser.add_argument("--top-k", type=int, default=25, help="Max companies to return per query")
    parser.add_argument("--min-score", type=float, default=0.12, help="Minimum final_score to qualify")
    args = parser.parse_args()

    if not args.query and not args.queries_file:
        parser.error("Provide --query or --queries-file")

    raw_companies = load_companies(args.data)
    pipeline = QualificationPipeline(raw_companies)
    print(f"Loaded {len(raw_companies)} companies from {args.data} "
          f"(embedding backend: {pipeline.embedder.name})", file=sys.stderr)

    if args.query:
        results = run_single(pipeline, args.query, args.top_k, args.min_score)
        print(json.dumps({"query": args.query, "results": results}, indent=2))
        return

    queries = json.loads(Path(args.queries_file).read_text(encoding="utf-8"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for query in queries:
        results = run_single(pipeline, query, args.top_k, args.min_score)
        out_path = out_dir / f"{slugify(query)}.json"
        out_path.write_text(
            json.dumps({"query": query, "results": results}, indent=2), encoding="utf-8"
        )
        print(f"[{len(results):3d} matches] {query}  ->  {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
