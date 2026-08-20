import argparse
import json
import re
import sys
from pathlib import Path

from company_data import load_companies, text_blob, llm_summary
from embeddings import build_index, score_query
from llm_check import check_company, ollama_available, OLLAMA_MODEL, OLLAMA_HOST
from matching import check_gates, score_company
from query_parser import parse_query, expanded_text

LLM_MATCH_BOOST = 0.15
LLM_REJECT_MULTIPLIER = 0.4


def slugify(text):
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "query"


def result_to_dict(company, scores, gate, parsed, llm_verdict):
    return {
        "operational_name": company["name"],
        "website": company["website"],
        "country": company["country_code"],
        "final_score": round(scores["final"], 4),
        "score_breakdown": {
            "embedding": round(scores["embedding"], 4),
            "naics": round(scores["naics"], 4),
            "keyword": round(scores["keyword"], 4),
        },
        "gate_passed": gate["passed"],
        "hard_fail_reasons": gate["fails"],
        "unverified_fields": gate["unverified"],
        "matched_concepts": parsed["concepts"],
        "llm_verdict": llm_verdict,
    }


def qualify(companies, vectorizer, matrix, query, top_k=25, min_score=0.12,
            use_llm=True, borderline_high=0.45, llm_model=OLLAMA_MODEL, llm_host=OLLAMA_HOST):
    parsed = parse_query(query)
    embedding_scores = score_query(vectorizer, matrix, expanded_text(parsed))

    results = []
    for company, emb_score, blob in zip(companies, embedding_scores, (text_blob(c) for c in companies)):
        gate = check_gates(company, parsed)
        if not gate["passed"]:
            continue
        scores = score_company(company, parsed, blob, emb_score, len(gate["unverified"]))
        if scores["final"] < min_score:
            continue
        results.append({"company": company, "scores": scores, "gate": gate, "llm_verdict": None})

    results.sort(key=lambda r: r["scores"]["final"], reverse=True)
    candidate_window = max(top_k * 3, 30)

    use_llm = use_llm and ollama_available(llm_host)
    if use_llm:
        for r in results[:candidate_window]:
            scores = r["scores"]
            # only skip the LLM check if the score is high AND the industry
            # code actually confirms it - a high score from keywords/embedding
            # alone isn't enough to trust without checking
            if scores["final"] >= borderline_high and scores["naics"] >= 1.0:
                continue
            verdict = check_company(query, llm_summary(r["company"]), model=llm_model, host=llm_host)
            r["llm_verdict"] = verdict
            if verdict["match"] is True:
                scores["final"] = min(1.0, scores["final"] + LLM_MATCH_BOOST)
            elif verdict["match"] is False:
                scores["final"] *= LLM_REJECT_MULTIPLIER
            # match is None means the llm call failed, leave score as is

    # a company the llm explicitly rejected should not stay "qualified"
    # just because its rule-based score was high before the check
    results = [
        r for r in results
        if r["scores"]["final"] >= min_score
        and not (r["llm_verdict"] and r["llm_verdict"]["match"] is False)
    ]
    results.sort(key=lambda r: r["scores"]["final"], reverse=True)
    results = results[:top_k]

    return [
        result_to_dict(r["company"], r["scores"], r["gate"], parsed, r["llm_verdict"])
        for r in results
    ]


def run_batch(companies, vectorizer, matrix, queries, output_dir, top_k, min_score, use_llm,
              borderline_high, llm_model, llm_host):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for query in queries:
        results = qualify(companies, vectorizer, matrix, query, top_k, min_score,
                           use_llm, borderline_high, llm_model, llm_host)
        out_path = out_dir / f"{slugify(query)}.json"
        out_path.write_text(json.dumps({"query": query, "results": results}, indent=2), encoding="utf-8")
        print(f"[{len(results):3d} matches] {query}  ->  {out_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Company intent-qualification system")
    parser.add_argument("--data", default="data/companies.jsonl")
    parser.add_argument("--query", help="Run a single query and print results to stdout")
    parser.add_argument("--queries-file", help="JSON file with a list of query strings")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--min-score", type=float, default=0.12)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--llm-model", default=OLLAMA_MODEL)
    parser.add_argument("--llm-host", default=OLLAMA_HOST)
    parser.add_argument("--borderline-high", type=float, default=0.45)
    args = parser.parse_args()

    if not args.query and not args.queries_file:
        parser.error("Provide --query or --queries-file")

    companies = load_companies(args.data)
    vectorizer, matrix = build_index([text_blob(c) for c in companies])
    use_llm = not args.no_llm
    if use_llm and not ollama_available(args.llm_host):
        print(f"Ollama not reachable at {args.llm_host}, continuing without llm verification", file=sys.stderr)

    print(f"Loaded {len(companies)} companies from {args.data}", file=sys.stderr)

    if args.query:
        results = qualify(companies, vectorizer, matrix, args.query, args.top_k, args.min_score,
                           use_llm, args.borderline_high, args.llm_model, args.llm_host)
        print(json.dumps({"query": args.query, "results": results}, indent=2))
        return

    queries = json.loads(Path(args.queries_file).read_text(encoding="utf-8"))
    run_batch(companies, vectorizer, matrix, queries, args.output_dir, args.top_k, args.min_score,
              use_llm, args.borderline_high, args.llm_model, args.llm_host)


if __name__ == "__main__":
    main()
