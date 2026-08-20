"""Validates the taxonomy's NAICS prefixes against the NAICS codes that
actually appear in the data.

This exists because of a real bug class found during development: a
taxonomy concept can look reasonable (a plausible NAICS code, a sensible
comment) while matching zero companies, because the dataset uses a
different NAICS revision (2022 vs. 2017) or the code was mistyped. Nothing
about running the pipeline surfaces this - results still look plausible,
scores are still in a normal range, the concept just silently falls back
to keyword/embedding-only scoring. See WRITEUP.md 3.3 for the concrete
cases this caught (software/HR concepts matching zero companies via the
2017-era "5112" instead of this dataset's "513210"; a one-digit typo in
the EV-battery/clean-energy battery code).

Usage:
    python audit_naics.py [--data data/companies.jsonl]

Exits non-zero if any concept's naics_prefixes matches zero codes in the
dataset, so this can be wired into CI or run before shipping a taxonomy
change - not just invoked by hand.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter

from qualifier.taxonomy import CONCEPTS


def load_naics_codes(path: str) -> Counter:
    codes: Counter = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for field in ("primary_naics", "secondary_naics"):
                raw = record.get(field)
                parsed = raw
                if isinstance(raw, str):
                    try:
                        parsed = ast.literal_eval(raw)
                    except (ValueError, SyntaxError):
                        continue
                if isinstance(parsed, dict) and parsed.get("code"):
                    codes[(str(parsed["code"]), parsed.get("label", ""))] += 1
    return codes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/companies.jsonl")
    parser.add_argument("--top", type=int, default=5, help="Show top N matching codes per concept")
    args = parser.parse_args()

    codes = load_naics_codes(args.data)
    print(f"{len(codes)} distinct NAICS codes across {args.data}\n")

    broken = []
    for concept_key, concept in CONCEPTS.items():
        prefixes = concept["naics_prefixes"]
        if not prefixes:
            print(f"[{concept_key}] no naics_prefixes declared (keyword/embedding-only concept) - skipped")
            continue
        matches = [
            (code, label, n) for (code, label), n in codes.items()
            if any(code.startswith(p) for p in prefixes)
        ]
        total = sum(n for _, _, n in matches)
        status = "OK" if total > 0 else "BROKEN - zero matches"
        print(f"[{concept_key}] prefixes={prefixes} -> {total} company-code-hits ({status})")
        for code, label, n in sorted(matches, key=lambda x: -x[2])[: args.top]:
            print(f"    {code:10s} {label:50s} x{n}")
        if total == 0:
            broken.append(concept_key)
        print()

    if broken:
        print(f"FAILED: {len(broken)} concept(s) with zero NAICS coverage: {broken}", file=sys.stderr)
        sys.exit(1)
    print("All concepts have at least one matching NAICS code in the data.")


if __name__ == "__main__":
    main()
