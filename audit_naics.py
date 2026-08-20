# quick script to check that each taxonomy concept's naics_prefixes
# actually matches some real codes in the data, so we don't ship a typo
# or an outdated code and silently lose the naics signal for a concept

import ast
import json
import sys
from collections import Counter

from taxonomy import CONCEPTS


def load_naics_codes(path):
    codes = Counter()
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


def main():
    data_path = "data/companies.jsonl"
    if len(sys.argv) > 1:
        data_path = sys.argv[1]

    codes = load_naics_codes(data_path)
    print(f"{len(codes)} distinct NAICS codes across {data_path}\n")

    broken = []
    for concept_key, concept in CONCEPTS.items():
        prefixes = concept["naics_prefixes"]
        if not prefixes:
            print(f"[{concept_key}] no naics_prefixes, skipped")
            continue
        matches = [
            (code, label, n) for (code, label), n in codes.items()
            if any(code.startswith(p) for p in prefixes)
        ]
        total = sum(n for _, _, n in matches)
        status = "ok" if total > 0 else "BROKEN - zero matches"
        print(f"[{concept_key}] {total} matches ({status})")
        for code, label, n in sorted(matches, key=lambda x: -x[2])[:5]:
            print(f"    {code:10s} {label:50s} x{n}")
        if total == 0:
            broken.append(concept_key)
        print()

    if broken:
        print(f"FAILED: no naics coverage for: {broken}")
        sys.exit(1)
    print("all concepts have at least one matching naics code")


if __name__ == "__main__":
    main()
