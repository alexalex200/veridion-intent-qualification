W_EMBEDDING = 0.45
W_NAICS = 0.30
W_KEYWORD = 0.25
UNVERIFIED_PENALTY = 0.04
NAICS_MISMATCH_KEYWORD_PENALTY = 0.5


def check_gates(company, parsed):
    # only reject when a field is actually known and clearly violates the
    # query - missing data should not disqualify a company
    fails = []
    unverified = []

    if parsed["region_specified"]:
        cc = company["country_code"]
        if cc is None:
            unverified.append("country unknown")
        elif cc not in parsed["country_codes"]:
            fails.append(f"country '{cc}' not in requested region")

    if parsed["employee_min"] is not None or parsed["employee_max"] is not None:
        if company["employee_count"] is None:
            unverified.append("employee_count unknown")
        else:
            if parsed["employee_min"] is not None and company["employee_count"] < parsed["employee_min"]:
                fails.append("employee_count too low")
            if parsed["employee_max"] is not None and company["employee_count"] > parsed["employee_max"]:
                fails.append("employee_count too high")

    if parsed["revenue_min"] is not None or parsed["revenue_max"] is not None:
        if company["revenue"] is None:
            unverified.append("revenue unknown")
        else:
            if parsed["revenue_min"] is not None and company["revenue"] < parsed["revenue_min"]:
                fails.append("revenue too low")
            if parsed["revenue_max"] is not None and company["revenue"] > parsed["revenue_max"]:
                fails.append("revenue too high")

    if parsed["founded_min"] is not None or parsed["founded_max"] is not None:
        if company["year_founded"] is None:
            unverified.append("year_founded unknown")
        else:
            if parsed["founded_min"] is not None and company["year_founded"] < parsed["founded_min"]:
                fails.append("founded too early")
            if parsed["founded_max"] is not None and company["year_founded"] > parsed["founded_max"]:
                fails.append("founded too late")

    if parsed["is_public"] is not None:
        if company["is_public"] is None:
            unverified.append("is_public unknown")
        elif company["is_public"] != parsed["is_public"]:
            fails.append("is_public mismatch")

    return {"passed": len(fails) == 0, "fails": fails, "unverified": unverified}


def naics_score(company, parsed):
    if not parsed["naics_prefixes"]:
        return 0.0

    def tier(code):
        if any(code.startswith(p) for p in parsed["strong_naics_prefixes"]):
            return 1.0
        if any(code.startswith(p) for p in parsed["naics_prefixes"]):
            return 0.5
        return 0.0

    best = 0.0
    if company["naics_code"]:
        best = max(best, tier(company["naics_code"]))
    for sec in company["secondary_naics"]:
        if sec.get("code"):
            best = max(best, tier(sec["code"]) * 0.6)
    return best


def keyword_score(company, parsed, blob):
    keywords = parsed["keywords"]
    if not keywords:
        return 0.0
    blob = blob.lower()
    hits = sum(1 for kw in keywords if kw in blob)
    return min(1.0, hits / max(3, len(keywords) * 0.5))


def score_company(company, parsed, blob, embedding_similarity, unverified_count):
    naics = naics_score(company, parsed)
    keyword = keyword_score(company, parsed, blob)
    embedding = max(0.0, float(embedding_similarity))

    # if the query expects a specific industry and this company's own
    # NAICS code disagrees entirely, a keyword-only match is weaker
    # evidence, so we discount it (e.g. an oil refiner mentioning
    # "distribution" isn't a logistics company just because the word matches)
    if parsed["naics_prefixes"] and naics == 0.0:
        keyword *= NAICS_MISMATCH_KEYWORD_PENALTY

    relevance = W_EMBEDDING * embedding + W_NAICS * naics + W_KEYWORD * keyword
    final = relevance - UNVERIFIED_PENALTY * unverified_count
    final = max(0.0, min(1.0, final))

    return {"embedding": embedding, "naics": naics, "keyword": keyword, "final": final}
