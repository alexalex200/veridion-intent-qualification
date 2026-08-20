import re

from taxonomy import COUNTRY_NAME_TO_CODE, REGION_ALIASES, REGION_GROUPS, CONCEPTS

NUM = r"[\d][\d,\.]*"


def to_number(text, multiplier=None):
    value = float(text.replace(",", ""))
    if not multiplier:
        return value
    m = multiplier.lower().rstrip(".")
    if m in ("k", "thousand"):
        return value * 1_000
    if m in ("m", "million", "mm"):
        return value * 1_000_000
    if m in ("b", "bn", "billion"):
        return value * 1_000_000_000
    return value


def parse_regions(text, parsed):
    codes = set()
    for alias, canonical in REGION_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", text):
            codes |= REGION_GROUPS[canonical]
    for group_name, group_codes in REGION_GROUPS.items():
        if re.search(rf"\b{re.escape(group_name)}\b", text):
            codes |= group_codes
    for name, code in COUNTRY_NAME_TO_CODE.items():
        if re.search(rf"\b{re.escape(name)}\b", text):
            codes.add(code)
    parsed["country_codes"] = codes
    parsed["region_specified"] = bool(codes)


def parse_numbers(text, parsed):
    m = re.search(rf"(more than|over|above|at least|>\s*)\s*({NUM})\+?\s*employees", text)
    if m:
        parsed["employee_min"] = to_number(m.group(2))
    m = re.search(rf"(fewer than|less than|under|below|<\s*)\s*({NUM})\s*employees", text)
    if m:
        parsed["employee_max"] = to_number(m.group(2))
    m = re.search(rf"({NUM})\+\s*employees", text)
    if m and parsed["employee_min"] is None:
        parsed["employee_min"] = to_number(m.group(1))

    m = re.search(
        rf"(?:revenue[s]?\s+)?(?:of\s+)?(more than|over|above|at least|>\s*)\s*\$?\s*({NUM})\s*"
        rf"(million|billion|thousand|m|bn|k)?\s*(?:in\s+)?(?:revenue)?",
        text,
    )
    if m and ("revenue" in text or "$" in text):
        parsed["revenue_min"] = to_number(m.group(2), m.group(3))
    m = re.search(
        rf"(?:revenue[s]?\s+)?(?:of\s+)?(less than|under|below|<\s*)\s*\$?\s*({NUM})\s*"
        rf"(million|billion|thousand|m|bn|k)?\s*(?:in\s+)?(?:revenue)?",
        text,
    )
    if m and ("revenue" in text or "$" in text):
        parsed["revenue_max"] = to_number(m.group(2), m.group(3))

    m = re.search(r"founded\s+after\s+(\d{4})|since\s+(\d{4})", text)
    if m:
        parsed["founded_min"] = int(m.group(1) or m.group(2))
    m = re.search(r"founded\s+before\s+(\d{4})", text)
    if m:
        parsed["founded_max"] = int(m.group(1))

    if re.search(r"\bpublic(ly traded)?\b", text):
        parsed["is_public"] = True
    elif re.search(r"\bprivate(ly held)?\b", text):
        parsed["is_public"] = False


def parse_concepts(text, parsed):
    for concept_key, concept in CONCEPTS.items():
        for alias in concept["aliases"]:
            if alias in text:
                parsed["concepts"].append(concept_key)
                parsed["naics_prefixes"] |= set(concept["naics_prefixes"])
                parsed["strong_naics_prefixes"] |= set(
                    concept.get("strong_naics_prefixes", concept["naics_prefixes"])
                )
                parsed["keywords"] |= set(concept["keywords"])
                parsed["expansion_terms"].extend(concept["expansion_terms"])
                break

    # also keep the query's own words as keywords, so queries outside the
    # taxonomy still get some keyword-overlap signal
    stopwords = {
        "the", "a", "an", "in", "of", "for", "with", "and", "or", "companies",
        "company", "that", "could", "than", "more", "fewer", "less", "over",
        "under", "employees", "revenue", "founded", "after", "before", "public",
        "private", "using", "similar", "platforms", "to",
    }
    words = re.findall(r"[a-z][a-z\-]{2,}", text)
    parsed["keywords"] |= {w for w in words if w not in stopwords}


def parse_query(query):
    parsed = {
        "raw_query": query,
        "country_codes": set(),
        "region_specified": False,
        "is_public": None,
        "employee_min": None,
        "employee_max": None,
        "revenue_min": None,
        "revenue_max": None,
        "founded_min": None,
        "founded_max": None,
        "concepts": [],
        "expansion_terms": [],
        "naics_prefixes": set(),
        "strong_naics_prefixes": set(),
        "keywords": set(),
    }
    text = query.lower()
    parse_regions(text, parsed)
    parse_numbers(text, parsed)
    parse_concepts(text, parsed)
    return parsed


def expanded_text(parsed):
    return parsed["raw_query"] + " " + " ".join(parsed["expansion_terms"])
