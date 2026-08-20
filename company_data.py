import ast
import json


def try_parse_dict(value):
    # some fields in the data come as real dicts, some as strings that look
    # like python dicts (e.g. "{'country_code': 'ro', ...}"), so we try both
    if value is None or isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
    return value


def to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_company(raw):
    address = try_parse_dict(raw.get("address"))
    country_code = None
    if isinstance(address, dict) and isinstance(address.get("country_code"), str):
        country_code = address["country_code"].lower()

    naics = try_parse_dict(raw.get("primary_naics"))
    naics_code = naics.get("code") if isinstance(naics, dict) else None
    naics_label = naics.get("label") if isinstance(naics, dict) else None

    secondary_naics = []
    sec_raw = try_parse_dict(raw.get("secondary_naics"))
    sec_list = sec_raw if isinstance(sec_raw, list) else ([sec_raw] if sec_raw else [])
    for item in sec_list:
        item = try_parse_dict(item)
        if isinstance(item, dict) and item.get("code"):
            secondary_naics.append({"code": str(item["code"]), "label": item.get("label")})

    name = raw.get("operational_name") or raw.get("website") or "Unknown company"

    return {
        "name": name,
        "website": raw.get("website"),
        "year_founded": to_int(raw.get("year_founded")),
        "employee_count": to_int(raw.get("employee_count")),
        "revenue": to_float(raw.get("revenue")),
        "is_public": raw.get("is_public") if isinstance(raw.get("is_public"), bool) else None,
        "description": raw.get("description") or "",
        "business_model": raw.get("business_model") or [],
        "target_markets": raw.get("target_markets") or [],
        "core_offerings": raw.get("core_offerings") or [],
        "country_code": country_code,
        "naics_code": str(naics_code) if naics_code is not None else None,
        "naics_label": naics_label,
        "secondary_naics": secondary_naics,
    }


def text_blob(company):
    # description and core offerings matter most, so repeat them for weight
    parts = [
        company["name"],
        company["description"],
        company["description"],
        " ".join(company["core_offerings"]),
        " ".join(company["core_offerings"]),
        " ".join(company["target_markets"]),
        " ".join(company["business_model"]),
    ]
    if company["naics_label"]:
        parts.append(company["naics_label"])
    for sec in company["secondary_naics"]:
        if sec.get("label"):
            parts.append(sec["label"])
    return " ".join(p for p in parts if p)


def llm_summary(company):
    return "\n".join([
        f"Name: {company['name']}",
        f"Country: {company['country_code'].upper() if company['country_code'] else 'unknown'}",
        f"Employees: {company['employee_count'] if company['employee_count'] is not None else 'unknown'}",
        f"Revenue: {company['revenue'] if company['revenue'] is not None else 'unknown'}",
        f"Public: {company['is_public'] if company['is_public'] is not None else 'unknown'}",
        f"Founded: {company['year_founded'] if company['year_founded'] is not None else 'unknown'}",
        f"Primary industry (NAICS): {company['naics_code']} - {company['naics_label']}",
        f"Business model: {', '.join(company['business_model']) or 'unknown'}",
        f"Core offerings: {', '.join(company['core_offerings']) or 'unknown'}",
        f"Target markets: {', '.join(company['target_markets']) or 'unknown'}",
        f"Description: {company['description'][:600]}",
    ])


def load_companies(path):
    seen = set()
    companies = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            key = json.dumps(raw, sort_keys=True)
            if key in seen:
                continue  # dataset has some exact duplicate rows
            seen.add(key)
            companies.append(normalize_company(raw))
    return companies
