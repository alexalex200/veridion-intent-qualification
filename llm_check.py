import json
import urllib.request

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"


def ollama_available(host=OLLAMA_HOST):
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def build_prompt(query, company_text):
    # telling the model to ignore vocabulary overlap and focus on the
    # company's actual business made it a lot stricter (it used to say
    # "match" for basically anything that mentioned the query's words).
    # also had to explicitly say numeric stuff like employee count is
    # already checked elsewhere, otherwise it would make up an answer
    # for fields that were actually "unknown" in the data
    return (
        "You are a strict B2B analyst qualifying ONE company against a search "
        "query for a company database.\n\n"
        f"Query: {query}\n\n"
        f"Company profile:\n{company_text}\n\n"
        "Task: decide if this company's actual BUSINESS/INDUSTRY/ROLE is what "
        "the query is asking for - not a company that merely uses, sells to, or "
        "is loosely associated with that space. Be strict: a company whose core "
        "business is something else does NOT match just because the query's "
        "vocabulary appears somewhere in its description.\n\n"
        "Important: any numeric or factual filters in the query (employee count, "
        "revenue, founding year, public/private, country) have ALREADY been "
        "checked separately before this company reached you - do not use them to "
        "justify your decision, and never assume a value marked 'unknown' in the "
        "profile satisfies the query. Base your decision ONLY on whether the "
        "company's business itself matches the query's industry/role/product "
        "intent.\n\n"
        "First think in a short \"reasoning\" field (1-2 sentences) about the "
        "business/industry/role fit only, then decide.\n"
        'Respond with ONLY compact JSON in this exact shape: '
        '{"reasoning": "<1-2 sentences>", "match": true or false}'
    )


def check_company(query, company_text, model=OLLAMA_MODEL, host=OLLAMA_HOST, timeout=30):
    payload = json.dumps({
        "model": model,
        "prompt": build_prompt(query, company_text),
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{host}/api/generate", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        result = json.loads(body["response"])
        reason = result.get("reasoning", result.get("reason", ""))
        return {"match": bool(result.get("match")), "reason": reason}
    except Exception as e:
        print(f"llm check failed for a company: {e}")
        return {"match": None, "reason": str(e)}
