"""Hand-authored domain knowledge: country/region gazetteer and an
industry "concept" ontology used to (a) detect what a query is about and
(b) expand it into vocabulary that actually shows up in company profiles.

This is the piece that lets a rule-based system handle queries like
"companies that could supply packaging materials for a cosmetics brand"
without an LLM: the query talks about the *buyer* (cosmetics brands),
but the matching companies' own descriptions talk about the *supplier's*
business (injection molding, corrugated boxes, contract packaging). A
concept entry bridges that gap with expansion_terms distinct from the
query's own words.

Extending to a new domain = adding one entry here. Nothing else in the
pipeline is domain-specific.
"""

from __future__ import annotations

# --- Country / region gazetteer -------------------------------------------------

COUNTRY_NAME_TO_CODE = {
    "romania": "ro", "germany": "de", "france": "fr", "united states": "us",
    "usa": "us", "u.s.": "us", "u.s.a.": "us", "united states of america": "us",
    "america": "us", "switzerland": "ch", "united kingdom": "gb", "uk": "gb",
    "great britain": "gb", "england": "gb", "sweden": "se", "norway": "no",
    "denmark": "dk", "finland": "fi", "iceland": "is", "netherlands": "nl",
    "holland": "nl", "spain": "es", "italy": "it", "ireland": "ie",
    "poland": "pl", "belgium": "be", "austria": "at", "portugal": "pt",
    "luxembourg": "lu", "china": "cn", "india": "in", "canada": "ca",
    "australia": "au", "brazil": "br", "new zealand": "nz", "japan": "jp",
    "indonesia": "id", "singapore": "sg", "south korea": "kr", "korea": "kr",
    "czechia": "cz", "czech republic": "cz", "hungary": "hu", "greece": "gr",
    "slovakia": "sk", "slovenia": "si", "croatia": "hr", "bulgaria": "bg",
    "estonia": "ee", "latvia": "lv", "lithuania": "lt", "mexico": "mx",
}

# Broad regional groupings, expressed as ISO-3166-1 alpha-2 country codes.
REGION_GROUPS = {
    "europe": {
        "de", "fr", "gb", "ch", "se", "no", "dk", "fi", "is", "nl", "es",
        "it", "ie", "pl", "be", "at", "pt", "lu", "ro", "cz", "hu", "gr",
        "sk", "si", "hr", "bg", "ee", "lv", "lt", "mt", "cy",
    },
    "scandinavia": {"se", "no", "dk"},
    "nordics": {"se", "no", "dk", "fi", "is"},
    "north america": {"us", "ca", "mx"},
    "dach": {"de", "at", "ch"},
    "benelux": {"be", "nl", "lu"},
}

REGION_ALIASES = {
    "scandinavian": "scandinavia",
    "nordic": "nordics",
}

# --- Industry concept ontology ---------------------------------------------------
# aliases: phrases whose presence in the query text triggers this concept.
# naics_prefixes: matched against the leading digits of primary/secondary NAICS code.
# keywords: scored via literal overlap against a company's text blob.
# expansion_terms: appended to the query text before embedding, to close the
#   vocabulary gap between how a *need* is phrased and how a *supplier*
#   describes itself.

CONCEPTS = {
    "logistics": {
        "aliases": ["logistic", "freight", "shipping", "warehousing", "supply chain",
                     "distribution", "3pl", "customs broker"],
        "naics_prefixes": ["484", "485", "486", "487", "488", "492", "493"],
        "keywords": ["freight", "logistics", "warehousing", "customs", "forwarding",
                      "transportation", "distribution", "supply chain", "trucking",
                      "shipping", "carrier"],
        "expansion_terms": ["freight forwarding", "warehousing", "customs brokerage",
                             "transportation services", "supply chain management"],
    },
    "software": {
        "aliases": ["software", "saas", "tech company", "it company"],
        # 513210 is "Software Publishers" under NAICS 2022 (this dataset's
        # revision - confirmed against BambooHR/Pandapé's actual codes,
        # which is where "5112"/"511210", the 2017-revision code, would
        # have matched instead and found nothing). Kept "5112" too as a
        # defensive fallback in case a differently-sourced dataset uses
        # the older scheme. The 5415x codes ("Computer Systems Design
        # Services", "Custom Computer Programming Services") are what
        # large IT-services/consulting firms (Fujitsu, Capgemini, Atos,
        # Genpact, CGI...) are classified under too - real, but not what
        # "software company" usually means to a reader. Kept in
        # naics_prefixes for partial credit but not in
        # strong_naics_prefixes, so they don't outrank actual
        # software-product companies.
        "naics_prefixes": ["513210", "5112", "5182", "541511", "541512", "541519"],
        "strong_naics_prefixes": ["513210", "5112"],
        "keywords": ["software", "saas", "platform", "cloud", "application", "app"],
        "expansion_terms": ["software platform", "cloud software", "technology company"],
    },
    "food_beverage": {
        "aliases": ["food and beverage", "food & beverage", "beverage manufacturer",
                     "food manufacturer", "food producer"],
        "naics_prefixes": ["311", "312"],
        "keywords": ["food", "beverage", "manufacturer", "producer", "processing",
                      "brewing", "bottling", "ingredients"],
        "expansion_terms": ["food production", "beverage manufacturing", "food processing"],
    },
    "packaging_supply": {
        "aliases": ["packaging", "packaging materials", "packaging supplier"],
        "naics_prefixes": ["3221", "32221", "322211", "326112", "326160", "327213",
                            "561910", "323111"],
        "keywords": ["packaging", "container", "bottle", "jar", "tube", "closure",
                      "label", "carton", "corrugated", "contract packaging",
                      "co-packing", "injection molding", "blow molding",
                      "primary packaging", "secondary packaging", "printing"],
        "expansion_terms": ["packaging manufacturer", "contract packaging",
                             "bottles and containers", "labels and cartons",
                             "injection molding", "plastic packaging",
                             "glass packaging", "cosmetic packaging supplier"],
    },
    "construction": {
        "aliases": ["construction"],
        "naics_prefixes": ["236", "237", "238"],
        "keywords": ["construction", "contractor", "builder", "civil engineering",
                      "infrastructure", "building"],
        "expansion_terms": ["construction contractor", "building construction",
                             "civil engineering"],
    },
    "pharma": {
        "aliases": ["pharmaceutical", "pharma", "drug maker", "biotech"],
        "naics_prefixes": ["3254", "541714"],
        "keywords": ["pharmaceutical", "drug", "medicine", "biotech", "clinical",
                      "therapeutics", "generics"],
        "expansion_terms": ["pharmaceutical manufacturing", "drug development",
                             "biotechnology"],
    },
    "saas_hr": {
        "aliases": ["hr solutions", "hr software", "human resources", "hr saas",
                     "payroll software"],
        "naics_prefixes": ["513210", "5112", "5415"],
        "strong_naics_prefixes": ["513210", "5112"],  # see "software" entry above
        "keywords": ["hr", "human resources", "payroll", "recruiting", "talent",
                      "workforce", "applicant tracking", "benefits administration",
                      "hris"],
        "expansion_terms": ["human resources software", "payroll platform",
                             "recruiting software", "workforce management"],
    },
    "clean_energy": {
        "aliases": ["clean energy", "renewable energy", "green energy"],
        "naics_prefixes": ["2211", "221114", "221115", "335911", "541690"],
        # 541690 ("Environmental Consulting Services") catches consulting
        # firms that talk about sustainability without generating or
        # storing energy themselves - weak on its own.
        "strong_naics_prefixes": ["2211", "221114", "221115", "335911"],
        "keywords": ["solar", "wind", "renewable", "clean energy", "battery storage",
                      "green hydrogen", "carbon", "sustainability", "decarbonization"],
        "expansion_terms": ["renewable energy company", "solar power", "wind energy",
                             "energy storage"],
    },
    "fintech": {
        "aliases": ["fintech", "financial technology", "digital bank", "neobank"],
        "naics_prefixes": ["522", "5223", "5224", "523", "5182"],
        # 523 (securities/investment) and 5182 (data hosting) are broader
        # and pull in companies with no real banking/payments angle.
        "strong_naics_prefixes": ["522", "5223", "5224"],
        "keywords": ["fintech", "digital bank", "neobank", "payments", "lending",
                      "banking-as-a-service", "challenger bank", "financial services"],
        "expansion_terms": ["digital banking", "payments platform", "online lending",
                             "financial technology company"],
    },
    "ecommerce_platform": {
        "aliases": ["e-commerce", "ecommerce", "online store", "shopify"],
        # Deliberately empty. "454110" (Electronic Shopping and Mail-Order
        # Houses, NAICS 2017) has zero hits in this dataset - audited
        # against all 105 distinct primary/secondary NAICS codes actually
        # present. There is no reliable structured signal for "uses
        # Shopify" in this schema at all: every company that mentions
        # e-commerce/DTC in its description is NAICS-classified under its
        # actual product category instead (Walmart -> Warehouse Clubs and
        # Supercenters, Decathlon -> Sporting Goods Retailers, Forever 21
        # -> Department Stores...). Leaving a non-matching prefix list here
        # would be worse than an empty one: the corroboration-discount rule
        # in scoring.py penalizes keyword-only matches whenever
        # naics_prefixes is non-empty but doesn't match, which would
        # unfairly punish this concept for a signal the schema simply
        # doesn't carry. See WRITEUP.md for the honest read on this query.
        "naics_prefixes": [],
        "keywords": ["e-commerce", "ecommerce", "online store", "shopify",
                      "direct-to-consumer", "dtc", "online marketplace",
                      "digital storefront"],
        "expansion_terms": ["e-commerce platform", "online retail",
                             "direct-to-consumer brand", "digital storefront"],
    },
    "renewable_equipment": {
        "aliases": ["renewable energy equipment", "wind turbine", "solar panel manufacturer"],
        "naics_prefixes": ["333611", "335312", "326199", "221115", "221114"],
        "keywords": ["wind turbine", "solar panel", "pv module", "renewable equipment",
                      "turbine manufacturer", "solar module"],
        "expansion_terms": ["wind turbine manufacturer", "solar panel manufacturer",
                             "renewable energy equipment"],
    },
    "ev_battery_supply_chain": {
        "aliases": ["battery production", "ev battery", "electric vehicle battery",
                     "battery components"],
        # 335910 (Battery Manufacturing) was originally entered as "335911"
        # (a typo/wrong-revision code) and matched nothing - caught by
        # auditing every concept's prefixes against the 105 actual NAICS
        # codes in this dataset. 325180 ("Other Basic Inorganic Chemical
        # Manufacturing") turned out to be the dominant code for the
        # battery-*materials* companies (cathode/lithium/electrolyte
        # suppliers) this query is really asking about - it wasn't in the
        # original list at all despite being exactly the intended target.
        "naics_prefixes": ["335910", "325180", "212220", "212221", "335999",
                            "325998", "336320", "334419"],
        # 336320 (vehicle electrical equipment), 334419 (electronic
        # components), and 335999 (misc electrical equipment) are broad
        # enough to catch companies with no actual battery-materials or
        # battery-manufacturing role.
        "strong_naics_prefixes": ["335910", "325180", "212220", "212221"],
        "keywords": ["battery cell", "cathode", "anode", "electrolyte", "separator",
                      "battery management system", "lithium-ion", "lithium", "cobalt",
                      "nickel", "gigafactory", "ev components", "battery pack"],
        "expansion_terms": ["battery cell manufacturer", "cathode material supplier",
                             "battery management system", "lithium-ion battery",
                             "battery materials supplier"],
    },
}
