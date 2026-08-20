# Country names/regions and industry keywords used to understand a query.
# Each concept has aliases (trigger words), naics_prefixes (industry codes
# that count as a match), keywords (for keyword overlap scoring) and
# expansion_terms (extra words added before running the embedding search,
# since a query like "supply packaging for a cosmetics brand" doesn't use
# the same words a packaging company would use to describe itself).

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
        # 513210 = Software Publishers in NAICS 2022 (this dataset uses the new codes,
        # not the older 5112). 5415x is IT consulting, kept but weighted lower below.
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
                            "561910", "323111", "326199"],
        "strong_naics_prefixes": ["3221", "32221", "322211", "326112", "326160",
                                   "327213", "561910", "323111"],
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
        "strong_naics_prefixes": ["513210", "5112"],
        "keywords": ["hr", "human resources", "payroll", "recruiting", "talent",
                      "workforce", "applicant tracking", "benefits administration",
                      "hris"],
        "expansion_terms": ["human resources software", "payroll platform",
                             "recruiting software", "workforce management"],
    },
    "clean_energy": {
        "aliases": ["clean energy", "renewable energy", "green energy"],
        "naics_prefixes": ["2211", "221114", "221115", "335910", "333611"],
        "strong_naics_prefixes": ["2211", "221114", "221115", "335910", "333611"],
        "keywords": ["solar", "wind", "renewable", "clean energy", "battery storage",
                      "green hydrogen", "carbon", "sustainability", "decarbonization"],
        "expansion_terms": ["renewable energy company", "solar power", "wind energy",
                             "energy storage"],
    },
    "fintech": {
        "aliases": ["fintech", "financial technology", "digital bank", "neobank"],
        "naics_prefixes": ["522", "5223", "5224", "523", "5182"],
        "strong_naics_prefixes": ["522", "5223", "5224"],
        "keywords": ["fintech", "digital bank", "neobank", "payments", "lending",
                      "banking-as-a-service", "challenger bank", "financial services"],
        "expansion_terms": ["digital banking", "payments platform", "online lending",
                             "financial technology company"],
    },
    "ecommerce_platform": {
        "aliases": ["e-commerce", "ecommerce", "online store", "shopify"],
        # no reliable naics code for "uses shopify" in this dataset, so we
        # leave this empty and rely on keywords/embedding only
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
        "naics_prefixes": ["335910", "325180", "212220", "212221", "335999",
                            "325998", "336320", "334419"],
        "strong_naics_prefixes": ["335910", "325180", "212220", "212221"],
        "keywords": ["battery cell", "cathode", "anode", "electrolyte", "separator",
                      "battery management system", "lithium-ion", "lithium", "cobalt",
                      "nickel", "gigafactory", "ev components", "battery pack"],
        "expansion_terms": ["battery cell manufacturer", "cathode material supplier",
                             "battery management system", "lithium-ion battery",
                             "battery materials supplier"],
    },
}
