"""
Build ANALYSIS SET.csv from the Google Form extraction file.

Project: Postharvest agricultural waste-derived biochar for immobilization
of potentially toxic elements in contaminated agricultural soils:
Implications for leafy vegetable safety and human health. A systematic review.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

SRC = Path(
    r"C:\Users\msema\OneDrive\Documentos\Manuscripts\NELSON"
    r"\Engineer Nelsons Review  (Responses) - Form Responses.csv"
)
DST = Path(r"C:\Users\msema\OneDrive\Documentos\Manuscripts\NELSON\ANALYSIS SET.csv")
LOG = Path(r"C:\Users\msema\OneDrive\Documentos\Manuscripts\NELSON\_standardize_log.txt")


# ---------------------------------------------------------------------------
# Text hygiene
# ---------------------------------------------------------------------------

SUPER = str.maketrans(
    {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
        "⁻": "-",
        "₊": "+",
        "⁺": "+",
    }
)

MISSING_TOKENS = {
    "",
    "n/r",
    "nr",
    "n.r",
    "n.r.",
    "na",
    "n/a",
    "n.a.",
    "not reported",
    "not reported.",
    "-",
    "--",
    ".",
    "/",
}


def fold_ascii(s: str) -> str:
    repl = {
        "á": "a",
        "à": "a",
        "ã": "a",
        "â": "a",
        "ä": "a",
        "é": "e",
        "è": "e",
        "ê": "e",
        "í": "i",
        "ì": "i",
        "î": "i",
        "ó": "o",
        "ô": "o",
        "õ": "o",
        "ö": "o",
        "ú": "u",
        "ù": "u",
        "ü": "u",
        "ç": "c",
        "ñ": "n",
    }
    s = s.lower()
    for a, b in repl.items():
        s = s.replace(a, b)
    return s


def clean_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value)
    s = s.replace("\u00a0", " ").replace("\u2009", " ").replace("\u202f", " ")
    s = s.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = s.replace("—", ", ").replace("–", "-").replace("−", "-").replace("‐", "-")
    s = s.replace("×", "x").replace("·", ".")
    s = s.replace("≈", "~").replace("∼", "~")
    s = s.replace("π", "pi").replace("Π", "pi")
    s = s.replace("↓", " decreased").replace("↑", " increased")
    s = s.replace("µ", "u").replace("μ", "u")
    s = s.replace("º", " deg").replace("°", " deg")
    s = s.translate(SUPER)
    s = s.replace("kg-1", "/kg").replace("g-1", "/g").replace("ha-1", "/ha")
    s = s.replace("pot-1", "/pot").replace("L-1", "/L").replace("cm-1", "/cm")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([,;:.])", r"\1", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"\bN/R\b", "NR", s, flags=re.I)
    return s


def is_missing(s: str) -> bool:
    return clean_text(s).lower().rstrip(".") in MISSING_TOKENS


def std_missing(s: str) -> str:
    s = clean_text(s)
    if is_missing(s):
        return "NR"
    low = s.lower().rstrip(".")
    if low in {"reported graphically", "graphically reported"}:
        return "Reported graphically"
    return s


def yes_no(s: str) -> str:
    s = clean_text(s)
    if is_missing(s):
        return "NR"
    low = s.lower()
    if low in {"yes", "y"}:
        return "Yes"
    if low in {"no", "n"}:
        return "No"
    return s


def to_float(s: str):
    if s is None or s == "":
        return None
    s = str(s).replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def first_number(s: str):
    m = re.search(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return to_float(m.group(0)) if m else None


def plusminus_mean_sd(s: str):
    s = clean_text(s)
    m = re.search(
        r"(-?\d+(?:\.\d+)?)\s*(?:\+/-|±)\s*(\d+(?:\.\d+)?)",
        s,
    )
    if m:
        return to_float(m.group(1)), to_float(m.group(2))
    return None, None


# ---------------------------------------------------------------------------
# Column rename
# ---------------------------------------------------------------------------

COLMAP = {
    0: "timestamp_raw",
    1: "study_id",
    2: "citation_raw",
    3: "year",
    4: "doi",
    5: "country_raw",
    6: "extraction_id",
    7: "feedstock_category_raw",
    8: "specific_feedstock_raw",
    9: "feedstock_single_or_mixed_raw",
    10: "feedstock_composition_raw",
    11: "co_amendment_raw",
    12: "carbonization_method_raw",
    13: "biochar_modification_raw",
    14: "pyrolysis_temp_raw",
    15: "particle_size_raw",
    16: "biochar_ph_raw",
    17: "surface_area_raw",
    18: "ash_content_raw",
    19: "functional_groups_raw",
    20: "cec_raw",
    21: "experiment_type_raw",
    22: "soil_type_raw",
    23: "contamination_source",
    24: "target_metals_raw",
    25: "initial_metal_concentration_raw",
    26: "soil_ph_raw",
    27: "biochar_application_rate_raw",
    28: "soil_incubation_raw",
    29: "plant_growth_raw",
    30: "experimental_duration_raw",
    31: "leafy_vegetable_species_raw",
    32: "edible_part_raw",
    33: "soil_bioavailability_method_raw",
    34: "change_in_soil_bioavailability_raw",
    35: "plant_metal_conc_control_raw",
    36: "plant_metal_conc_treated_raw",
    37: "change_in_plant_uptake_raw",
    38: "bcf_raw",
    39: "baf_raw",
    40: "tf_raw",
    41: "hq_raw",
    42: "hi_raw",
    43: "cr_raw",
    44: "immobilization_mechanism_raw",
    45: "main_findings_raw",
    46: "randomization_reported",
    47: "replication_reported",
    48: "control_treatment_present",
    49: "biochar_characterization_adequate",
    50: "statistical_analysis_reported",
    51: "overall_risk_of_bias_raw",
    52: "reviewer_initials",
    53: "residence_time_raw",
}


# ---------------------------------------------------------------------------
# Domain maps
# ---------------------------------------------------------------------------

COUNTRY_MAP = {
    "china": "China",
    "pakistan": "Pakistan",
    "iran": "Iran",
    "poland": "Poland",
    "brazil": "Brazil",
    "turkey": "Turkey",
    "turkiye": "Turkey",
    "türkiye": "Turkey",
    "india": "India",
    "vietnam": "Vietnam",
    "south korea": "South Korea",
    "republic of korea (south korea)": "South Korea",
    "republic of korea": "South Korea",
    "austria": "Austria",
    "canada": "Canada",
    "denmark": "Denmark",
    "united kingdom": "United Kingdom",
    "uk": "United Kingdom",
}

FEEDSTOCK_SPECIFIC = {
    "rice husk": "Rice husk",
    "rice hull": "Rice husk",
    "rice straw": "Rice straw",
    "rice straw (oryza sativa)": "Rice straw",
    "rice stem": "Rice stem",
    "wheat straw": "Wheat straw",
    "wheat husk": "Wheat husk",
    "maize straw": "Maize straw",
    "corn straw": "Maize straw",
    "maize stover": "Maize stover",
    "corn stover": "Maize stover",
    "corn stalk": "Maize stalk",
    "maize straw + cow dung": "Maize straw + cow dung",
    "peanut shell": "Peanut shell",
    "pistachio shell": "Pistachio shell",
    "green coconut shell": "Coconut shell",
    "coconut husk": "Coconut husk",
    "sugarcane bagasse": "Sugarcane bagasse",
    "sugarcane filter-cake": "Sugarcane filter cake",
    "sugar beet pulp": "Sugar beet pulp",
    "licorice root pulp": "Licorice root pulp",
    "hazelnut husk": "Hazelnut husk",
    "oil palm bunch (palm bunch, pb)": "Oil palm bunch",
    "camellia oleifera shell": "Camellia oleifera shell",
    "tobacco straw waste": "Tobacco straw",
    "acai seed (euterpe oleracea mart.)": "Acai seed",
    "acai seed": "Acai seed",
    "açaí seed (euterpe oleracea mart.)": "Acai seed",
    "açaí seed": "Acai seed",
    "seed coat residues from cereal and grass seed production": "Cereal and grass seed residues",
    "vegetable waste": "Vegetable waste",
    "vegetable waste + thiourea": "Vegetable waste + thiourea",
    "lemon waste": "Lemon waste",
    "orange peel": "Orange peel",
    "orange shell (orange bagasse)": "Orange bagasse",
    "banana peel waste": "Banana peel",
    "plantain peel": "Plantain peel",
    "pigeon pea stalk": "Pigeon pea stalk",
    "cotton stalks/cotton sticks": "Cotton stalk",
    "wheat straw + orange peel + rice husk": "Wheat straw + orange peel + rice husk",
}

FEEDSTOCK_CATEGORY = {
    "straw": "Straw",
    "husk": "Husk",
    "shell": "Shell",
    "crop residue": "Crop residue",
    "agricultural residue": "Agricultural residue",
    "fruit processing waste": "Fruit processing waste",
    "rice processing waste": "Husk",
    "seed residue": "Seed residue",
    "vegetable processing waste": "Vegetable processing waste",
    "mixed agricultural residues + livestock manure": "Mixed",
    "processing residue (agricultural waste)": "Agricultural residue",
    "pulp (sugar crop residue)": "Pulp",
    "fruit waste (plantain peel)": "Peel",
    "peels": "Peel",
    "crop residue + fruit waste": "Mixed",
    "fruit processing waste, husk": "Mixed",
    "agricultural processing residue": "Agricultural residue",
    "agro-industrial processing residue": "Agricultural residue",
}

FEEDSTOCK_TO_CATEGORY = {
    "Rice husk": "Husk",
    "Rice straw": "Straw",
    "Rice stem": "Straw",
    "Wheat straw": "Straw",
    "Wheat husk": "Husk",
    "Maize straw": "Straw",
    "Maize stover": "Crop residue",
    "Maize stalk": "Crop residue",
    "Maize straw + cow dung": "Mixed",
    "Peanut shell": "Shell",
    "Pistachio shell": "Shell",
    "Coconut shell": "Shell",
    "Coconut husk": "Husk",
    "Sugarcane bagasse": "Crop residue",
    "Sugarcane filter cake": "Agricultural residue",
    "Sugar beet pulp": "Pulp",
    "Licorice root pulp": "Pulp",
    "Hazelnut husk": "Husk",
    "Oil palm bunch": "Crop residue",
    "Camellia oleifera shell": "Shell",
    "Tobacco straw": "Straw",
    "Acai seed": "Seed residue",
    "Cereal and grass seed residues": "Seed residue",
    "Vegetable waste": "Vegetable processing waste",
    "Vegetable waste + thiourea": "Vegetable processing waste",
    "Lemon waste": "Fruit processing waste",
    "Orange peel": "Peel",
    "Orange bagasse": "Fruit processing waste",
    "Banana peel": "Peel",
    "Plantain peel": "Peel",
    "Pigeon pea stalk": "Crop residue",
    "Cotton stalk": "Crop residue",
    "Wheat straw + orange peel + rice husk": "Mixed",
}

CARBONIZATION = [
    (r"slash-and-char|smolder", "Slash-and-char"),
    (r"fast pyrolysis", "Fast pyrolysis"),
    (r"co-pyrolysis", "Co-pyrolysis"),
    (r"slow pyrolysis|oxygen-limited|anoxic|o2-limited|n2|tubular furnace|rotary kiln", "Slow pyrolysis"),
    (r"^pyrolysis", "Pyrolysis"),
]

ATMOSPHERE = [
    (r"\bn2\b|nitrogen", "N2"),
    (r"oxygen-limited|o2-limited|anoxic|oxygen-poor|sealed vessel|limited oxygen", "Oxygen-limited"),
]

MOD_CLASS = [
    (r"n/?zvi|zero-valent iron|ero-valent iron", "nZVI"),
    (r"nano|ball mill", "Particle-size reduced (nano)"),
    (r"nitrogen-doped|\bhnc\b", "N-doped"),
    (r"azotobacter|bacillus|microb", "Microbially loaded"),
    (r"rock phosphate|tsp|ssp|kh2po4|phosphate", "Phosphate-enriched"),
    (r"fe-?mn|kmno4|kmino4", "Fe-Mn modified"),
    (r"fe-?mg|bimetallic", "Fe-Mg modified"),
    (r"fecl3", "FeCl3 modified"),
    (r"fe-loaded|fe\(no3|aqueous co-precipitation|\bfe\b", "Fe modified"),
    (r"physically mixing|composite", "Physical composite"),
]

CO_CLASS = [
    (r"none", "None"),
    (r"bacter|trichoderma|neorhizobium|azotobacter|inocul|psb|pgp", "Microbial"),
    (r"compost", "Organic"),
    (r"humic", "Organic"),
    (r"thiourea", "Chemical"),
    (r"selenite|selenium|se ", "Chemical"),
    (r"fertilizer|n-p2o5|compound fertilizer", "Fertilizer"),
    (r"phosphogypsum|\bssp\b|\btsp\b|superphosphate|dicalcium|rock phosphate", "Phosphate mineral"),
    (r"lime|zeolite|eggshell|alkaline mineral|perlite", "Mineral"),
    (r"fe-?mn|fe\s*\+\s*mg|fe\+ ?mg|fe\(no3|chitosan|edta|fe3o4", "Metal/chemical"),
]

EXPERIMENT_TYPE = [
    (r"lysimeter", "Lysimeter"),
    (r"field", "Field"),
    (r"chamber|growth chamber", "Growth chamber"),
    (r"tunnel", "Pot"),
    (r"greenhouse|pot", "Pot"),
]

SOIL_CLASS = [
    (r"mine|mining|tailing|gold", "Mine-affected"),
    (r"industrially|industrial", "Industrial"),
    (r"paddy|waterlogg", "Paddy"),
    (r"sewage|wastewater", "Wastewater-irrigated"),
    (r"sand \(medium quartz|uncontaminated|botanical garden|autoclaved", "Experimental/other"),
    (r"agricultural|ultisol|alfisol|oxisol|inceptisol|aridisol|luvisol|cambisol|fluvic|calcixerept|ferralsol|loam|sandy|silt|clay|calcareous|red soil", "Agricultural"),
]

EDIBLE = [
    (r"fruit|pod", "Fruit/pods"),
    (r"storage root|radish", "Mixed edible tissues"),
    (r"leav|shoot|aerial|aboveground|above-ground|plant tissue|edible tissue|consumption", "Shoot (leaves)"),
]

RISK_MAP = {
    "very low": "Very low",
    "low": "Low",
    "low risk": "Low",
    "moderate": "Moderate",
    "high": "High",
}

METAL_ORDER = ["Al", "As", "B", "Cd", "Cr", "Cr(VI)", "Cu", "F", "Fe", "Hg", "Mn", "Mo", "Ni", "Pb", "Se", "Zn"]

SPECIES = [
    (r"ipomoea aquatica|water spinach", "Water spinach", "Ipomoea aquatica"),
    (r"basella alba|vine spinach|malabar spinach", "Malabar spinach", "Basella alba"),
    (r"chrysanthemum coronarium|glebionis|crown daisy", "Crown daisy", "Glebionis coronaria"),
    (r"american brauner|fanela|var\.?\s*longifolia|var\.?\s*ramosa|asparagus lettuce|lactuca sativa|(?<![a-z])lettuce(?![a-z])", "Lettuce", "Lactuca sativa"),
    (r"spinacia oleracea|(?<![a-z])spinach(?![a-z])", "Spinach", "Spinacia oleracea"),
    (r"coriandrum sativum|(?<![a-z])coriander(?![a-z])", "Coriander", "Coriandrum sativum"),
    (r"brassica napus|fengyou", "Rapeseed", "Brassica napus"),
    (r"brassica campestris|shanghaiqing", "Shanghaiqing", "Brassica campestris"),
    (r"brassica parachinensis|choi sum", "Choi sum", "Brassica rapa var. parachinensis"),
    (r"crispifolia", "Mustard greens", "Brassica juncea var. crispifolia"),
    (r"broccoli", "Broccoli", "Brassica oleracea var. italica"),
    (r"brassica oleracea var\.?\s*oleracea|(?<!chinese )(?<!cabbage )(?<![a-z])cabbage(?![a-z])(?! mustard)", "Cabbage", "Brassica oleracea"),
    (r"pekinensis|chinese cabbage(?!.*chinensis)", "Chinese cabbage", "Brassica rapa ssp. pekinensis"),
    (r"brassica chinensis|ssp\.?\s*chinensis|pak\s*choi|pakchoi|brassica rapa var\.?\s*chinensis|(?<![a-z])rassica rapa", "Pak choi", "Brassica rapa var. chinensis"),
    (r"cabbage mustard|brassica juncea|(?<![a-z])mustard(?![a-z])", "Mustard", "Brassica juncea"),
    (r"sinapis alba", "White mustard", "Sinapis alba"),
    (r"phaseolus vulgaris|(?<![a-z])bean(?![a-z])", "Common bean", "Phaseolus vulgaris"),
    (r"gynura", "Gynura", "Gynura cusimbua"),
    (r"parsley", "Parsley", "Petroselinum crispum"),
    (r"(?<![a-z])dill(?![a-z])", "Dill", "Anethum graveolens"),
    (r"amaranth", "Amaranth", "Amaranthus spp."),
    (r"ice plant", "Ice plant", "Mesembryanthemum crystallinum"),
]

FG_PATTERNS = [
    ("OH", r"hydroxyl|\b-?oh\b|o-h|phenolic|ar-oh"),
    ("COOH", r"carboxyl|cooh|coo-|o=c-oh|o=c-oh"),
    ("C=O", r"carbonyl|c=o"),
    ("C=C", r"c=c|aromatic"),
    ("C-O", r"c-o(?!oh)|ether|c-o-c"),
    ("C-H", r"c-h|aliphatic|ch2|ch3|\bch\b"),
    ("CO3", r"co3|carbonate|caco3"),
    ("Si-O", r"si-o|sio2|silic|si-oh|si-h"),
    ("Fe-O", r"fe-o|metal-o|\bm-o\b"),
    ("Mn-O", r"mn-o"),
    ("PO4", r"po4|p-o|phosphate"),
    ("N-H", r"n-h|amino|amine"),
    ("C-N", r"c-n"),
    ("pi-pi", r"pi-pi|π-π|π–π"),
]

MECH_PATTERNS = [
    ("Increased soil pH", r"increased soil ph|\bph increase|soil ph|alkalin|acidity"),
    ("Adsorption", r"adsorp|sorption|\bsorb"),
    ("Surface complexation", r"complexation|complexed|chelat"),
    ("Ion exchange", r"ion exchange|cation exchange"),
    ("Precipitation", r"precipit"),
    ("Electrostatic interaction", r"electrostatic"),
    ("Cation-pi interaction", r"cation-?pi|cation.?π|pi-pi|π"),
    ("Phosphate precipitation", r"phosphate|pyromorphite"),
    ("Carbonate precipitation", r"carbonate|cdco3"),
    ("Fe/Mn oxide binding", r"fe-?mn|fe/mn|oxide-bound|fe oxide|mn oxide"),
    ("Cr(VI) reduction", r"cr\(vi\)|reduc(?:ed|tion) of (?:toxic )?cr|nzvi"),
    ("Microbial immobilization", r"microb|bacter|rhizob|inocul|enzyme activity"),
    ("Increased CEC", r"\bcec\b|cation exchange capacity"),
    ("Increased SOM/DOC", r"organic matter|\bsom\b|\bdoc\b|organic carbon"),
    ("Pore filling / surface area", r"poros|pore|surface area|porous"),
    ("Functional group binding", r"functional group|carboxyl|hydroxyl|phenolic|oxygen-containing"),
]

BIOAV_PATTERNS = [
    ("DTPA", r"(?<!ab-)dtpa"),
    ("AB-DTPA", r"ab-dtpa"),
    ("CaCl2", r"cacl2"),
    ("TCLP", r"tclp|usepa 1311|us epa 1311"),
    ("SBET", r"sbet|bioaccessibility"),
    ("BCR sequential extraction", r"\bbcr\b"),
    ("Tessier sequential extraction", r"tessier"),
    ("Wenzel sequential extraction", r"wenzel"),
    ("EDTA", r"edta"),
    ("NH4NO3", r"nh4no3"),
    ("NH4OAc", r"nh4oac"),
    ("NH4Cl", r"nh4cl"),
    ("Water extraction", r"water \(|h2o|water-soluble|water extraction"),
    ("Pore water", r"pore[ -]?water"),
    ("Isotope dilution (E-value)", r"isotope dilution|e-value|\becd"),
    ("Sequential extraction", r"sequential extraction|sequential arsenic|singh et al"),
    ("Acid digestion (total)", r"acid digestion|nitric acid"),
    ("XRD/SEM only", r"\bxrd\b|sem-edx"),
    ("AOAC", r"\baoac\b"),
    ("Alkaline digestion (Cr(VI))", r"alkaline digestion"),
]


# ---------------------------------------------------------------------------
# Field standardizers
# ---------------------------------------------------------------------------

def std_country(s: str) -> str:
    s = clean_text(s)
    if is_missing(s):
        return "NR"
    parts = re.split(r"[;,/]| and ", s)
    out = []
    for p in parts:
        key = clean_text(p).lower()
        out.append(COUNTRY_MAP.get(key, clean_text(p).title() if key else ""))
    out = [x for x in out if x]
    # unique preserve order
    seen = []
    for x in out:
        if x not in seen:
            seen.append(x)
    return "; ".join(seen) if seen else "NR"


def std_citation(s: str, year) -> str:
    s = clean_text(s)
    s = s.rstrip(").").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace(" et al ", " et al. ")
    s = s.replace(" et al.", " et al.")
    s = re.sub(r"et al\.?", "et al.", s)
    s = s.replace(", et al.", " et al.")
    s = re.sub(r"\s+and\s+", " & ", s)
    s = re.sub(r"\s+\(\d{4}\)", "", s)
    s = re.sub(r"\s+\d{4}\.?$", "", s)
    if not s.lower().endswith("et al."):
        s = re.sub(r"\.$", "", s)
    if s.lower().endswith("et al"):
        s += "."
    s = s.replace("Ma & Liu.", "Ma & Liu")
    s = s.replace("Sehar, et al.", "Sehar et al.")
    if re.search(r",|&", s) and "et al." not in s:
        authors = re.split(r"\s*&\s*|\s*,\s*", s)
        authors = [a for a in authors if a and not re.fullmatch(r"\d{4}", a)]
        if len(authors) >= 3:
            s = f"{authors[0]} et al."
        elif len(authors) == 2:
            s = f"{authors[0]} & {authors[1]}"
        elif authors:
            s = authors[0]
    s = clean_text(s)
    y = str(year).strip()
    return f"{s} {y}".strip()


def first_author(citation: str) -> str:
    c = citation.replace(" et al.", "").strip()
    c = re.split(r"\s+\d{4}$", c)[0]
    if "&" in c:
        return clean_text(c.split("&")[0])
    return clean_text(c.split()[0]) if c else "NR"


def std_feedstock(s: str) -> str:
    s = clean_text(s)
    if is_missing(s):
        return "NR"
    return FEEDSTOCK_SPECIFIC.get(fold_ascii(s), s)


def std_feedstock_cat(raw_cat: str, specific: str) -> str:
    if specific in FEEDSTOCK_TO_CATEGORY:
        return FEEDSTOCK_TO_CATEGORY[specific]
    key = clean_text(raw_cat).lower()
    return FEEDSTOCK_CATEGORY.get(key, clean_text(raw_cat).title() if not is_missing(raw_cat) else "NR")


def std_single_mixed(s: str) -> str:
    s = clean_text(s).lower()
    if s.startswith("mix"):
        return "Mixed"
    if s.startswith("single"):
        return "Single"
    return "NR" if is_missing(s) else clean_text(s).title()


def match_first(s: str, rules, default="NR"):
    low = clean_text(s).lower()
    if is_missing(low):
        return default
    for pat, lab in rules:
        if re.search(pat, low):
            return lab
    return default


def match_all(s: str, rules) -> str:
    low = clean_text(s).lower()
    if is_missing(low):
        return "NR"
    hits = []
    for lab, pat in rules:
        if re.search(pat, low, flags=re.I):
            hits.append(lab)
    return "; ".join(hits) if hits else "Other"


def std_doi(s: str) -> str:
    s = clean_text(s)
    if is_missing(s):
        return "NR"
    s = s.replace("https://doi.org/", "").replace("http://doi.org/", "")
    s = s.replace("doi:", "").strip()
    return s


def parse_temp(s: str):
    s = clean_text(s)
    if is_missing(s):
        return None, None, None, "NR"
    s = s.replace(" deg C", "").replace("degC", "")
    nums = [to_float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]
    nums = [n for n in nums if n is not None]
    if not nums:
        return None, None, None, s
    if re.search(r"[-~]", s) and len(nums) >= 2:
        lo, hi = min(nums[0], nums[1]), max(nums[0], nums[1])
        mid = (lo + hi) / 2
        return mid, lo, hi, s
    if s.startswith("<") and nums:
        return None, None, nums[0], s
    return nums[0], nums[0], nums[0], str(int(nums[0]) if nums[0].is_integer() else nums[0])


def parse_ph(s: str):
    s = clean_text(s)
    if is_missing(s):
        return None, None, "NR"
    mean, sd = plusminus_mean_sd(s)
    if mean is not None:
        return mean, sd, f"{mean}" + (f" +/- {sd}" if sd is not None else "")
    nums = [to_float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]
    nums = [n for n in nums if n is not None and 0 < n < 14]
    if not nums:
        return None, None, std_missing(s)
    if "-" in s and len(nums) >= 2:
        mid = (nums[0] + nums[1]) / 2
        return mid, None, f"{nums[0]}-{nums[1]}"
    return nums[0], None, str(nums[0])


def parse_numeric_measure(s: str):
    s = clean_text(s)
    if is_missing(s) or "table s1" in s.lower() or "supplementary" in s.lower():
        return None, None, "NR"
    mean, sd = plusminus_mean_sd(s)
    if mean is not None:
        return mean, sd, f"{mean}" + (f" +/- {sd}" if sd is not None else "")
    n = first_number(s)
    if n is None:
        return None, None, std_missing(s)
    return n, None, str(n)


def parse_cec(s: str):
    s = clean_text(s)
    if is_missing(s):
        return None, None, "NR", "NR"
    reported_for = "soil" if "soil cec" in s.lower() else "biochar"
    mean, sd = plusminus_mean_sd(s)
    val = mean if mean is not None else first_number(s)
    if val is None:
        return None, None, "NR", reported_for
    low = s.lower()
    if "mmol" in low:
        val = val / 10.0
        if sd is not None:
            sd = sd / 10.0
    # meq 100 g and cmol are equivalent
    return val, sd, (f"{val}" + (f" +/- {sd}" if sd is not None else "")), reported_for


def parse_particle(s: str):
    s = clean_text(s)
    if is_missing(s) or "no specific size" in s.lower():
        return "NR", None, "NR"
    low = s.lower()
    if "nano" in low or "<100 nm" in low:
        return "<0.0001 mm (nano)", 0.0001, "Nano"
    if "micrometre" in low or "micrometer" in low:
        return "Micrometer-sized", None, "Micro"
    if "millimetre" in low or "millimeter" in low:
        return "Millimeter-sized", None, "Milli"
    mesh = re.search(r"(\d+)\s*mesh", low)
    mesh_to_mm = {10: 2.0, 20: 0.841, 60: 0.25, 80: 0.177, 100: 0.149}
    if mesh:
        m = int(mesh.group(1))
        mm = mesh_to_mm.get(m)
        return f"{m} mesh" + (f" ({mm} mm)" if mm else ""), mm, "Mesh"
    if re.fullmatch(r"100", s):
        return "100 mesh (0.149 mm)", 0.149, "Mesh"
    # um
    um = re.search(r"(\d+(?:\.\d+)?)\s*um", low)
    if um:
        mm = to_float(um.group(1)) / 1000.0
        return f"{um.group(1)} um", mm, "Sieved"
    # mm values
    mm_m = re.search(r"(<=|<|≤)?\s*(\d+(?:\.\d+)?)\s*mm", low)
    if mm_m:
        n = to_float(mm_m.group(2))
        pref = mm_m.group(1) or ""
        pref = "<" if pref in {"<", "≤", "<="} else pref
        lab = f"{pref}{n} mm" if pref else f"{n} mm"
        klass = "Sieved"
        return lab, n, klass
    n = first_number(s)
    if n is not None and n <= 10:
        return f"{n} mm", n, "Sieved"
    return s, None, "Other"


def parse_rate(s: str):
    s = clean_text(s)
    if is_missing(s):
        return "NR", None, None
    low = s.lower()
    pct = None
    tha = None
    # highest-performing / selected
    m_high = re.search(r"(\d+(?:\.\d+)?)\s*%\s*\(w/w\)\s*was highest", low)
    if m_high:
        pct = to_float(m_high.group(1))
    if pct is None:
        m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:\(w/w\)|w/w)", low)
        if m:
            pct = to_float(m.group(1))
    if pct is None:
        m = re.search(r"(\d+(?:\.\d+)?)\s*g/kg", low)
        if m:
            pct = to_float(m.group(1)) / 10.0
    if pct is None:
        m = re.search(r"(\d+(?:\.\d+)?)\s*%", low)
        if m:
            pct = to_float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:t|mg)\s*/ha", low)
    if m:
        tha = to_float(m.group(1))
        if "mg /ha" in low.replace("  ", " ") or re.search(r"\bmg\s*/ha", low):
            # Mg/ha is tonnes/ha
            tha = to_float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*mg /ha", low)
    if m and tha is None:
        tha = to_float(m.group(1))
    return s, pct, tha


def parse_duration_days(s: str):
    s = clean_text(s)
    if is_missing(s):
        return "NR", None
    low = s.lower()
    if re.search(r"none|no pre-incubation|no separate|immediate|applied before sowing|mixed before", low):
        if re.search(r"\d+", low) is None or re.search(r"none|immediate|no pre-incubation|no separate", low):
            if not re.search(r"\d+\s*(day|week|month|year)", low):
                return s, 0.0
    # explicit days
    days_hits = [to_float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*days?", low)]
    week_hits = [to_float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*weeks?", low)]
    month_hits = [to_float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*months?", low)]
    year_hits = [to_float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*years?", low)]
    vals = []
    vals += days_hits
    vals += [w * 7 for w in week_hits if w is not None]
    vals += [m * 30 for m in month_hits if m is not None]
    vals += [y * 365 for y in year_hits if y is not None]
    if "1 year" in low and not year_hits:
        vals.append(365)
    if re.search(r"1 month", low) and not month_hits:
        vals.append(30)
    vals = [v for v in vals if v is not None]
    if not vals:
        n = first_number(s)
        if n is not None and n < 1000 and re.search(r"day|week|month|year|~", low):
            return s, n
        return s, None
    # if multiple stages summed in text with +, sum; else use max as the focal period
    if low.count("+") >= 1 and len(vals) >= 2 and "harvest" not in low:
        return s, sum(vals[:2]) if "equilibrat" in low or "after" in low else max(vals)
    return s, max(vals)


def std_metals(s: str) -> str:
    s = clean_text(s)
    if is_missing(s):
        return "NR"
    # normalize chromium VI
    s = re.sub(r"cr\s*\(?\s*vi\s*\)?", "Cr(VI)", s, flags=re.I)
    found = []
    # longer tokens first
    for metal in sorted(METAL_ORDER, key=len, reverse=True):
        if metal == "Cr(VI)":
            pat = r"cr\(vi\)"
        else:
            pat = r"\b" + re.escape(metal) + r"\b"
        if re.search(pat, s, flags=re.I):
            found.append(metal)
    # preserve canonical order
    ordered = [m for m in METAL_ORDER if m in found]
    return "; ".join(ordered) if ordered else s


def std_species(s: str):
    s = clean_text(s)
    if is_missing(s):
        return "NR", "NR", "NR", 0
    low = s.lower()
    commons, scient = [], []
    for pat, common, sci in SPECIES:
        if re.search(pat, low):
            if common == "Spinach" and any(x in commons for x in ("Water spinach", "Malabar spinach")):
                continue
            if common == "Mustard" and ("Rapeseed" in commons or "Mustard greens" in commons):
                continue
            if common not in commons:
                commons.append(common)
                scient.append(sci)
    if not commons:
        return s, s, "NR", 1
    labels = [f"{c} ({sci})" for c, sci in zip(commons, scient)]
    return "; ".join(labels), "; ".join(commons), "; ".join(scient), len(commons)


def std_edible(s: str) -> str:
    s = clean_text(s)
    if is_missing(s):
        return "NR"
    return match_first(s, EDIBLE, default=s)


def std_risk(s: str) -> str:
    s = clean_text(s).lower()
    if is_missing(s):
        return "NR"
    return RISK_MAP.get(s, clean_text(s).title())


def std_outcome(s: str) -> str:
    s = clean_text(s)
    if is_missing(s):
        return "NR"
    low = s.lower()
    if "graphically" in low:
        return "Reported graphically"
    if low in {"not reported", "not reported."}:
        return "NR"
    # normalize reduction language
    s = re.sub(r"\bN/?R\b", "NR", s, flags=re.I)
    s = re.sub(r"\beduced\b", "reduced", s)
    s = re.sub(r"\bREDUCED BY\b", "reduced by", s)
    s = re.sub(r"\breduction\b", "reduction", s, flags=re.I)
    s = re.sub(r"\s+", " ", s)
    return s


def reported_flag(s: str) -> str:
    if s in {"NR", ""}:
        return "No"
    if s == "Reported graphically":
        return "Graphical only"
    return "Yes"


def std_reviewer(s: str) -> str:
    s = clean_text(s)
    return s.title() if s else "NR"


def std_timestamp(s: str) -> str:
    s = clean_text(s)
    try:
        dt = pd.to_datetime(s, dayfirst=False)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return s


def std_composition(s: str, specific: str) -> str:
    s = clean_text(s)
    if is_missing(s):
        return specific if specific != "NR" else "NR"
    # drop redundant "biochar" suffix when it is just the feedstock
    s = re.sub(r"\sbiochar$", "", s, flags=re.I)
    mapped = FEEDSTOCK_SPECIFIC.get(fold_ascii(s))
    return mapped if mapped else s


def std_co_amendment(s: str) -> str:
    s = clean_text(s)
    if is_missing(s) or s.lower() == "none":
        return "None"
    s = s.replace(")Bacillus", ") Bacillus")
    s = s.replace("metal-immobilizing bacteria )", "metal-immobilizing bacteria (")
    return s


def std_modification(s: str) -> str:
    s = clean_text(s)
    if is_missing(s) or s.lower() == "none":
        return "None"
    s = s.replace("ero-valent", "zero-valent")
    return s


def fmt_num(n, digits=4):
    if n is None or (isinstance(n, float) and pd.isna(n)):
        return ""
    if abs(n - round(n)) < 1e-9:
        return str(int(round(n)))
    return f"{n:.{digits}f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def main():
    raw = pd.read_csv(SRC, dtype=str, keep_default_na=False, encoding="utf-8")
    assert raw.shape[1] == 54, raw.shape
    raw.columns = [COLMAP[i] for i in range(54)]

    rows = []
    notes = []
    for i, r in raw.iterrows():
        rec = i + 1
        year = clean_text(r["year"])
        citation = std_citation(r["citation_raw"], year)
        specific = std_feedstock(r["specific_feedstock_raw"])
        category = std_feedstock_cat(r["feedstock_category_raw"], specific)
        mixed = std_single_mixed(r["feedstock_single_or_mixed_raw"])
        if specific == "Wheat straw + orange peel + rice husk" or "+" in specific:
            mixed = "Mixed"
        country = std_country(r["country_raw"])
        metals = std_metals(r["target_metals_raw"])
        species_std, species_common, species_sci, species_n = std_species(r["leafy_vegetable_species_raw"])
        temp_mid, temp_lo, temp_hi, temp_rep = parse_temp(r["pyrolysis_temp_raw"])
        ph, ph_sd, ph_rep = parse_ph(r["biochar_ph_raw"])
        sa, sa_sd, sa_rep = parse_numeric_measure(r["surface_area_raw"])
        ash, ash_sd, ash_rep = parse_numeric_measure(r["ash_content_raw"])
        cec, cec_sd, cec_rep, cec_for = parse_cec(r["cec_raw"])
        psz, psz_mm, psz_class = parse_particle(r["particle_size_raw"])
        sph, sph_sd, sph_rep = parse_ph(r["soil_ph_raw"])
        rate_rep, rate_pct, rate_tha = parse_rate(r["biochar_application_rate_raw"])
        inc_rep, inc_d = parse_duration_days(r["soil_incubation_raw"])
        pg_rep, pg_d = parse_duration_days(r["plant_growth_raw"])
        ed_rep, ed_d = parse_duration_days(r["experimental_duration_raw"])
        res_t, res_sd, res_rep = parse_numeric_measure(r["residence_time_raw"])
        # 180 (3 hours) already numeric
        if res_t is None and not is_missing(r["residence_time_raw"]):
            res_t = first_number(clean_text(r["residence_time_raw"]))
            res_rep = fmt_num(res_t) if res_t is not None else "NR"

        carbon = match_first(r["carbonization_method_raw"], CARBONIZATION, "NR")
        atmo = match_first(r["carbonization_method_raw"], ATMOSPHERE, "NR")
        if carbon == "NR" and not is_missing(r["carbonization_method_raw"]):
            carbon = clean_text(r["carbonization_method_raw"])

        mod = std_modification(r["biochar_modification_raw"])
        mod_class = "None" if mod == "None" else match_first(mod, MOD_CLASS, "Other")
        co = std_co_amendment(r["co_amendment_raw"])
        co_class = "None" if co == "None" else match_first(co, CO_CLASS, "Other")

        bcf = std_outcome(r["bcf_raw"])
        baf = std_outcome(r["baf_raw"])
        tf = std_outcome(r["tf_raw"])
        hq = std_outcome(r["hq_raw"])
        hi = std_outcome(r["hi_raw"])
        cr = std_outcome(r["cr_raw"])

        row = {
            "record_id": rec,
            "timestamp": std_timestamp(r["timestamp_raw"]),
            "study_id": int(clean_text(r["study_id"])) if clean_text(r["study_id"]).isdigit() else clean_text(r["study_id"]),
            "extraction_id": clean_text(r["extraction_id"]),
            "citation": citation,
            "first_author": first_author(citation),
            "year": int(year) if year.isdigit() else year,
            "doi": std_doi(r["doi"]),
            "country": country,
            "n_countries": country.count(";") + 1 if country != "NR" else 0,
            "feedstock_category": category,
            "specific_feedstock": specific,
            "feedstock_single_or_mixed": mixed,
            "feedstock_composition": std_composition(r["feedstock_composition_raw"], specific),
            "co_amendment": co,
            "co_amendment_present": "No" if co == "None" else "Yes",
            "co_amendment_class": co_class,
            "carbonization_method": carbon,
            "carbonization_atmosphere": atmo,
            "biochar_modification": mod,
            "modification_present": "No" if mod == "None" else "Yes",
            "modification_class": mod_class,
            "pyrolysis_temp_c": fmt_num(temp_mid),
            "pyrolysis_temp_min_c": fmt_num(temp_lo),
            "pyrolysis_temp_max_c": fmt_num(temp_hi),
            "pyrolysis_temp_reported": temp_rep,
            "residence_time_min": fmt_num(res_t),
            "residence_time_reported": res_rep if res_rep != "NR" or res_t is not None else "NR",
            "particle_size": psz,
            "particle_size_max_mm": fmt_num(psz_mm, 6),
            "particle_size_class": psz_class,
            "biochar_ph": fmt_num(ph, 3),
            "biochar_ph_sd": fmt_num(ph_sd, 3),
            "biochar_ph_reported": ph_rep,
            "surface_area_m2_g": fmt_num(sa, 3),
            "surface_area_sd": fmt_num(sa_sd, 3),
            "surface_area_reported": sa_rep,
            "ash_content_pct": fmt_num(ash, 3),
            "ash_content_sd": fmt_num(ash_sd, 3),
            "ash_content_reported": ash_rep,
            "functional_groups": std_missing(r["functional_groups_raw"]),
            "functional_groups_std": match_all(r["functional_groups_raw"], FG_PATTERNS) if not is_missing(r["functional_groups_raw"]) else "NR",
            "cec_cmolc_kg": fmt_num(cec, 3),
            "cec_sd": fmt_num(cec_sd, 3),
            "cec_reported": cec_rep,
            "cec_reported_for": cec_for if cec_rep != "NR" else "NR",
            "experiment_type": match_first(r["experiment_type_raw"], EXPERIMENT_TYPE, clean_text(r["experiment_type_raw"]) or "NR"),
            "soil_type": clean_text(r["soil_type_raw"]) or "NR",
            "soil_type_class": match_first(r["soil_type_raw"], SOIL_CLASS, "Other"),
            "contamination_source": clean_text(r["contamination_source"]),
            "target_metals": metals,
            "n_metals": 0 if metals == "NR" else metals.count(";") + 1,
            "initial_metal_concentration": std_missing(r["initial_metal_concentration_raw"]),
            "soil_ph": fmt_num(sph, 3),
            "soil_ph_sd": fmt_num(sph_sd, 3),
            "soil_ph_reported": sph_rep,
            "biochar_application_rate": rate_rep,
            "application_rate_pct_ww": fmt_num(rate_pct, 3),
            "application_rate_t_ha": fmt_num(rate_tha, 3),
            "soil_incubation": inc_rep if inc_rep != "NR" else "NR",
            "soil_incubation_days": fmt_num(inc_d),
            "plant_growth_period": pg_rep if pg_rep != "NR" else "NR",
            "plant_growth_days": fmt_num(pg_d),
            "experimental_duration": ed_rep if ed_rep != "NR" else "NR",
            "experimental_duration_days": fmt_num(ed_d),
            "leafy_vegetable_species": species_std,
            "species_common": species_common,
            "species_scientific": species_sci,
            "n_species": species_n,
            "edible_part": std_edible(r["edible_part_raw"]),
            "soil_bioavailability_method": std_missing(r["soil_bioavailability_method_raw"]),
            "bioavailability_method_std": match_all(r["soil_bioavailability_method_raw"], BIOAV_PATTERNS) if not is_missing(r["soil_bioavailability_method_raw"]) else "NR",
            "change_in_soil_bioavailability": std_outcome(r["change_in_soil_bioavailability_raw"]),
            "plant_metal_conc_control": std_outcome(r["plant_metal_conc_control_raw"]),
            "plant_metal_conc_treated": std_outcome(r["plant_metal_conc_treated_raw"]),
            "change_in_plant_metal_uptake": std_outcome(r["change_in_plant_uptake_raw"]),
            "bcf": bcf,
            "baf": baf,
            "tf": tf,
            "hq": hq,
            "hi": hi,
            "cr": cr,
            "bcf_reported": reported_flag(bcf),
            "baf_reported": reported_flag(baf),
            "tf_reported": reported_flag(tf),
            "health_risk_reported": "Yes" if any(reported_flag(x) != "No" for x in (hq, hi, cr)) else "No",
            "immobilization_mechanism": std_missing(r["immobilization_mechanism_raw"]),
            "mechanism_std": match_all(r["immobilization_mechanism_raw"], MECH_PATTERNS) if not is_missing(r["immobilization_mechanism_raw"]) else "NR",
            "main_findings": clean_text(r["main_findings_raw"]) or "NR",
            "randomization_reported": yes_no(r["randomization_reported"]),
            "replication_reported": yes_no(r["replication_reported"]),
            "control_treatment_present": yes_no(r["control_treatment_present"]),
            "biochar_characterization_adequate": yes_no(r["biochar_characterization_adequate"]),
            "statistical_analysis_reported": yes_no(r["statistical_analysis_reported"]),
            "overall_risk_of_bias": std_risk(r["overall_risk_of_bias_raw"]),
            "reviewer_initials": std_reviewer(r["reviewer_initials"]),
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    # sort by study then extraction
    def extract_key(eid):
        parts = str(eid).split("-")
        try:
            return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
        except ValueError:
            return (9999, 0)

    out["_k"] = out["extraction_id"].map(extract_key)
    out = out.sort_values(["_k", "record_id"]).drop(columns="_k").reset_index(drop=True)
    out["record_id"] = range(1, len(out) + 1)

    out.to_csv(DST, index=False, encoding="utf-8-sig")

    codebook = [
        ("record_id", "Integer", "Row number in the analysis set after sorting by extraction_id."),
        ("timestamp", "Datetime", "Form submission time, ISO-like YYYY-MM-DD HH:MM:SS."),
        ("study_id", "Integer", "Study identifier from the extraction form."),
        ("extraction_id", "Text", "Study-treatment identifier (study_id-arm). One row is one extraction."),
        ("citation", "Text", "Harmonized short citation: First author et al. YEAR, or Author1 & Author2 YEAR."),
        ("first_author", "Text", "First-author surname, for grouping."),
        ("year", "Integer", "Publication year."),
        ("doi", "Text", "DOI with prefix and whitespace removed. NR if missing."),
        ("country", "Text", "Country of the experiment. Multi-country values are semicolon-separated."),
        ("n_countries", "Integer", "Number of countries listed."),
        ("feedstock_category", "Category", "Harmonized feedstock class: Straw, Husk, Shell, Crop residue, Peel, Pulp, Seed residue, Fruit processing waste, Vegetable processing waste, Agricultural residue, Mixed."),
        ("specific_feedstock", "Category", "Harmonized feedstock name (e.g., Rice hull -> Rice husk; corn straw -> Maize straw; Açaí seed -> Acai seed)."),
        ("feedstock_single_or_mixed", "Category", "Single or Mixed."),
        ("feedstock_composition", "Text", "Cleaned composition string. Co-applied non-feedstock materials are also in co_amendment."),
        ("co_amendment", "Text", "Co-applied amendment. None if biochar was used alone."),
        ("co_amendment_present", "Yes/No", "Yes if any co-amendment was used."),
        ("co_amendment_class", "Category", "None, Organic, Microbial, Phosphate mineral, Mineral, Fertilizer, Chemical, Metal/chemical."),
        ("carbonization_method", "Category", "Slow pyrolysis, Fast pyrolysis, Co-pyrolysis, Slash-and-char, or NR."),
        ("carbonization_atmosphere", "Category", "N2, Oxygen-limited, or NR if not stated."),
        ("biochar_modification", "Text", "Cleaned modification description. None if unmodified."),
        ("modification_present", "Yes/No", "Yes if the biochar was modified after or during production."),
        ("modification_class", "Category", "None, Phosphate-enriched, Fe modified, Fe-Mn modified, Fe-Mg modified, FeCl3 modified, nZVI, N-doped, Particle-size reduced (nano), Microbially loaded, Physical composite."),
        ("pyrolysis_temp_c", "Number", "Pyrolysis temperature in C. Midpoint if a range was reported."),
        ("pyrolysis_temp_min_c", "Number", "Lower bound of reported temperature range."),
        ("pyrolysis_temp_max_c", "Number", "Upper bound of reported temperature range."),
        ("pyrolysis_temp_reported", "Text", "Original cleaned temperature string."),
        ("residence_time_min", "Number", "Residence time in minutes."),
        ("residence_time_reported", "Text", "Original cleaned residence time."),
        ("particle_size", "Text", "Harmonized particle-size label, with mesh converted to mm where possible."),
        ("particle_size_max_mm", "Number", "Numeric particle size in mm (mesh converted; upper bound if <x mm)."),
        ("particle_size_class", "Category", "Sieved, Mesh, Nano, Micro, Milli, NR, Other."),
        ("biochar_ph", "Number", "Biochar pH. Midpoint if a range was reported."),
        ("biochar_ph_sd", "Number", "Reported SD of biochar pH, if given."),
        ("biochar_ph_reported", "Text", "Cleaned reported biochar pH."),
        ("surface_area_m2_g", "Number", "BET or reported surface area in m2/g."),
        ("surface_area_sd", "Number", "Reported SD of surface area."),
        ("surface_area_reported", "Text", "Cleaned reported surface area. Supplementary-only mentions coded NR."),
        ("ash_content_pct", "Number", "Ash content in percent."),
        ("ash_content_sd", "Number", "Reported SD of ash content."),
        ("ash_content_reported", "Text", "Cleaned reported ash content."),
        ("functional_groups", "Text", "Cleaned FTIR/XPS description. NR if not reported."),
        ("functional_groups_std", "Text", "Semicolon-separated controlled terms (OH, COOH, C=O, C=C, C-O, C-H, CO3, Si-O, Fe-O, Mn-O, PO4, N-H, C-N, pi-pi)."),
        ("cec_cmolc_kg", "Number", "CEC converted to cmolc/kg (mmol/kg divided by 10; meq/100 g treated as equivalent)."),
        ("cec_sd", "Number", "Reported SD of CEC, on the cmolc/kg scale."),
        ("cec_reported", "Text", "Cleaned reported CEC."),
        ("cec_reported_for", "Category", "biochar, soil, or NR. Use to avoid mixing soil CEC with biochar CEC."),
        ("experiment_type", "Category", "Pot, Field, Growth chamber, or Lysimeter."),
        ("soil_type", "Text", "Cleaned original soil description."),
        ("soil_type_class", "Category", "Agricultural, Mine-affected, Industrial, Wastewater-irrigated, Paddy, Experimental/other."),
        ("contamination_source", "Category", "Naturally contaminated, Artificially spiked, or Both."),
        ("target_metals", "Text", "Semicolon-separated metal symbols in a fixed PTE order."),
        ("n_metals", "Integer", "Number of target metals/metalloids."),
        ("initial_metal_concentration", "Text", "Cleaned initial soil metal concentrations. Units left as reported after ASCII normalization (mg/kg)."),
        ("soil_ph", "Number", "Initial soil pH."),
        ("soil_ph_sd", "Number", "Reported SD of soil pH."),
        ("soil_ph_reported", "Text", "Cleaned reported soil pH."),
        ("biochar_application_rate", "Text", "Cleaned original application-rate string."),
        ("application_rate_pct_ww", "Number", "Application rate as % w/w when extractable. g/kg converted as value/10."),
        ("application_rate_t_ha", "Number", "Application rate in t/ha when reported (including Mg/ha)."),
        ("soil_incubation", "Text", "Cleaned original incubation description."),
        ("soil_incubation_days", "Number", "Incubation/equilibration time in days. None/immediate coded 0. Weeks x7, months x30, years x365."),
        ("plant_growth_period", "Text", "Cleaned original plant-growth duration."),
        ("plant_growth_days", "Number", "Plant growth period in days."),
        ("experimental_duration", "Text", "Cleaned original total duration."),
        ("experimental_duration_days", "Number", "Total experimental duration in days."),
        ("leafy_vegetable_species", "Text", "Harmonized Common name (Scientific name); semicolon-separated if multiple."),
        ("species_common", "Text", "Harmonized common names only."),
        ("species_scientific", "Text", "Harmonized scientific names only."),
        ("n_species", "Integer", "Number of vegetable species in the extraction."),
        ("edible_part", "Category", "Shoot (leaves), Fruit/pods, or Mixed edible tissues."),
        ("soil_bioavailability_method", "Text", "Cleaned original bioavailability method."),
        ("bioavailability_method_std", "Text", "Semicolon-separated controlled methods (DTPA, CaCl2, TCLP, BCR, Tessier, EDTA, etc.)."),
        ("change_in_soil_bioavailability", "Text", "Cleaned soil-bioavailability change. NR or Reported graphically if no usable number."),
        ("plant_metal_conc_control", "Text", "Cleaned control plant metal concentration."),
        ("plant_metal_conc_treated", "Text", "Cleaned treated plant metal concentration."),
        ("change_in_plant_metal_uptake", "Text", "Cleaned plant-uptake change."),
        ("bcf", "Text", "Bioconcentration factor, cleaned. NR if not reported."),
        ("baf", "Text", "Bioaccumulation factor, cleaned."),
        ("tf", "Text", "Translocation factor, cleaned."),
        ("hq", "Text", "Hazard quotient, cleaned."),
        ("hi", "Text", "Hazard index, cleaned."),
        ("cr", "Text", "Carcinogenic risk, cleaned."),
        ("bcf_reported", "Category", "Yes, No, or Graphical only."),
        ("baf_reported", "Category", "Yes, No, or Graphical only."),
        ("tf_reported", "Category", "Yes, No, or Graphical only."),
        ("health_risk_reported", "Yes/No", "Yes if HQ, HI, or CR was reported numerically or qualitatively."),
        ("immobilization_mechanism", "Text", "Cleaned original mechanism text."),
        ("mechanism_std", "Text", "Semicolon-separated controlled mechanisms for counting/plotting."),
        ("main_findings", "Text", "Cleaned main-findings sentence."),
        ("randomization_reported", "Yes/No", "Risk-of-bias item."),
        ("replication_reported", "Yes/No", "Risk-of-bias item."),
        ("control_treatment_present", "Yes/No", "Risk-of-bias item."),
        ("biochar_characterization_adequate", "Yes/No/NR", "Risk-of-bias item."),
        ("statistical_analysis_reported", "Yes/No", "Risk-of-bias item."),
        ("overall_risk_of_bias", "Category", "Very low, Low, Moderate, or High."),
        ("reviewer_initials", "Category", "Nelson, Oscar, or Emmy."),
    ]
    cb = pd.DataFrame(codebook, columns=["column", "type", "definition"])
    cb_path = DST.with_name("ANALYSIS SET codebook.csv")
    cb.to_csv(cb_path, index=False, encoding="utf-8-sig")
    print("wrote", cb_path)

    # profile
    lines = [
        f"rows={len(out)} cols={out.shape[1]}",
        f"unique study_id={out['study_id'].nunique()}",
        f"unique extraction_id={out['extraction_id'].nunique()}",
        f"unique citation={out['citation'].nunique()}",
        f"unique doi={out['doi'].nunique()}",
        "",
        "=== KEY CATEGORIES ===",
    ]
    for col in [
        "country",
        "feedstock_category",
        "specific_feedstock",
        "feedstock_single_or_mixed",
        "co_amendment_class",
        "carbonization_method",
        "carbonization_atmosphere",
        "modification_class",
        "experiment_type",
        "soil_type_class",
        "contamination_source",
        "target_metals",
        "edible_part",
        "species_common",
        "overall_risk_of_bias",
        "reviewer_initials",
        "particle_size_class",
        "health_risk_reported",
    ]:
        lines.append(f"\n[{col}]")
        for v, n in out[col].value_counts().items():
            lines.append(f"  {n}\t{v}")

    lines.append("\n=== NUMERIC COVERAGE ===")
    for col in [
        "pyrolysis_temp_c",
        "residence_time_min",
        "biochar_ph",
        "surface_area_m2_g",
        "ash_content_pct",
        "cec_cmolc_kg",
        "soil_ph",
        "application_rate_pct_ww",
        "application_rate_t_ha",
        "soil_incubation_days",
        "plant_growth_days",
        "experimental_duration_days",
        "particle_size_max_mm",
    ]:
        n = (out[col].astype(str).str.strip() != "").sum()
        lines.append(f"  {col}: {n}/100")

    LOG.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", DST)
    print("wrote", LOG)
    print(out.shape)


if __name__ == "__main__":
    main()
