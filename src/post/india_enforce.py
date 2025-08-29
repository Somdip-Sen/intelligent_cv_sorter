import re
from .india_bank import IndiaBank
from .placeholders import replace_placeholders_text, replace_placeholders_json

_PH_RE = re.compile(r"\[\[(CITY|STATE|COLLEGE|PIN)(\d+)?\]\]")

_TITLE_TOKENS = {"data scientist", "software engineer", "ml engineer", "backend", "frontend",
                 "full stack", "senior", "lead", "principal", "compensation", "benefits", "specialist"}


def _looks_like_title(x: str | None) -> bool:
    if not x: return True
    s = str(x).strip().lower()
    return (len(s) < 3) or (s in _TITLE_TOKENS) or ("engineer" in s) or ("scientist" in s)


def _is_placeholder_contact(v: object) -> bool:
    if v is None: return True
    s = str(v).strip().lower()
    return (
            s in {"[your name]", "[your.email@example.com]", "[your phone number]", "email", "phone", "linkedin"} or
            s.startswith("[[") or s.endswith("]]") or
            ("example.com" in s) or
            ("yourprofile" in s) or
            ("xxxxxxxxxx" in s) or
            ("xxxx" in s) or
            (not "@" in s and any(tok in s for tok in ("email", "mail")))  # emails without '@'
    )


def _normalize_link_string(links) -> str:
    # Schema wants a single string. Prefer LinkedIn if present.
    def norm_one(u: str) -> str:
        u = u.strip()
        if not u: return ""
        if not u.startswith("http"):
            u = "https://" + u
        return u

    if isinstance(links, str):
        return norm_one(links)
    if isinstance(links, list):
        if not links: return ""
        # prefer Linkedin-looking link
        li = next((l for l in links if "linkedin.com" in str(l).lower()), links[0])
        return norm_one(str(li))
    return ""


def _replace_contacts_in_text(text: str, name: str, email: str, phone: str, link: str) -> str:
    if not text: return text
    repls = [
        (r"\bemail(?:\.com)?\b", email),
        (r"\bemail@example\.com\b", email),
        (r"\byour\.email@example\.com\b", email),
        (r"\b\+91[- ]?X{10}\b", phone),
        (r"\bphone\b", phone),
        (r"\blinked?in(?:\.com)?/in/yourprofile\b", link),
        (r"\blinked?in\b", link),
        (r"\[\[FULL_NAME\]\]", name),
        (r"\bYour Name\b", name),
    ]
    out = text
    for pat, val in repls:
        out = re.sub(pat, val, out, flags=re.IGNORECASE)
    return out


def _scrub_summary_bullets(sections: list[dict], contacts: dict) -> list[dict]:
    if not isinstance(sections, list): return []
    name = (contacts.get("name") or "").strip()
    email = (contacts.get("email") or "").strip()
    phone = (contacts.get("phone") or "").strip()
    new = []
    for sec in sections:
        if not isinstance(sec, dict):
            continue
        b = []
        for line in (sec.get("bullets") or []):
            s = str(line).strip()
            if not s: continue
            if s == name or s == email or s == phone:  # drop self-echo
                continue
            b.append(s)
        # if Summary ended up empty, synthesize a single bullet
        if (sec.get("header", "").lower() == "summary") and not b:
            b = [f"{name.split()[0]}: Analytical professional with strengths in data analysis and tooling."]
        new.append({"header": sec.get("header") or "Summary", "bullets": b, "evidence": sec.get("evidence") or []})
    return new


# ---- CV enforcer ----
def enforce_india_cv(cv: dict, bank: IndiaBank, senior_hint: str | None = None) -> dict:
    # 1) placeholder replacement first (cities/college/pin)
    cv, phmap = replace_placeholders_json(cv, bank, _PH_RE=_PH_RE)
    cv["raw_markdown"], phmap = replace_placeholders_text(cv.get("raw_markdown") or "", bank, phmap, _PH_RE=_PH_RE)
    cv["raw_text"], _ = replace_placeholders_text(cv.get("raw_text") or "", bank, phmap, _PH_RE=_PH_RE)

    # 2) contacts (PII normalization)
    c = cv.get("contacts") or {}
    name = c.get("name");
    email = c.get("email");
    phone = c.get("phone");
    links = c.get("links")
    if _is_placeholder_contact(name) or _looks_like_title(name):  name = None
    if _is_placeholder_contact(email): email = None
    if _is_placeholder_contact(phone): phone = None

    name = name or bank.sample_name()
    email = email or bank.sample_email(name)
    phone = phone or bank.sample_phone()
    link = _normalize_link_string(links) or "https://www.linkedin.com/in/" + name.lower().replace(" ", "")
    cv["contacts"] = {"name": name, "email": email, "phone": phone, "links": link}  # <-- string, not list

    # 3) rewrite common contact placeholders in MD/Text
    cv["raw_markdown"] = _replace_contacts_in_text(cv.get("raw_markdown") or "", name, email, phone, link)
    cv["raw_text"] = _replace_contacts_in_text(cv.get("raw_text") or "", name, email, phone, link)

    # 4) sanitize non-Indian tokens (keeps ₹ now)
    cv["raw_markdown"] = bank.strip_non_indian_tokens(cv["raw_markdown"])
    cv["raw_text"] = bank.strip_non_indian_tokens(cv["raw_text"])

    # 5) scrub sections (remove name/email/phone bullets; synthesize if empty)
    cv["sections"] = _scrub_summary_bullets(cv.get("sections") or [], cv["contacts"])

    # 6) seniority floor (optional)
    if senior_hint:
        years = cv.get("years_experience")
        if not years or float(years) < 5:
            cv["years_experience"] = 5

    return cv


# ---- JD enforcer ----
def enforce_india_jd(jd: dict, bank: IndiaBank, senior_hint: str | None = None) -> dict:
    # Placeholders → values
    jd, phmap = replace_placeholders_json(jd, bank, _PH_RE=_PH_RE)
    jd["raw_text"], _ = replace_placeholders_text(jd.get("raw_text") or "", bank, phmap, _PH_RE=_PH_RE)

    # Hard defaults
    jd["salaryCurrency"] = "INR"
    if not jd.get("jobLocationType"): jd["jobLocationType"] = "HYBRID"
    # Ensure location (if missing)
    if not jd.get("jobLocation"):
        city = phmap.get("CITY1") if 'phmap' in locals() else bank.sample_city()
        state = bank.sample_state(city)
        jd["jobLocation"] = [{
            "@type": "Place",
            "address": {"@type": "PostalAddress", "addressLocality": city,
                        "addressRegion": state, "addressCountry": "IN"}
        }]

    # Derive simple 'location' string if missing
    if not jd.get("location"):
        try:
            addr = jd["jobLocation"][0]["address"]
            jd["location"] = f"{addr.get('addressLocality', '')}, {addr.get('addressRegion', '')}, IN".strip(", ")
        except Exception:
            jd["location"] = "India"

    # Company stage default (if you don't have a bank list, set a neutral value)
    if not jd.get("company_stage"):
        jd["company_stage"] = "growth-stage"

    # Remove college tails from must_haves and trailing "from"
    mh = [s for s in (jd.get("must_haves") or []) if s]
    clean = []
    for s in jd.get("must_haves", []) or []:
        s = re.sub(r"\s+from\s+[^.;\n]+$", "", s).strip()
        if s:
            clean.append(s)
    jd["must_haves"] = clean

    # Ensure salary hint + keep ₹ in MD
    lo, hi = bank.sample_salary_lpa((senior_hint or "mid").lower())
    jd["salary_hint"] = jd.get("salary_hint") or f"{lo}-{hi} LPA"

    md = bank.strip_non_indian_tokens(jd.get("raw_text") or "")
    if "₹" not in md and "LPA" not in md:
        md += f"\n\nCompensation: ₹{lo}–₹{hi} LPA (INR)."
    jd["raw_text"] = md

    return jd
