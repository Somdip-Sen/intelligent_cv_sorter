import re
from .india_bank import IndiaBank
from .placeholders import replace_placeholders_text, replace_placeholders_json

_PH_RE = re.compile(r"\[\[(CITY|STATE|COLLEGE|PIN|FULL_NAME|EMAIL|PHONE|LINKEDIN_PROFILE|LINKEDIN_URL|GITHUB_URL|GITHUB_HANDLE|PORTFOLIO_URL|WEBSITE)(\d+)?\]\]", re.I)

def _has_placeholders(s: str) -> bool:
    return bool(s and _PH_RE.search(s))

def _is_placeholder_contact(v: object) -> bool:
    if v is None: return True
    s = str(v or "").strip().lower()
    return (
        not s or
        s in {"[your name]","[your email]","[your phone number]","your name","candidate","candidate name"} or
        s in {"name","email","phone"} or
        s.startswith("[") and s.endswith("]")
    )

def _is_placeholder_link(s: str | None) -> bool:
    if not s: return True
    t = s.strip().lower()
    return (
        "yourprofile" in t or
        "[[" in t or "]]" in t or
        t in {"linkedin","github","portfolio","website"} or
        t.endswith("@example.com")
    )

def _normalize_link(s: str | None) -> str:
    if not s: return ""
    s = s.strip()
    # collapse double schemes
    s = re.sub(r"https?://https?://", "https://", s)
    # add scheme if missing and looks like a domain path
    if re.match(r"^([a-z0-9-]+\.)+[a-z]{2,}(/.*)?$", s, re.I):
        s = "https://" + s
    return s

def enforce_india_cv(cv: dict, bank: IndiaBank, senior_hint: str | None = None) -> dict:
    # 0) CONTACTS — fill if missing or placeholderish
    contacts = cv.get("contacts") or {}
    name  = contacts.get("name")
    email = contacts.get("email")
    phone = contacts.get("phone")
    links = contacts.get("links")

    if _is_placeholder_contact(name):  name = None
    if _is_placeholder_contact(email): email = None
    if _is_placeholder_contact(phone): phone = None

    # links can be string / list / dict / None → normalize to one string
    link_str = ""
    if isinstance(links, str):
        link_str = links
    elif isinstance(links, list):
        for v in links:
            if isinstance(v, str) and v.strip():
                link_str = v; break
    elif isinstance(links, dict):
        for k in ("linkedin","github","portfolio","url","site"):
            v = links.get(k)
            if isinstance(v, str) and v.strip():
                link_str = v; break

    if _is_placeholder_link(link_str):
        # if we already know (seeded) FULL_NAME, build linkedin from it, else just sample
        link_str = bank.sample_linkedin(name or bank.sample_name())

    # finally materialize
    name  = name  or bank.sample_name()
    email = email or bank.sample_email(name)
    phone = phone or bank.sample_phone()
    link_str = _normalize_link(link_str)

    cv["contacts"] = {"name": name, "email": email, "phone": phone, "links": link_str}

    # 1) Build a seed map so placeholders across JSON/MD use the SAME values as contacts
    seed_map = {
        "FULL_NAME1": name,
        "EMAIL1": email,
        "PHONE1": phone,
        "LINKEDIN_PROFILE1": link_str,
        "LINKEDIN_URL1": link_str,
        # handy defaults if someone uses bare CITY/STATE without index
        "CITY1": None, "STATE1": None, "PIN1": None,
    }

    # 2) Replace placeholders in sections/raw fields
    sections, seed_map = replace_placeholders_json(cv.get("sections") or [], bank, seed_map, _PH_RE=_PH_RE)
    md, seed_map       = replace_placeholders_text(cv.get("raw_markdown") or "", bank, seed_map, _PH_RE=_PH_RE)
    tx, seed_map       = replace_placeholders_text(cv.get("raw_text") or md, bank, seed_map, _PH_RE=_PH_RE)

    # sanitize to Indian context (ASCII-safe; we phrase compensation without ₹)
    md = bank.strip_non_indian_tokens(md)
    tx = bank.strip_non_indian_tokens(tx)

    # Drop silly bullets that equal the name
    for sec in sections:
        if isinstance(sec, dict) and isinstance(sec.get("bullets"), list):
            sec["bullets"] = [b for b in sec["bullets"] if isinstance(b, str) and b.strip() and b.strip() != name]

    cv["sections"] = sections
    cv["raw_markdown"] = md
    cv["raw_text"] = tx

    # 3) Seniority floor (optional)
    if senior_hint:
        years = cv.get("years_experience")
        try:
            if not years or float(years) < 5:
                cv["years_experience"] = 5
        except Exception:
            cv["years_experience"] = 5

    # Log if anything slips through
    if _has_placeholders(cv.get("raw_markdown","")) or _has_placeholders(cv.get("raw_text","")):
        with open("logs/raw/_placeholders_unresolved.log","a",encoding="utf-8") as f:
            f.write(f"CV {cv.get('id','unknown')} still has placeholders\n")

    return cv


def enforce_india_jd(jd: dict, bank: IndiaBank, senior_hint: str | None = None) -> dict:
    # defaults
    jd["salaryCurrency"] = "INR"
    if not jd.get("jobLocationType"):
        jd["jobLocationType"] = "HYBRID"

    # location (handle None / wrong types)
    jl = jd.get("jobLocation")
    if not jl or not isinstance(jl, list):
        city = bank.sample_city()
        state = bank.sample_state(city)
        jd["jobLocation"] = [{
            "@type": "Place",
            "address": {"@type": "PostalAddress", "addressLocality": city, "addressRegion": state, "addressCountry": "IN"}
        }]

    # unify markdown, then placeholder pass
    md0 = jd.get("raw_text") or ""
    md0 = bank.strip_non_indian_tokens(md0)
    md1, ph = replace_placeholders_text(md0, bank, {}, _PH_RE=_PH_RE)
    md1 = bank.strip_non_indian_tokens(md1)  # keep ASCII-safe

    # ensure salary hint (avoid ₹ since ASCII)
    if ("inr" not in md1.lower()) and (jd.get("baseSalary") is None):
        lo, hi = bank.sample_salary_lpa((senior_hint or "mid").lower())
        md1 += f"\n\nCompensation: INR {lo}-{hi} LPA."

    jd["raw_text"] = md1
    jd["salary_hint"] = jd.get("salary_hint") or f"INR {lo}-{hi} LPA" if "lo" in locals() else None

    # Cleanup: remove “ from …” in must_haves
    mh = jd.get("must_haves") or []
    cleaned = []
    for item in mh:
        if not isinstance(item, str):
            continue
        s = re.sub(r"\s+from\s+.*$", "", item.strip(), flags=re.I)
        if s and s.lower() != "from":
            cleaned.append(s)
    jd["must_haves"] = cleaned

    if _has_placeholders(jd.get("raw_text","")):
        with open("logs/raw/_placeholders_unresolved.log","a",encoding="utf-8") as f:
            f.write(f"JD {jd.get('id','unknown')} still has placeholders\n")

    return jd