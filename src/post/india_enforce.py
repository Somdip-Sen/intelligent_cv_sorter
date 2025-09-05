import re
from typing import Tuple

from .india_bank import IndiaBank
from .placeholders import replace_placeholders_text, replace_placeholders_json, fix_common_placeholders, \
    canonicalize_contacts

_PH_RE = re.compile(
    r"\[\[(CITY|STATE|COLLEGE|PIN|FULL_NAME|EMAIL|PHONE|LINKEDIN_PROFILE|LINKEDIN_URL|GITHUB_URL|GITHUB_HANDLE|PORTFOLIO_URL|WEBSITE)(\d+)?\]\]",
    re.I)
RUPEE = "₹"


def _has_placeholders(s: str) -> bool:
    return bool(s and _PH_RE.search(s))


def _is_placeholder_contact(v: object) -> bool:
    if v is None: return True
    s = str(v or "").strip().lower()
    return (
            not s or
            s in {"[your name]", "[your email]", "[your phone number]", "your name", "candidate", "candidate name"} or
            s in {"name", "email", "phone"} or
            s.startswith("[") and s.endswith("]")
    )


def _is_placeholder_link(s: str | None) -> bool:
    if not s: return True
    t = s.strip().lower()
    return (
            "yourprofile" in t or
            "[[" in t or "]]" in t or
            t in {"linkedin", "github", "portfolio", "website"} or
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


# --- Salary helpers mapped from YAML ---
def _norm_seniority(s: str | None) -> str:
    s = (s or "mid").lower()
    if any(k in s for k in ("principal", "staff", "lead", "architect", "senior")):
        return "senior"
    if any(k in s for k in ("junior", "entry", "fresher", "grad")):
        return "junior"
    return "mid"


def _salary_lpa_range(bank: IndiaBank, senior_hint: str | None) -> tuple[int, int]:
    """
    Read salary bands from YAML: bank.data['salary_lpa_by_seniority'][junior|mid|senior] -> [lo, hi]
    Fallbacks safely; guarantees lo < hi.
    """
    table = (getattr(bank, "data", {}) or {}).get("salary_lpa_by_seniority", {}) or {}
    key = _norm_seniority(senior_hint)
    band = table.get(key) or table.get("mid") or [10, 20]
    try:
        lo, hi = int(band[0]), int(band[1])
    except Exception:
        lo, hi = 10, 20
    if lo > hi:
        lo, hi = hi, lo
    if hi == lo:
        hi = lo + 1
    return lo, hi


def _strip_college_from_degree(s: str) -> str:
    # Keep "Bachelor's degree in X" and drop any trailing 'from ...'
    if not isinstance(s, str): return s
    # remove super long college enumerations
    s = re.sub(r"\sfrom\s.+$", "", s, flags=re.IGNORECASE)
    return s


def enforce_india_cv(cv: dict, bank: IndiaBank, senior_hint: str | None = None) -> dict:
    # 1) Resolve placeholders across the entire CV dict
    cv, mapping = replace_placeholders_json(cv, bank)

    # 2) Contacts canonicalization (string links, real email/phone/name)
    cv["contacts"] = canonicalize_contacts(cv.get("contacts") or {}, bank, mapping)

    # 3) Fix raw_markdown/raw_text leftovers using the same mapping
    for k in ("raw_markdown", "raw_text"):
        txt = cv.get(k) or ""
        if txt:
            txt, mapping = replace_placeholders_text(txt, bank, mapping)
            txt = fix_common_placeholders(txt, mapping)
            cv[k] = txt

    # 4) Sections fallback if empty or “name-only” bullet
    secs = cv.get("sections") or []

    def _bad_summary(bullets):
        if not bullets: return True
        only = [b.strip() for b in bullets if b and b.strip()]
        return all(re.fullmatch(r"[A-Za-z .'-]{2,40}", b or "") for b in only) and len(only) <= 2

    if not secs or (len(secs) == 1 and secs[0].get("header", "").lower() == "summary" and _bad_summary(
            secs[0].get("bullets", []))):
        claim = cv.get("role_claim") or "Software Professional"
        yrs = cv.get("years_experience")
        pin = f" with {yrs}+ years" if yrs else ""
        bullet = f"{claim}{pin} experienced in delivering impact across modern stacks."
        cv["sections"] = [{"header": "Summary", "bullets": [bullet], "evidence": []}]

    return cv


def enforce_india_jd(jd: dict, bank: IndiaBank, senior_hint: str | None = None) -> dict:
    # 1) Resolve placeholders everywhere with a single mapping
    jd_json, mapping = replace_placeholders_json(jd, bank)

    # 2) Normalize must_haves / nice_to_haves and remove colleges
    for key in ("must_haves", "nice_to_haves", "responsibilities"):
        vals = jd_json.get(key) or []
        clean = []
        for v in vals:
            if isinstance(v, str):
                vv = _strip_college_from_degree(v)
                clean.append(vv.strip())
        jd_json[key] = [x for x in clean if x]

    # 3) Salary hint + compensation line (never null)
    lo, hi = _salary_lpa_range(bank, senior_hint)
    jd_json["salary_hint"] = {"currency": "INR", "min_lpa": lo, "max_lpa": hi}

    md = (jd_json.get("raw_text") or "").strip()
    if md:
        # Fix compensation footer placeholder
        md = re.sub(r"_Compensation.*$", "", md, flags=re.IGNORECASE | re.MULTILINE).strip()
        md += f"\n\n_Compensation shown/paid in INR ({RUPEE}{lo}–{hi} LPA)._"
        # Sweep any leftover sentinels
        md = fix_common_placeholders(md, mapping)
        jd_json["raw_text"] = md

    # 4) Fill null location/company_stage as last resort
    if not jd_json.get("jobLocationType"):
        jd_json["jobLocationType"] = "HYBRID"
    jd_json.setdefault("company_stage", "growth-stage")
    if not jd_json.get("jobLocation"):
        city = bank.sample_city()
        jd_json["jobLocation"] = [{
            "@type": "Place",
            "address": {"@type": "PostalAddress", "addressLocality": city, "addressCountry": "IN"}
        }]

    return jd_json
