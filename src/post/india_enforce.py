import random
import re

from .india_bank import IndiaBank
from .placeholders import replace_placeholders_text, replace_placeholders_json, fix_common_placeholders

_PH_RE = re.compile(
    r"\[\[(CITY|STATE|COLLEGE|PIN|FULL_NAME|EMAIL|PHONE|LINKEDIN_PROFILE|LINKEDIN_URL|GITHUB_URL|GITHUB_HANDLE|PORTFOLIO_URL|WEBSITE)(\d+)?]]",
    re.I)
RUPEE = "₹"
_COLLEGE_TRIGGER = re.compile(r"\b(IIT|NIT|IIIT|BITS|University|Institute|College)\b", re.I)

_DUMMY_EMAILS = {
    "email", "e-mail", "email@example.com", "your.email@example.com", "name@email.com",
    "mail@example.com", "candidate@email.com"
}
_DUMMY_LINKS = {
    "linkedin.com/in/yourprofile", "https://linkedin.com/in/yourprofile",
    "linkedin.com/in/candidatelinkedin", "https://linkedin.com/in/candidatelinkedin",
    "github.com/candidategithub", "https://github.com/candidategithub"
}

_email_com_fix = re.compile(r'(\.com){2,}$')  # collapse .com.com...


def _normalize_email(s: str, name: str, bank) -> str:
    s = (s or "").strip().lower()
    if not s or s in _DUMMY_EMAILS or "example.com" in s:
        return bank.sample_email(name)

    # keep only first local, last domain; remove spaces; collapse .com.com...
    parts = [p for p in s.replace(" ", "").split("@") if p]
    if len(parts) < 2:
        return bank.sample_email(name)
    local, domain = parts[0], parts[-1]
    # strip weird chars from local, basic domain cleanup
    local = re.sub(r"[^a-z0-9._+-]", "", local)
    domain = _email_com_fix.sub(".com", domain)
    if not local or "." not in domain:
        return bank.sample_email(name)
    return f"{local}@{domain}"


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


# --- common bare-placeholder scrubbers (LLM emits these often) ---
_BARE_CONTACT_PATTERNS = [
    # name
    (re.compile(r"\b(Candidate\s+Name|Your\s+Name|Full\s+Name)\b", flags=re.I), "{name}"),
    # email
    (re.compile(r"\bemail@example\.com\b", flags=re.I), "{email}"),
    # phone masks
    (re.compile(r"\+91[-\s]?X{3,}[-\s]?X{3,}[-\s]?X{3,}", flags=re.I), "{phone}"),
    (re.compile(r"\+91-?XXXXXXXXXX\b", flags=re.I), "{phone}"),
    # LinkedIn/GitHub generic slugs
    (re.compile(r"(https?://)?(www\.)?linkedin\.com/in/(yourprofile|candidatelinkedin|fullname)\b", flags=re.I),
     "{linkedin}"),
    (re.compile(r"(https?://)?github\.com/(yourusername|candidategithub)\b", flags=re.I), "{github}"),
    # also bare hostnames
    (re.compile(r"\blinkedin\.com/in/(yourprofile|candidatelinkedin|fullname)\b", flags=re.I), "{linkedin}"),
]


def _apply_bare_contact_subs(text: str, contacts: dict) -> str:
    if not text:
        return text
    rep = {
        "{name}": contacts.get("name", ""),
        "{email}": contacts.get("email", ""),
        "{phone}": contacts.get("phone", ""),
        "{linkedin}": contacts.get("links", ""),
        "{github}": contacts.get("links", ""),  # if you later add github separately, swap here
    }
    out = text
    for pat, token in _BARE_CONTACT_PATTERNS:
        out = pat.sub(rep[token], out)
    return out


def enforce_india_cv(cv: dict, bank: IndiaBank, senior_hint: str | None = None) -> dict:
    # 1) Resolve placeholders across the entire CV dict
    # 1) Contacts: fill/normalize first
    contacts = cv.get("contacts") or {}
    name = contacts.get("name")
    email = contacts.get("email")
    phone = contacts.get("phone")
    links = contacts.get("links")

    if _is_placeholder_contact(name):  name = None
    if _is_placeholder_contact(email): email = None
    if _is_placeholder_contact(phone): phone = None

    name = name or bank.sample_name()
    email = _normalize_email(email, name, bank)
    phone = phone or bank.sample_phone()
    links = links or bank.sample_linkedin(name)  # default to LinkedIn-like URL

    if isinstance(links,
                  list):  # model may return list; but links must be a single string (your schema), and not a dummy
        links = next((v for v in links if isinstance(v, str) and v.strip()), "") or ""
    elif not isinstance(links, str):
        links = str(links or "").strip()
    if links in _DUMMY_LINKS:
        links = ""

    cv["contacts"] = {"name": name, "email": email, "phone": phone, "links": links}

    # 2) One shared mapping for all placeholders
    phmap = {
        "FULL_NAME1": name, "FULL_NAME": name,
        "EMAIL1": email, "EMAIL": email,
        "PHONE1": phone, "PHONE": phone,
        "LINKEDIN_PROFILE1": links, "LINKEDIN_PROFILE": links,
    }

    # 3) Replace placeholders in JSON first (keeps consistency if JSON contains [[...]])
    cv, phmap = replace_placeholders_json(cv, bank, phmap)

    # 4) Replace placeholders in MD and Text using the SAME mapping
    md, phmap = replace_placeholders_text(cv.get("raw_markdown") or "", bank, phmap)
    tx, _ = replace_placeholders_text(cv.get("raw_text") or md, bank, phmap)

    # 5) Scrub “bare” placeholders (Your Name, email@example.com, +91-XXXXX-XXXXX, etc.)
    md = _apply_bare_contact_subs(md, cv["contacts"])
    tx = _apply_bare_contact_subs(tx, cv["contacts"])

    # 6) Sanitize to Indian context (but keep INR glyph readable)
    md = bank.strip_non_indian_tokens(md)
    tx = bank.strip_non_indian_tokens(tx)

    def _patch_contact_dummies(txt: str) -> str:
        if not txt: return txt
        txt = txt.replace("Candidate Name", name).replace("[Your Name]", name)
        txt = re.sub(r"\bemail@example\.com\b|\byour\.email@example\.com\b", email, txt)
        txt = re.sub(r"\+91[- ]?X{4,}[- ]?X{4,}", phone, txt)  # +91-XXXXX-XXXXX
        for d in list(_DUMMY_LINKS):
            txt = txt.replace(d, links or f"https://linkedin.com/in/{name.lower().replace(' ', '')}")
        return txt

    def _fix_city_state_pairs(text: str) -> str:
        if not text:
            return text
        txt = text

        # Build dynamic maps from BANK (no hardcoding)
        state_to_cities = (getattr(bank, "state_to_cities", {}) or {})
        city_to_state = {}
        for st, cities in state_to_cities.items():
            for c in (cities or []):
                city_to_state[str(c)] = st

        # Optional aliases from YAML (e.g., {"Gurgaon":"Gurugram","Ropar":"Rupnagar"})
        aliases = (getattr(bank, "city_aliases", {}) or {})

        # 1) Normalize alias city mentions to their canonical city name
        for alias, canon in aliases.items():
            # turn "alias, X" into "canon, X" (state fixed in next step)
            txt = re.sub(rf'(?<!\w){re.escape(alias)}\s*,', f'{canon},', txt, flags=re.I)

        # 2) Correct "City, WrongState" → "City, CorrectState" for all known cities
        for city, correct in city_to_state.items():
            txt = re.sub(
                rf'(?<!\w){re.escape(city)}\s*,\s*(?!{re.escape(correct)}\b)[A-Za-z .-]+',
                f'{city}, {correct}',
                txt,
                flags=re.I
            )
        return txt

    def _collapse_college_runs(txt: str) -> str:
        if not txt: 
            return txt

        # find long “A - B - C - …” chains and keep just one item
        def _pick_one(run: str) -> str:
            parts = [p.strip() for p in run.split(" - ") if p.strip()]
            if len(parts) >= 2:
                return bank.sample_college()
            return run

        # replace any long run before/after “Education” or degree line
        txt = re.sub(r"([A-Za-z0-9().,&'/-]+\s(?:University|Institute|College|IIT|NIT|IIIT|BITS)[^#\n]{20,})",
                     lambda m: _pick_one(m.group(1)), txt)
        return txt

    # --- Education fixes: year ranges & institution/city/state ---
    def _fix_edu_years(text: str) -> str:
        """Normalize '20212025' or '2021 2025' or '2021 to 2025' → '2021-2025' in the Education block only."""
        if not text:
            return text
        m = re.search(
            r'(Education:?)([\s\S]+?)(\n(?:Experience|Projects|Skills|Certifications|Work Experience|Professional Experience)\b|$)',
            text, flags=re.I)
        if not m:
            return text
        head, edu, tail = m.group(1), m.group(2), text[m.end(2):]

        def _line_fix(s: str) -> str:
            # "2016 2020" or "20162020" → "2016-2020"
            s = re.sub(r'\b((?:19|20)\d{2})\s*((?:19|20)\d{2})\b', r'\1-\2', s)
            # "2016 to 2020" / "2016 – 2020" / "2016-2020" → "2016-2020"
            s = re.sub(r'\b((?:19|20)\d{2})\s*(?:to|–|-)\s*((?:19|20)\d{2})\b', r'\1-\2', s, flags=re.I)
            return s

        edu_fixed = "\n".join(_line_fix(ln) for ln in edu.splitlines())
        return text[:m.start()] + head + edu_fixed + tail

    def _strip_institute_location(text: str) -> str:
        if not text:
            return text
        m = re.search(
            r'(Education:?)([\s\S]+?)(\n(?:Experience|Projects|Skills|Certifications|Work Experience|Professional Experience)\b|$)',
            text, flags=re.I)
        if not m:
            return text
        head, edu, tail = m.group(1), m.group(2), text[m.end(2):]

        lines = []
        for ln in edu.splitlines():
            # Only touch likely degree/institute lines that also mention years/CGPA or a bar
            if re.search(r'\b(19|20)\d{2}\b|\| |CGPA|GPA', ln, flags=re.I):
                # Pattern: "<… Institute …>, City, State"  → keep "<… Institute …>"
                ln = re.sub(
                    r'(?i)(?P<inst>\b(?:IIT|IIIT|NIT|BITS|University|Institute|College)[^,\n|]*)\s*,\s*[A-Za-z .-]+\s*,\s*[A-Za-z .-]+',
                    r'\g<inst>', ln)
                # If it was only "<inst>, City" (no explicit state), drop the city too
                ln = re.sub(
                    r'(?i)(?P<inst>\b(?:IIT|IIIT|NIT|BITS|University|Institute|College)[^,\n|]*)\s*,\s*[A-Za-z .-]+(?!,)',
                    r'\g<inst>', ln)
            lines.append(ln)
        edu_fixed = "\n".join(lines)
        return text[:m.start()] + head + edu_fixed + tail

    cv["raw_markdown"] = _collapse_college_runs(_fix_city_state_pairs(_fix_edu_years(_strip_institute_location(_patch_contact_dummies(md)))))
    cv["raw_text"] = _collapse_college_runs(_fix_city_state_pairs(_fix_edu_years(_strip_institute_location(_patch_contact_dummies(tx)))))

    # 7) Seniority floor
    if senior_hint:
        years = cv.get("years_experience")
        if not years or float(years) < 5:
            cv["years_experience"] = 5

    for sec in cv.get("sections", []):
        if (sec.get("header") or "").strip().lower() == "summary" and not sec.get("bullets"):
            role = (cv.get("role_claim") or "Software Engineer").strip()
            sen = (cv.get("seniority_claim") or (senior_hint or "")).strip()

            # avoid "Senior Senior ..." if role already includes the seniority word
            if sen and sen.lower() in role.lower():
                sen = ""

            skills_src = cv.get("skills") or []
            skills = [s for s in skills_src if isinstance(s, str) and s and len(s) < 40][:4]
            areas = ", ".join(skills) if skills else "backend systems, APIs, cloud"

            summary = f"{(sen.title() + ' ') if sen else ''}{role} with impact across {areas}."
            sec["bullets"] = [summary]
    secs = cv.get("sections")
    if isinstance(secs, list):
        for s in secs:
            # keep only non-empty bullets
            if isinstance(s.get("bullets"), list):
                s["bullets"] = [b.strip() for b in s["bullets"] if isinstance(b, str) and b.strip()]
            # drop empty evidence key
            if "evidence" in s and (not isinstance(s["evidence"], list) or not s["evidence"]):
                s.pop("evidence", None)
    # Log unresolved placeholders, if any
    if _has_placeholders(cv.get("raw_markdown", "")) or _has_placeholders(cv.get("raw_text", "")):
        with open("logs/raw/_placeholders_unresolved.log", "a", encoding="utf-8") as f:
            f.write(f"CV {cv.get('id', 'unknown')} still has placeholders\n")
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
    lo, hi = bank.sample_salary_lpa((senior_hint or "mid").lower())
    # Make salary_hint a string to satisfy JobJD schema
    jd_json["salary_hint"] = f"INR {lo}–{hi} LPA"

    def _collapse_college_runs(txt: str) -> str:
        if not txt: return txt

        # find long “A - B - C - …” chains and keep just one item
        def _pick_one(run: str) -> str:
            parts = [p.strip() for p in run.split(" - ") if p.strip()]
            if len(parts) >= 2:
                return random.choice(parts)  # or bank.sample_college()
            return run

        # replace any long run before/after “Education” or degree line
        txt = re.sub(r"([A-Za-z0-9().,&'/-]+\s(?:University|Institute|College|IIT|NIT|IIIT|BITS)[^#\n]{20,})",
                     lambda m: _pick_one(m.group(1)), txt)
        return txt

    md = bank.strip_non_indian_tokens(jd.get("raw_text") or "")
    if md:
        # Fix compensation footer placeholder
        if "₹" not in md and jd.get("baseSalary") is None:
            md = re.sub(r"_Compensation.*$", "", md, flags=re.IGNORECASE | re.MULTILINE).strip()
            md += f"\n\n_Compensation shown/paid in INR ({RUPEE}{lo}–{hi} LPA)._"
        # Sweep any leftover sentinels
        md = fix_common_placeholders(md, mapping)

        jd_json["raw_text"] = _collapse_college_runs(md)

    # and ensure must_haves never contain “from <college>”
    if isinstance(jd.get("must_haves"), list):
        jd["must_haves"] = [re.sub(r"\s+from\b.*$", "", s, flags=re.I).strip() if isinstance(s, str) else s
                            for s in jd["must_haves"]]

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
