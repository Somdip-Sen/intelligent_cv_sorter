import random
import re
from typing import Any, Tuple
from .india_bank import IndiaBank

# Accept a wider set of placeholders (with or without index)
# e.g. [[CITY]], [[CITY1]], [[LINKEDIN_URL]], [[GITHUB_URL]]
_PH_RE = re.compile(
    r"\[\[("
    r"CITY|STATE|COLLEGE|PIN|FULL_NAME|EMAIL|PHONE|"
    r"LINKEDIN_PROFILE|LINKEDIN_URL|LINKEDIN|GITHUB_URL|GITHUB|WEBSITE_URL|URL"
    r")(\d+)?]]",
    flags=re.IGNORECASE
)

# Common dummy/sentinel tokens we should overwrite even if not bracketed
_COMMON_SENTINELS = [
    r"\bCandidate Name\b",
    r"\bFull Name\b",
    r"\bemail@example\.com\b",
    r"\bemail\b",
    r"\+91-?XXXXXXXXXX\b",
    r"\+91-?9876543210\b",
    r"\blinkedin\.com/in/yourprofile\b",
    r"\bcandidatelinkedin\b",
    r"\bgithub\.com/candidategithub\b",
    r"https?://\[\[(?:LINKEDIN|LINKEDIN_URL|GITHUB|GITHUB_URL)\]\]",
    r"\[\[(?:LINKEDIN|LINKEDIN_URL|GITHUB|GITHUB_URL)\]\]"
]
_COMMON_SENTINELS_RE = re.compile("|".join(_COMMON_SENTINELS), flags=re.IGNORECASE)


def _slugify_name(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "", (name or "").lower())
    return s or "profile"


def _add_scheme(url: str) -> str:
    if not url:
        return ""
    if not re.match(r"^https?://", url, flags=re.I):
        return "https://" + url.lstrip("/")
    return url


def normalize_links_str(x) -> str:
    """Normalize links to a single string for CVDoc.contacts.links (Pydantic expects str)."""
    if not x:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, list):
        for v in x:
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""
    if isinstance(x, dict):
        for k in ("linkedin", "github", "portfolio", "url", "site"):
            v = x.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""
    return str(x).strip()


def _key(tag: str, idx: str | None) -> str:
    return f"{tag.upper()}{(idx or '1')}"


def replace_placeholders_text(
        text: str,
        bank: IndiaBank,
        mapping: dict[str, str] | None = None,

) -> Tuple[str, dict[str, str]]:
    """
    Replace placeholders (location + PII) inside a string and return (new_text, mapping).
    Keeps a stable mapping so the same token resolves consistently across JSON/MD.
    """
    if not text:
        return text or "", dict(mapping or {})
    mapping = dict(mapping or {})

    def _k(tag, idx):
        return f"{tag}{(idx or '1')}"

    def _derive_from_city(idx):
        k_city, k_state, k_pin = _k("CITY", idx), _k("STATE", idx), _k("PIN", idx)
        city = mapping.get(k_city)
        if not city: return
        # always align state/pin to CITY to avoid mismatches
        mapping[k_state] = bank.sample_state(city)
        mapping[k_pin] = bank.sample_pin(city, mapping[k_state])

    def _value(tag: str, idx: str | None) -> str:
        key = _k(tag, idx)
        if key in mapping:
            return mapping[key]
        if tag == "CITY":
            mapping[key] = bank.sample_city()
            _derive_from_city(idx)  # <-- force alignment
        elif tag == "STATE":
            city = mapping.get(_k("CITY", idx))
            mapping[key] = bank.sample_state(city)
        elif tag == "PIN":
            city = mapping.get(_k("CITY", idx))
            state = mapping.get(_k("STATE", idx))
            mapping[key] = bank.sample_pin(city, state)
        elif tag == "COLLEGE":
            mapping[key] = bank.sample_college()
        elif tag == "FULL_NAME":
            mapping[key] = bank.sample_name()
        elif tag == "EMAIL":
            nm = mapping.get(_k("FULL_NAME", idx)) or bank.sample_name()
            mapping[key] = bank.sample_email(nm)
        elif tag == "PHONE":
            mapping[key] = bank.sample_phone()
        elif tag in ("LINKEDIN_PROFILE", "LINKEDIN_URL"):
            nm = mapping.get(_k("FULL_NAME", idx)) or bank.sample_name()
            v = bank.sample_linkedin(nm)
            slug = re.sub(r"[^a-z0-9]+", "", nm.lower())
            mapping[key] = v or f"https://linkedin.com/in/{slug}{random.randint(10, 99)}"
        elif tag == "GITHUB_URL":
            nm = mapping.get(_k("FULL_NAME", idx)) or bank.sample_name()
            v = bank.sample_github(nm)
            slug = re.sub(r"[^a-z0-9]+", "", nm.lower())
            mapping[key] = v or f"https://github.com/{slug}{random.randint(10, 99)}"
        else:
            mapping[key] = ""
        return mapping[key]

    def _repl(m: re.Match) -> str:
        tag, idx = m.group(1), m.group(2)
        val = _value(tag, idx)
        # if CITY was just produced, we may have overridden STATE/PIN – fine
        return val

    new_text = _PH_RE.sub(_repl, text)

    # Also erase common sentinels with actual resolved values
    if _COMMON_SENTINELS_RE.search(new_text):
        # Ensure we have a FULL_NAME1/EMAIL1/PHONE1/LINKEDIN_URL1 ready
        _ = _value("FULL_NAME", None)
        _ = _value("EMAIL", None)
        _ = _value("PHONE", None)
        _ = _value("LINKEDIN_URL", None)

        repls = {
            r"\bCandidate Name\b": mapping[_key("FULL_NAME", None)],
            r"\bFull Name\b": mapping[_key("FULL_NAME", None)],
            r"\bemail@example\.com\b": mapping[_key("EMAIL", None)],
            r"\bemail\b": mapping[_key("EMAIL", None)],
            r"\+91-?XXXXXXXXXX\b": mapping[_key("PHONE", None)],
            r"\+91-?9876543210\b": mapping[_key("PHONE", None)],
            r"\blinkedin\.com/in/yourprofile\b": mapping[_key("LINKEDIN_URL", None)],
            r"\bcandidatelinkedin\b": mapping[_key("LINKEDIN_URL", None)],
            r"\bgithub\.com/candidategithub\b": f"https://github.com/{_slugify_name(mapping[_key('FULL_NAME', None)])}",
        }
        for pat, rep in repls.items():
            new_text = re.sub(pat, rep, new_text, flags=re.IGNORECASE)

        # handle https://[[LINKEDIN_URL]] and bare [[GITHUB_URL]]
        new_text = re.sub(
            r"https?://\[\[(?:LINKEDIN|LINKEDIN_URL)]]",
            _add_scheme(mapping[_key("LINKEDIN_URL", None)]),
            new_text,
            flags=re.IGNORECASE
        )
        new_text = re.sub(
            r"\[\[(?:GITHUB|GITHUB_URL)]]",
            f"https://github.com/{_slugify_name(mapping[_key('FULL_NAME', None)])}",
            new_text,
            flags=re.IGNORECASE
        )

    return new_text, mapping


def replace_placeholders_json(
        obj: Any,
        bank: IndiaBank,
        mapping: dict[str, str] | None = None,
) -> tuple[Any, dict[str, str]]:
    """
    Recursively replace placeholders anywhere in a JSON-like structure.
    Returns (new_obj, mapping).
    """
    mapping = dict(mapping or {})
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            nv, mapping = replace_placeholders_json(v, bank, mapping)
            out[k] = nv
        return out, mapping
    if isinstance(obj, list):
        out = []
        for v in obj:
            nv, mapping = replace_placeholders_json(v, bank, mapping)
            out.append(nv)
        return out, mapping
    if isinstance(obj, str):
        return replace_placeholders_text(obj, bank, mapping)
    return obj, mapping


def _fix_city_state_pairs(txt: str, bank: IndiaBank) -> str:
    if not txt: return txt
    # naive correction for "City, SomeState"
    for city, state in bank.city_to_state.items():
        # wrong state after this city → fix
        txt = re.sub(rf"\b{re.escape(city)}\s*,\s*(?!{re.escape(state)}\b)[A-Za-z .]+\b",
                     f"{city}, {state}", txt)
    return txt


def canonicalize_contacts(contacts: dict, bank: IndiaBank, mapping: dict[str, str]) -> dict:
    """Ensure name/email/phone/links are real, consistent with mapping, and links is a single string."""
    contacts = dict(contacts or {})
    name = contacts.get("name") or mapping.get("FULL_NAME1") or bank.sample_name()
    email = contacts.get("email") or mapping.get("EMAIL1") or bank.sample_email(name)
    phone = contacts.get("phone") or mapping.get("PHONE1") or bank.sample_phone()
    links_raw = contacts.get("links")
    basic = normalize_links_str(links_raw)
    if not basic or _COMMON_SENTINELS_RE.search(basic) or "[[" in basic:
        basic = mapping.get("LINKEDIN_URL1") or bank.sample_linkedin(name)
    basic = _add_scheme(basic)

    # Clean obvious sentinels from email too
    if re.fullmatch(r"email|email@example\.com", str(email), flags=re.I):
        email = mapping.get("EMAIL1") or bank.sample_email(name)

    return {"name": name, "email": email, "phone": phone, "links": basic}


def fix_common_placeholders(s: str, mapping: dict[str, str]) -> str:
    """Best-effort sweep for leftover dummy tokens in text blocks."""
    if not s:
        return s
    out = s
    repls = {
        r"\bCandidate Name\b": mapping.get("FULL_NAME1", ""),
        r"\bFull Name\b": mapping.get("FULL_NAME1", ""),
        r"\bemail@example\.com\b": mapping.get("EMAIL1", ""),
        r"\bemail\b": mapping.get("EMAIL1", ""),
        r"\+91-?XXXXXXXXXX\b": mapping.get("PHONE1", ""),
        r"\+91-?9876543210\b": mapping.get("PHONE1", ""),
        r"\blinkedin\.com/in/yourprofile\b": mapping.get("LINKEDIN_URL1", ""),
        r"https?://\[\[(?:LINKEDIN|LINKEDIN_URL)\]\]": mapping.get("LINKEDIN_URL1", ""),
        r"\[\[(?:GITHUB|GITHUB_URL)\]\]": f"https://github.com/{_slugify_name(mapping.get('FULL_NAME1', ''))}",
    }
    for pat, rep in repls.items():
        if rep:
            out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    return out
