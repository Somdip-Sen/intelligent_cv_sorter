import re
from typing import Any, Tuple
from .india_bank import IndiaBank

# Recognize both location and PII placeholders.
# Examples: [[CITY1]], [[STATE2]], [[PIN]], [[FULL_NAME]], [[EMAIL]], [[PHONE]], [[LINKEDIN_PROFILE]]
_PH_RE = re.compile(r"\[\[(CITY|STATE|COLLEGE|PIN|FULL_NAME|EMAIL|PHONE|LINKEDIN_PROFILE)(\d+)?\]\]")


def replace_placeholders_text(
        text: str,
        bank: IndiaBank,
        mapping: dict[str, str] | None = None,
        _PH_RE=None
) -> Tuple[str, dict[str, str]]:
    """
    Replace [[CITY1]], [[STATE1]], [[COLLEGE1]], [[PIN1]] and PII [[FULL_NAME1]], [[EMAIL1]], [[PHONE1]], [[LINKEDIN_PROFILE1]]
    in a string. Returns (new_text, mapping) where mapping holds all resolved tokens for consistency.
    - Keeps a stable mapping so the same token (e.g., CITY1) resolves consistently everywhere.
    - Derives STATEi and PINi from CITYi when available; PINi uses (CITYi, STATEi) if present.
    """
    if not text:
        return text or "", dict(mapping or {})
    mapping = dict(mapping or {})

    def _k(tag: str, idx: str | None) -> str:
        return f"{tag}{(idx or '1')}"

    def _ensure_derived(idx: str | None):
        k_city = _k("CITY", idx)
        k_state = _k("STATE", idx)
        k_pin = _k("PIN", idx)
        if k_city in mapping and k_state not in mapping:
            mapping[k_state] = bank.sample_state(mapping[k_city])
        if k_pin not in mapping:
            city = mapping.get(k_city)
            state = mapping.get(k_state)
            if city or state:
                mapping[k_pin] = bank.sample_pin(city, state)

    def _value(tag: str, idx: str | None) -> str:
        key = _k(tag, idx)
        if key in mapping:
            return mapping[key]

        if tag == "CITY":
            v = bank.sample_city()
            mapping[key] = v
            _ensure_derived(idx)
        elif tag == "STATE":
            v = bank.sample_state(mapping.get(_k("CITY", idx)))
            mapping[key] = v
            _ensure_derived(idx)
        elif tag == "COLLEGE":
            v = bank.sample_college()
            mapping[key] = v
        elif tag == "PIN":
            v = bank.sample_pin(mapping.get(_k("CITY", idx)), mapping.get(_k("STATE", idx)))
            mapping[key] = v
        elif tag == "FULL_NAME":
            v = bank.sample_name()
            mapping[key] = v
        elif tag == "EMAIL":
            name = mapping.get(_k("FULL_NAME", idx)) or bank.sample_name()
            v = bank.sample_email(name)
            mapping[key] = v
        elif tag == "PHONE":
            v = bank.sample_phone()
            mapping[key] = v
        elif tag == "LINKEDIN_PROFILE":
            name = mapping.get(_k("FULL_NAME", idx)) or bank.sample_name()
            v = bank.sample_linkedin(name)
            mapping[key] = v
        else:
            v = ""
            mapping[key] = v
        return v

    def _repl(m: re.Match) -> str:
        return _value(m.group(1), m.group(2))

    new_text = _PH_RE.sub(_repl, text)
    return new_text, mapping


def replace_placeholders_json(
        obj: Any,
        bank: IndiaBank,
        mapping: dict[str, str] | None = None,
        _PH_RE=_PH_RE
) -> tuple[Any, dict[str, str]]:
    """
    Recursively replace placeholders anywhere in a JSON-like structure.
    Returns (new_obj, mapping).
    """
    mapping = dict(mapping or {})
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            nv, mapping = replace_placeholders_json(v, bank, mapping, _PH_RE=_PH_RE)
            out[k] = nv
        return out, mapping
    if isinstance(obj, list):
        out = []
        for v in obj:
            nv, mapping = replace_placeholders_json(v, bank, mapping, _PH_RE=_PH_RE)
            out.append(nv)
        return out, mapping
    if isinstance(obj, str):
        return replace_placeholders_text(obj, bank, mapping, _PH_RE=_PH_RE)
    return obj, mapping
