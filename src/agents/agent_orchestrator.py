# smoke test -> export N_PERSONAS=10 N_JOBS=2 GENAI_RPM_SLEEP=1.1 (for the whole terminal) or
# N_PERSONAS=5 N_JOBS=1 GENAI_RPM_SLEEP=1.1 python -m src.pipeline.generate_epoch --output data/synth/raw --generation 0/1/2/3/...
import logging
# with pro
# export GEMINI_MODEL_JOB="models/gemini-2.5-flash"
# export GEMINI_MODEL_CANDIDATE="models/gemini-2.5-flash"
#
# # stronger judge on Pro
# export GEMINI_MODEL_RECRUITER="models/gemini-2.5-pro"
#
# # (optional) run-size & budget for the day
# export N_PERSONAS=8 N_JOBS=2 LLM_BUDGET=80 GENAI_RPM_SLEEP=0.8
# python -m src.pipeline.generate_epoch --output data/synth/raw --generation 3

import os
import random
import re
import json
import time
import uuid
import hashlib
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from src.evolution.style_genes import gen_style_profile
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import yaml
from src.config.settings import load_cfg
from src.utils.run_ctx import set_seed
from src.utils.cache import call_with_cache, debug_dump

from src.post.india_bank import IndiaBank
from src.post.india_enforce import enforce_india_cv, enforce_india_jd
from src.post.placeholders import normalize_links_str

cfg = load_cfg()  # or load_cfg(cli_seed=args.seed)
set_seed(cfg.seed)  # your existing helper

# Configuration & Setup
# -----------------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file.")
genai.configure(api_key=API_KEY)

MODEL_NAME = os.getenv("GEMINI_MODEL", "models/gemini-1.5-flash-latest")
RPM_SLEEP_SEC = float(os.getenv("GENAI_RPM_SLEEP", "1.1"))  # throttle safety

BANK = IndiaBank("data/india_bank.yaml", p_invent=float(os.getenv("INDIA_P_INVENT", "0.15")), seed=cfg.seed)


def mk_model(env_var: str, default_name: str):
    name = os.getenv(env_var, default_name)
    try:
        # Newer SDKs accept the 'model_name' kwarg
        return genai.GenerativeModel(model_name=name)
    except TypeError:
        # Older examples allow positional
        return genai.GenerativeModel(name)


job_model = mk_model("GEMINI_MODEL_JOB", "gemini-1.5-flash")
candidate_model = mk_model("GEMINI_MODEL_CANDIDATE", "gemini-1.5-flash")
recruiter_model = mk_model("GEMINI_MODEL_RECRUITER", "gemini-1.5-pro")

# Utilities
# -----------------------------

# Indian context ====
IND_FIRST = ["Aarav", "Vihaan", "Vivaan", "Aditya", "Arjun", "Kabir", "Rohit", "Rahul", "Siddharth", "Pranav", "Rajat",
             "Abhinav", "Karthik", "Manish", "Aakash", "Harsh", "Ishaan", "Yash", "Rakesh", "Sumit", "Vikram", "Nikhil",
             "Rohan", "Ankit", "Saurabh", "Amit", "Varun", "Deepak", "Sandeep"]
IND_LAST = ["Sharma", "Verma", "Gupta", "Patel", "Reddy", "Iyer", "Menon", "Naidu", "Rao", "Singh", "Kumar", "Das",
            "Mukherjee", "Chatterjee", "Nair", "Shah", "Mehta", "Agarwal", "Bansal", "Ghosh", "Chowdhury", "Mishra",
            "Tripathi", "Pandey", "Yadav", "Jain", "Kulkarni", "Shetty", "Pillai"]
IND_EMAIL_DOMAINS = ["gmail.com", "outlook.com", "yahoo.co.in", "proton.me", "zoho.in"]

_used_names, _used_phones, _used_emails = set(), set(), set()


def _rand_indian_name(rng):
    return f"{rng.choice(IND_FIRST)} {rng.choice(IND_LAST)}"


def _rand_indian_phone(rng):
    start = rng.choice(["6", "7", "8", "9"])
    return "+91" + start + "".join(str(rng.randint(0, 9)) for _ in range(9))


def _rand_indian_email(name, rng):
    base = name.lower().replace(" ", ".")
    dom = rng.choice(IND_EMAIL_DOMAINS)
    suffix = str(rng.randint(11, 99))
    return f"{base}{suffix}@{dom}"


# ============

def load_prompt(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"ERROR: Prompt file not found at {file_path}"


def _skills_to_strings(sk) -> list[str]:
    """
    Accepts skills as:
      - list[str]
      - list[dict] like {"name": "...", "keywords": ["..."], "level": "..."}
    Returns a deduped flat list[str].
    """
    if not sk:
        return []
    out = []
    for item in sk:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            # prefer 'name', then merge a few keywords
            name = item.get("name")
            if name:
                out.append(str(name))
            kws = item.get("keywords") or []
            # keep it tight to avoid blowing up the list
            for k in kws[:6]:
                if isinstance(k, str):
                    out.append(k)
    # dedupe, preserve order
    seen = set()
    flat = []
    for s in out:
        s = s.strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            flat.append(s)
    return flat


def _strip_code_fences(s: str) -> str:
    if not s: return ""
    return re.sub(r"```(?:json|JSON)?|```", "", s).strip()


def _json_sanitize(s: str) -> str:
    if not s: return ""
    s = _strip_code_fences(s)
    s = s.replace("\r", "")
    # remove // and /* */ comments
    s = re.sub(r"//.*?$", "", s, flags=re.MULTILINE)
    s = re.sub(r"/\*[\s\S]*?\*/", "", s)
    # Python → JSON literals
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    # fix trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def _extract_top_json_array(s: str) -> str | None:
    """Return the first top-level JSON array substring using bracket balancing."""
    start = s.find("[")
    if start == -1: return None
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(s[start:], start=start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "\"":
                in_str = False
            continue
        if ch == "\"":
            in_str = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def _parse_committee_json(text: str):
    """Best-effort parse for committee outputs."""
    s = _json_sanitize(text)

    # 1) try whole payload
    try:
        j = json.loads(s)
        if isinstance(j, list):
            return j
    except json.JSONDecodeError:
        pass

    # 2) try top-level array slice
    arr_str = _extract_top_json_array(s)
    if arr_str:
        try:
            return json.loads(_json_sanitize(arr_str))
        except json.JSONDecodeError:
            pass

    # 3) fallback: collect dicts and keep those with expected keys
    objs = []
    for m in re.finditer(r"\{[^{}]*}", s, flags=re.DOTALL):
        chunk = _json_sanitize(m.group(0))
        try:
            o = json.loads(chunk)
            if isinstance(o, dict) and ("subscores" in o or "overall" in o):
                objs.append(o)
        except Exception:
            continue
    return objs


def parse_multi_part_response(response_text: str) -> Dict[str, Any]:
    """
    Robustly split a response that may contain:
        - Recruiter: pure JSON (object or array) → {"json": <obj/arr>, "markdown": "", "diag": None}
        - JD/Candidate: JSON --- Markdown resume [--- YAML] → returns all three parts
        - Fallbacks: first JSON/array found; else YAML-as-JSON; else markdown-only
    Returns: {'json': dict,  'markdown': str, 'diag': dict }
    """
    s = (response_text or "").strip()
    out = {"markdown": "", "json": {}, "diag": None}
    if not s:
        return out

    def strip_fences(x: str) -> str:
        # remove ```json / ```yaml / ``` fences if present
        return re.sub(r"```(?:json|yaml)?|```", "", x, flags=re.IGNORECASE).strip()

    # 0) Pure JSON (recruiter single/committee)
    try:
        out["json"] = json.loads(strip_fences(s))
        return out
    except json.JSONDecodeError:
        pass

    def _first_balanced_json(payload: str) -> tuple[str | None, int | None]:
        s = payload
        start_idx = None
        depth = 0
        in_str = False
        esc = False
        opener = None
        for i, ch in enumerate(s):
            if start_idx is None:
                if ch in "{[":
                    start_idx = i
                    opener = ch
                    depth = 1
                    in_str = False
                    esc = False
                continue
            # inside candidate JSON
            if in_str:
                if not esc and ch == '"':
                    in_str = False
                esc = (ch == '\\') and not esc
                continue
            else:
                if ch == '"':
                    in_str = True
                    esc = False
                    continue
                if ch in "{[":
                    depth += 1
                elif ch in "}]":
                    depth -= 1
                    if depth == 0:
                        return s[start_idx:i + 1], start_idx
        return None, None

    # 1) New canonical: JSON --- Markdown [--- YAML]
    parts = re.split(r'(?m)^\s*---\s*$', s)
    if len(parts) >= 2:
        first = strip_fences(parts[0])
        # JSON object or array
        try:
            out["json"] = json.loads(first)
        except json.JSONDecodeError:
            # extract first {...} or [...] if extra text crept in
            m = re.search(r'[{\[][\s\S]*[}\]]', first)
            if m:
                try:
                    out["json"] = json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass

        out["markdown"] = parts[1].strip()

        if len(parts) >= 3:
            diag_raw = strip_fences(parts[2])
            try:
                out["diag"] = yaml.safe_load(diag_raw)
            except yaml.YAMLError as e:
                # Now you know for sure the failure was due to invalid YAML
                logging.warning(f"YAML parsing failed, keeping raw string. Error: {e}")
                out["diag"] = diag_raw  # keep raw if YAML parse fails
        return out

    # 2) Fallback: find first JSON object/array anywhere; markdown is the prefix
    no_fence = strip_fences(s)
    frag, idx0 = _first_balanced_json(no_fence)
    if frag:
        try:
            out["json"] = json.loads(frag)
            out["markdown"] = no_fence[idx0+len(frag):].lstrip()
            return out
        except json.JSONDecodeError as e:
            # Best practice is to log that this attempt failed
            logging.warning(f"Regex found a JSON-like object that failed to parse: {e}")
            # The original intent was to do nothing and let the code continue to the next fallback
            pass

    # 3) Fallback: try parsing whole payload YAML-as-JSON
    try:
        y = yaml.safe_load(no_fence)
        if isinstance(y, (dict, list)):
            out["json"] = y
            return out
    except yaml.YAMLError as e:
        # Now you know for sure the failure was due to invalid YAML
        logging.warning(f"YAML parsing failed, keeping raw string. Error: {e}")

    # 4) Give up → markdown-only
    print("[parse-fail]", s[:200].replace("\n", " "), "...")
    out["markdown"] = s
    return out


def _model(agent: str) -> genai.GenerativeModel:
    """
    agent ∈ {"job","candidate","recruiter"} → choose model per agent via env.(compatible with old/new SDKs.)
    Fallbacks keep you productive even if env vars are unset.
    """
    # precedence: agent-specific → global → sensible default
    default_map = {
        "job": "models/gemini-2.5-flash",
        "candidate": "models/gemini-2.5-flash",
        "recruiter": "models/gemini-2.5-pro",
    }
    name = (
            os.getenv(f"GEMINI_MODEL_{agent.upper()}") or
            os.getenv("GEMINI_MODEL") or
            default_map.get(agent, "models/gemini-2.5-flash")
    )

    m = None
    for ctor in (
            lambda n: genai.GenerativeModel(model_name=n),  # newer SDKs
            lambda n: genai.GenerativeModel(model_name=n),  # some mid versions
            lambda n: genai.GenerativeModel(n),  # older positional
    ):
        try:
            m = ctor(name)
            break
        except TypeError:
            continue
    if m is None:
        raise RuntimeError("Unable to construct GenerativeModel; upgrade google.generativeai.")

    # optional: expose a label for caching without touching read-only props
    try:
        setattr(m, "_cache_model_label", name)
    except AttributeError:
        pass
    return m


def _sleep():
    # simple throttle guard
    time.sleep(RPM_SLEEP_SEC)


# --- Curly-safe formatter: only {role_concept} etc. are substituted; all other { } stay literal.
def safe_curly_format(template: str, vars: dict) -> str:
    # 1) temporarily replace our placeholders {key} with sentinels
    keys = sorted(vars.keys(), key=len, reverse=True)
    sent = {k: f"@@__{k}__@@" for k in keys}
    for k, s in sent.items():
        template = re.sub(rf"\{{re.escape(k)}}", s, template)
    # 2) escape all remaining braces so JSON stays intact
    template = template.replace("{", "{{").replace("}", "}}")
    # 3) restore placeholders and format
    for k, s in sent.items():
        template.replace(s, f"{{{k}}}")
    return template.format(**vars)


# -----------------------------
# Your original single-agent helpers (kept, adjusted)
# -----------------------------
def run_job_agent(job_details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns {'markdown': <str>, 'json': <dict>} from the job agent prompt.
    """
    model = _model("job")
    prompt_template = load_prompt("src/prompts/job_agent_prompt.txt")

    # Ensure all placeholders exist
    defaults = {
        "role_concept": "", "company_profile": "", "team_context": "",
        "location_mode": "", "seniority": "", "domain": "",
        "employment_type": "", "compensation_policy": "", "visa_policy": "",
        "clearance": "", "travel": "", "application_process": "",
        "equal_opportunity_statement": "", "valid_through": ""
    }
    defaults.update(job_details)
    prompt = safe_curly_format(prompt_template, defaults)

    gen_cfg = {
        "temperature": 0.45 + 0.20 * random.random(),
        "top_p": 0.9, "top_k": 48,
        "max_output_tokens": int(os.getenv("MAX_OUTPUT_TOKENS", '11000')),
        "stop_sequences": ["\n---\n"],  # <- back to the separator the parser expects
        # "response_mime_type": "application/json",  # # JSON-only mode (ignored by older SDKs)
    }
    _print_prompt_tokens(model, prompt, "job_agent")
    data = call_with_cache(model, prompt, generation_config=gen_cfg)  # cache key = (MODEL, prompt)
    if not data.get("cached"):
        _sleep()
    parts = parse_multi_part_response(data["text"])
    debug_dump("job", data["text"], parts.get("json"), data.get("finish_reason"))
    if not parts.get("json"):
        # one retry: force JSON-only
        data = call_with_cache(
            model,
            prompt + "\n\n[OUTPUT CONTRACT]\nReturn JSON ONLY. Start with '{' and end with '}'.",
            generation_config=gen_cfg,
        )
        if not data.get("cached"): _sleep()
        parts = parse_multi_part_response(data["text"])
        debug_dump("job", data["text"], parts.get("json"), data.get("finish_reason"))

        if not parts.get("json"):
            parts["json"] = {"_parse_error": True}

    return parts


def run_candidate_agent(candidate_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns {'markdown': <str>, 'json': <dict>} from the candidate agent prompt.
    """
    model = _model("candidate")
    prompt_template = load_prompt("src/prompts/candidate_agent_prompt.txt")

    defaults = {
        "persona": "", "job_markdown": "", "job_json": "null",
        "recruiter_feedback": "null", "generation": 1, "mutation_seed": "null",
        "max_pages": 1, "locale": "en-IN", "salary_expectation": "null",
        "legal": "", "redact_pii": "false", "region_hint": "India"
    }
    gen_cfg = {
        "temperature": 0.45 + 0.20 * random.random(),  # 0.45–0.65
        "top_p": 0.9, "top_k": 48,
        # "response_mime_type": "application/json",  # # JSON-only mode (ignored by older SDKs)
        "max_output_tokens": 11000,
        "stop_sequences": ["\n## END"]
    }
    contract = (
        "\n\n[OUTPUT CONTRACT]\n"
        "Emit exactly:\n"
        "  1) A single JSON object (no trailing commas).\n"
        "  2) A line with only ---\n"
        "  3) Markdown resume.\n"
        "ASCII only. No extra prose before/after. Do not include code fences."
    )
    defaults.update(candidate_inputs)
    prompt = safe_curly_format(prompt_template, defaults) + contract
    # seed variability (deterministic across runs)
    seed_src = f"cand|{defaults.get('generation')}|{defaults.get('mutation_seed')}|{cfg.seed}"
    seed = int(hashlib.sha256(seed_src.encode()).hexdigest(), 16) % 10 ** 9
    rng = random.Random(seed)

    _print_prompt_tokens(model, prompt, "candidate_agent")
    data = call_with_cache(model, prompt, generation_config=gen_cfg)
    if not data.get("cached"): _sleep()
    parts = parse_multi_part_response(data["text"])
    debug_dump("candidate", data["text"], parts.get("json"), data.get("finish_reason"))

    # Retry once if JSON missing
    if not parts.get("json"):
        data = call_with_cache(
            model,
            prompt + "\n\n[OUTPUT CONTRACT]\nReturn JSON ONLY. Start with '{' and end with '}'.",
            generation_config=gen_cfg,
        )
        if not data.get("cached"):
            _sleep()
        parts = parse_multi_part_response(data["text"])
        if not parts.get("json"):
            parts["json"] = {"_parse_error": True}

    # --- synthesize Markdown when model returns JSON-only ---
    if not parts.get("markdown") and isinstance(parts.get("json"), dict):
        j = parts["json"] or {}
        print(j)
        basics = j.get("basics", {}) or {}
        name = basics.get("name") or "Candidate"
        email = basics.get("email") or ""
        phone = basics.get("phone") or ""
        headline = basics.get("summary") or ""

        # collect skills (strings + dict.keywords/name), dedupe
        skills: list[str] = []
        for s in (j.get("skills") or []):
            if isinstance(s, dict):
                if s.get("name"):
                    skills.append(s["name"])
                skills.extend(s.get("keywords") or [])
            elif isinstance(s, str):
                skills.append(s)
        seen = set()
        skills_norm = []
        for s in skills:
            k = s.strip().lower()
            if k and k not in seen:
                seen.add(k)
                skills_norm.append(s.strip())
        skills_norm = skills_norm[:15]

        # build a compact, human-readable resume markdown
        md = [f"# {name}"]
        contact = " · ".join([p for p in [email, phone] if p])
        if contact:
            md.append(contact)
        if headline:
            md.append(f"\n**Summary:** {headline}")
        if skills_norm:
            md.append("\n**Skills:** " + ", ".join(skills_norm))

        for w in (j.get("work") or [])[:2]:
            if not isinstance(w, dict):
                continue
            pos = w.get("position") or ""
            org = w.get("name") or ""
            loc = (w.get("location") or "") or ""
            start = w.get("startDate") or ""
            end = w.get("endDate") or "present"
            dates = "–".join([x for x in [start, end] if x])
            header = " ".join([t for t in [pos and f"**{pos}**,", org] if t]).strip()
            if header:
                md.append(f"\n{header} ({loc}) — {dates}".strip())
            for h in (w.get("highlights") or [])[:4]:
                md.append(f"- {h}")

        parts["markdown"] = "\n".join(md).strip()

    return parts


def run_recruiter_agent_json(job_description: str, cv_text: str, judge_id: str = "judge-0",
                             profile: dict | None = None) -> Dict[str, Any]:
    """
    Uses recruiter prompt but enforces JSON subscores + overall.
    Expected JSON shape:
      {
        "subscores": {
            "must_have_coverage": 0-1,
            "impact_evidence": 0-1,
            "recency_seniority": 0-1,
            "tech_depth": 0-1,
            "ats_readability": 0-1,
            "realism": 0-1
        },
        "overall": 0-1
      }
    """
    model = _model("recruiter")
    prompt_template = load_prompt("src/prompts/recruiter_agent_prompt.txt")

    # judge persona / strictness (affects rubric)
    profile = profile or {}
    strict = profile.get("strictness", 1.0)
    focus = profile.get("focus", [])  # e.g., ["ats_readability","tech_depth"]

    # Append a strict output contract footer (doesn't change rubric content)
    contract = (
        "\n\n[OUTPUT CONTRACT]\n"
        "Return JSON only (no prose), minified, with keys 'subscores' and 'overall'. "
        "All values in [0,1]. Unknown → 0.0."
    )

    persona_hint = (
        f"\n\n[JUDGE PROFILE]\n"
        f"- id: {judge_id}\n"
        f"- style: {profile.get('label', 'balanced')}\n"
        f"- strictness: {strict}\n"
        f"- emphasis: {', '.join(focus) if focus else 'none'}\n"
        "Adjust subscore thresholds accordingly; still keep values in [0,1]."
    )

    # stable per-(cv,jd,judge) randomness → reproducible but different across judges
    seed_src = f"{judge_id}|{hash(job_description) % 10 ** 9}|{hash(cv_text) % 10 ** 9}|{cfg.seed}"
    seed = int(hashlib.sha256(seed_src.encode()).hexdigest(), 16) % 10 ** 9
    rng = random.Random(seed)

    seed_token = rng.randint(1000, 9999)
    persona_hint += f"\n- calibration_token: {seed_token}"
    vars_dict = {"job_description": job_description, "cv_text": cv_text}
    prompt = safe_curly_format(prompt_template, vars_dict) + "\n\n[ACTIVE MODE] A" + persona_hint + contract

    gen_cfg = GenerationConfig(
        temperature=0.25 + 0.20 * rng.random(),  # 0.35–0.60
        top_p=0.9,
        top_k=50,
        candidate_count=1,
        max_output_tokens=12000
    )
    gen_cfg_dict = {
        "temperature": float(gen_cfg.temperature),
        "top_p": float(gen_cfg.top_p),
        "top_k": int(gen_cfg.top_k),
        "response_mime_type": "application/json",  # # JSON-only mode (ignored by older SDKs)
        "candidate_count": int(gen_cfg.candidate_count),
        "max_output_tokens": int(gen_cfg.max_output_tokens)
    }
    _print_prompt_tokens(model, prompt, "recruiter_agent")
    print(f"[cfg recruiter_single] max_output_tokens={gen_cfg_dict['max_output_tokens']}")
    data = call_with_cache(model, prompt, generation_config=gen_cfg_dict)
    if not data.get("cached"): _sleep()

    # Try to parse JSON directly; fallback to extracting first valid JSON object
    txt = data["text"].strip()
    try:
        js = json.loads(txt)
    except Exception:
        # Fallback: find first JSON object
        m = re.search(r'\[[\s\S]*]|\{[\s\S]*}', txt)  # first JSON object/array
        js = json.loads(m.group(0)) if m else {"subscores": {}, "overall": 0.0}

    # ---- SHAPE GUARD: if model slipped an ARRAY for single, take first ----
    if isinstance(js, list):
        js = js[0] if js else {"subscores": {}, "overall": 0.0}
    elif not isinstance(js, dict):
        js = {"subscores": {}, "overall": 0.0}

    # Normalize ranges/keys
    subs = js.get("subscores", {})

    def _clamp(x):
        try:
            return max(0.0, min(1.0, float(x)))
        except Exception:
            return 0.0

    subscores = {
        "must_have_coverage": _clamp(subs.get("must_have_coverage", 0.0)),
        "impact_evidence": _clamp(subs.get("impact_evidence", 0.0)),
        "recency_seniority": _clamp(subs.get("recency_seniority", 0.0)),
        "tech_depth": _clamp(subs.get("tech_depth", 0.0)),
        "ats_readability": _clamp(subs.get("ats_readability", 0.0)),
        "realism": _clamp(subs.get("realism", 0.0)),
    }
    overall = _clamp(js.get("overall", 0.0))
    return {
        "judge_id": judge_id,
        "subscores": subscores,
        "overall": overall
    }


def run_recruiter_committee_once(job_description: str, cv_text: str, profiles: list[dict], jd_json: Any) -> list[dict]:
    model = _model("recruiter")
    tmpl = load_prompt("src/prompts/recruiter_agent_prompt.txt")

    # Build orthogonal judge profiles: weights + views + calibration tokens
    prof_lines = []
    for i, p in enumerate(profiles):
        w = p.get("weights", {
            "must_have_coverage": 0.20, "impact_evidence": 0.15, "recency_seniority": 0.10,
            "tech_depth": 0.20, "ats_readability": 0.10, "realism": 0.25
        })
        # Build judge-specific JD views
        musts = _to_dict({"m": job_description}).get("m")  # job_description already compact
        jd_full = job_description
        jd_musts = "Must haves: " + ", ".join(
            (jd_json.get("must_haves", []) if 'jd_json' in locals() else [])) or jd_full
        jd_resp = "Top responsibilities: " + "; ".join(
            (jd_json.get("responsibilities", []) if 'jd_json' in locals() else [])) or jd_full

        required_view = p.get("view", "full")  # <- use the judge’s requested view; fallback only if missing
        views = {"full": jd_full[:3000], "must_haves_only": jd_musts[:1500], "resp_only": jd_resp[:1500]}
        view_txt = views.get(required_view, jd_full)  # <- the actual JD slice text for that view

        cal = 1117 + 37 * i  # simple per-judge calibration token

        prof_lines.append(
            " - judge_id: judge-{i}; style: {label}; strictness: {s}; emphasis: {focus}; "
            "weights: {w}; view: {required_view}; calibration_token: {cal}\n"
            "   [JD VIEW TEXT]\n{view_txt}\n".format(
                i=i,
                label=p.get("label", "balanced"),
                s=p.get("strictness", 1.0),
                focus=", ".join(p.get("focus", [])) or "none",
                w=w,
                required_view=required_view,  # <- just the label: "full" | "must_haves_only" | "resp_only"
                cal=cal,
                view_txt=view_txt  # <- the concrete slice the judge should read
            ))

    committee_hint = (
            "\n\n[JUDGE COMMITTEE]\n" + "\n".join(prof_lines) +
            "\nEach judge must evaluate INDEPENDENTLY using their own weights/view; "
            "do NOT copy or average other judges' scores."
            "\nEnforce diversity: the pairwise L1 distance between judges' SUBSCORE vectors "
            "should be >= 0.06 when evidence permits. If too similar, adjust minimally per profile."
            "\nReturn an ARRAY of JSON objects (same order)."
    )

    # Optional: give different JD 'views' per judge (the model still sees all, but is told what to consider)
    # (keep jd_desc concise)
    jd_full = job_description
    vars_dict = {"job_description": jd_full, "cv_text": cv_text}
    prompt = (safe_curly_format(tmpl, vars_dict)
              + "\n\n[ACTIVE MODE] B\n[COMMITTEE SIZE] " + str(len(profiles))
              + committee_hint
              + "\n\n[OUTPUT CONTRACT]\nReturn a MINIFIED JSON ARRAY of exactly "
              + str(len(profiles)) + " objects in the specified schema. No prose."
              )

    gen_cfg_dict = {"temperature": 0.30,
                    "top_p": 0.9,
                    "top_k": 50,
                    "max_output_tokens": 13000,
                    "response_mime_type": "application/json"  # # JSON-only mode
                    }
    _print_prompt_tokens(model, prompt, "recruiter_commitee_agent")
    data = call_with_cache(model, prompt, generation_config=gen_cfg_dict)
    txt = data["text"].strip()

    # Parse committee array
    arr = _parse_committee_json(txt)
    out = []

    def clamp01(x):
        try:
            return max(0.0, min(1.0, float(x)))
        except:
            return 0.0

    for obj in arr:
        subs = obj.get("subscores", {})
        out.append({
            "judge_id": obj.get("judge_id", "judge-0"),
            "subscores": {
                "must_have_coverage": clamp01(subs.get("must_have_coverage", 0)),
                "impact_evidence": clamp01(subs.get("impact_evidence", 0)),
                "recency_seniority": clamp01(subs.get("recency_seniority", 0)),
                "tech_depth": clamp01(subs.get("tech_depth", 0)),
                "ats_readability": clamp01(subs.get("ats_readability", 0)),
                "realism": clamp01(subs.get("realism", 0)),
            },
            "overall": clamp01(obj.get("overall", 0))
        })
    return out


# Epoch-facing API (used by generate_epoch.py)
# -----------------------------
def generate_personas(generation: int, n_personas: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Creates a simple population of persona dicts.
    Matches the Persona schema fields expected by your Pydantic model.
    """
    n = n_personas or int(os.getenv("N_PERSONAS", "20"))
    roles = ["SWE", "Data Scientist", "ML Engineer", "Backend", "Frontend"]
    domains = ["fintech", "ecommerce", "infra", "healthtech", "logistics"]
    seniorities = ["junior", "mid", "senior", "staff", "lead", "principal"]

    rng = random.Random(cfg.seed * cfg.generation_prime + generation)  # stable per generation

    out = []
    for i in range(n):
        role = roles[i % len(roles)]
        dom = domains[(i // len(roles)) % len(domains)]
        sen = seniorities[(i // (len(roles) * len(domains))) % len(seniorities)]
        style = gen_style_profile(rng)  # <-- style DNA
        out.append({
            "id": f"persona-{generation}-{i}",
            "core_story": f"{role} with {2 + (i % 8)} years in {dom}",
            "role_seed": role,
            "seniority": sen,
            "domain": dom,
            "skills_seed": ["Python", "SQL"] if "Data" in role or "ML" in role else ["Java", "AWS"],
            "constraints": {"style_profile": style}
        })
    return out


def generate_jobs(generation: int, n_jobs: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Calls your job agent prompt multiple times and returns Job objects
    that include both structured fields and the raw markdown as `raw_text`.
    """
    k = n_jobs or int(os.getenv("N_JOBS", "5"))
    # Minimal job_details; you can extend via env/config
    modes = ["Hybrid (India)", "Remote (India)", "WFH (India)"]
    base = {
        "role_concept": "Software/ML roles in Indian market across domains",
        "seniority": "Mixed",
        "employment_type": "Full-time",
    }
    jobs = []

    for j in range(k):
        base_with_random_mode = {**base, "location_mode": random.choice(modes)}
        jd_parts = run_job_agent(base_with_random_mode)
        jd_json_raw = jd_parts.get("json") or {}
        jd_json: dict[str, Any] = dict(jd_json_raw)  # ensure built-in dict
        jd_md = jd_parts.get("markdown", "")

        jd_json = dict(jd_json)
        qualifications = jd_json.get("qualifications", {})
        # LLMs sometimes regress or vary casing.So a light fallback mapper (only runs when needed)
        if "must_haves" not in jd_json and "mustHave" in qualifications:
            jd_json["must_haves"] = qualifications.get("mustHave", []) or []
        if "nice_to_haves" not in jd_json and "niceToHave" in qualifications:
            jd_json["nice_to_haves"] = qualifications.get("niceToHave", []) or []
        # ensure required fields exist
        jd_json.setdefault("title", jd_json.get("title") or "Unknown Title")
        jd_json.setdefault("domain", jd_json.get("domain") or "general")
        jd_json["title"] = jd_json.get("title") or "Unknown Title"
        jd_json["domain"] = jd_json.get("domain") or "general"
        jd_json["id"] = jd_json.get("id") or f"jd-{generation}-{j}"
        jd_json["raw_text"] = jd_md
        jd_json = _normalize_job_json(jd_json, jd_md)

        # --- Indianization + placeholder resolution---
        # Hard defaults only when missing
        jd_json.setdefault("salaryCurrency", "INR")
        jd_json.setdefault("applicantLocationRequirements", ["India"])
        if not jd_json.get("jobLocationType"):
            jd_json["jobLocationType"] = "HYBRID"
        if not jd_json.get("jobLocation"):
            city = random.choice(["Kolkata", "Bengaluru", "Hyderabad", "Pune", "Gurgaon", "Noida", "Chennai", "Mumbai"])
            jd_json["jobLocation"] = [{
                "@type": "Place",
                "address": {"@type": "PostalAddress", "addressLocality": city, "addressCountry": "IN"}
            }]

        # Ensure MD mentions INR once
        if "₹" not in (jd_json.get("raw_text") or ""):
            jd_json["raw_text"] = (jd_json.get("raw_text") or "") + "\n\n_Compensation shown/paid in INR (₹, LPA)._"

        # Call the enforcer ONCE (handles placeholders in JSON, lists, and Markdown)
        title_lc = (jd_json.get("title") or "").lower()
        senior_hint = "senior" if any(k in title_lc for k in ("senior", "staff", "lead", "principal")) else "mid"
        jd_json = enforce_india_jd(jd_json, BANK, senior_hint=senior_hint)
        val = jd_json.get("salary_hint")
        if isinstance(val, dict):
            cur = val.get("currency", "INR")
            lo = val.get("min_lpa") or val.get("min") or val.get("low")
            hi = val.get("max_lpa") or val.get("max") or val.get("high")
            if lo and hi:
                jd_json["salary_hint"] = f"{cur} {lo}–{hi} LPA"
            else:
                # fallback if dict is incomplete
                lo, hi = BANK.sample_salary_lpa((senior_hint or "mid").lower())
                jd_json["salary_hint"] = f"INR {lo}–{hi} LPA"
        jobs.append(jd_json)
    return jobs


def tailor_cv(persona: Any, jd: Any) -> Dict[str, Any]:
    """
    Calls your candidate agent and converts the response into a CV dict
    that matches the CVDoc schema keys used in generate_epoch.py.
    """
    persona = _to_dict(persona)
    # normalize JD to dict + markdown
    if isinstance(jd, str):
        try:
            jd_json, jd_md = parse_json_md(jd)
            jd_json["raw_text"] = jd_md
        except Exception:
            jd_json = {"raw_text": jd}
    else:
        jd_json = _to_dict(jd)

    job_md = jd_json.get("raw_text", "")
    style = persona.get("constraints", {}).get("style_profile", {})
    title_lc = (jd_json.get("title") or "").lower()
    is_senior_jd = any(k in title_lc for k in ("senior", "staff", "lead", "principal"))
    candidate_inputs = {
        "persona": persona.get("core_story", ""),
        "job_markdown": job_md,
        "job_json": json.dumps(jd_json, ensure_ascii=False),
        "generation": persona.get("id", "gen"),
        "mutation_seed": str(uuid.uuid4()),
        "style_profile": style
    }
    cv_parts = run_candidate_agent(candidate_inputs)
    cv_md = cv_parts.get("markdown", "")
    cv_json = cv_parts.get("json", {}) or {}
    skills_list = _skills_to_strings(cv_json.get("skills"))
    # Build a schema-friendly CV dict (safe defaults)

    # senior calibration for more competent CV generation
    if is_senior_jd:
        # ensure seniority claim and minimum years
        cv_json["seniority_claim"] = cv_json.get("seniority_claim") or "senior"
        yrs = cv_json.get("years_experience")
        if yrs is None or float(yrs) < 5:
            cv_json["years_experience"] = 5
    # print(f"CV - JSON \n\n {json.dumps(cv_json, indent=2)}")

    basic_url = (cv_json.get("basics", {}) or {}).get("url")
    contacts_src = cv_json.get("contacts") or {}
    basic_url = (cv_json.get("basics", {}) or {}).get("url")
    contacts = {
        "name": contacts_src.get("name") or (cv_json.get("basics", {}) or {}).get("name") or "",
        "email": contacts_src.get("email") or (cv_json.get("basics", {}) or {}).get("email") or "",
        "phone": contacts_src.get("phone") or (cv_json.get("basics", {}) or {}).get("phone") or "",
        "links": normalize_links_str(contacts_src.get("links") or basic_url),
    }

    cv = {
        "id": f"cv-{uuid.uuid4().hex[:8]}",
        "persona_id": persona.get("id"),
        "jd_id": jd_json.get("id"),
        "role_claim": cv_json.get("role_claim") or jd_json.get("title") or persona.get("role_seed", "Unknown"),
        "seniority_claim": cv_json.get("seniority_claim") or persona.get("seniority"),
        "years_experience": cv_json.get("years_experience"),
        "contacts": contacts,
        "skills": skills_list,  # ← normalized here
        "sections": cv_json.get("sections") or [{"header": "Summary", "bullets": [], "evidence": []}],
        "raw_markdown": cv_md,
        "raw_text": _markdown_to_text(cv_md),
        "render_pdf_path": None
    }

    # Compute seniority hint FROM THE JD TITLE once
    title_lc = (jd_json.get("title") or "").lower()
    senior_hint = "senior" if any(k in title_lc for k in ("senior", "staff", "lead", "principal")) else "mid"
    # Enforce India context & resolve placeholders ONCE (this also normalizes contacts + links to string)
    cv = enforce_india_cv(cv, BANK, senior_hint=senior_hint)  # <-- then replace placeholders, INR, cities, etc.
    return cv


# minimal parser for "JSON --- Markdown"
def parse_json_md(payload: str):
    s = (payload or "").strip()
    # drop any dumper headers like "# model: ..."
    s = "\n".join(line for line in s.splitlines() if not line.startswith("# "))
    parts = re.split(r'(?m)^\s*---\s*$', s)
    js = json.loads(re.sub(r"```(?:json|yaml)?|```", "", parts[0], flags=re.I))
    md = parts[1] if len(parts) > 1 else ""
    return js, md


def score_cv_committee(cv: Any, jd: Any, n_judges: int = 3) -> List[Dict[str, Any]]:
    """
    One-call committee by default. Falls back to single-judge calls only if parsing fails.
    Returns list of {cv_id, jd_id, judge_id, subscores, overall}.
    """

    # normalize JD to dict + markdown
    if isinstance(jd, str):
        jd_json, jd_md = parse_json_md(jd)  # raw "JSON --- Markdown"
        jd_json["raw_text"] = jd_md
    else:
        jd_json = _to_dict(jd)  # Pydantic or dict
        jd_md = jd_json.get("raw_text", "")
    job_md = jd_md or jd_json.get("raw_text", "")
    cv = _to_dict(cv)

    # Token-diet JD for speed
    musts = jd_json.get("must_haves", [])[:8]
    resp = jd_json.get("responsibilities", [])[:5]
    jd_desc = "Must haves: " + ", ".join(musts)
    if resp:
        jd_desc += "\nTop responsibilities: " + "; ".join(resp)

    cv_text = cv.get("raw_markdown") or cv.get("raw_text", "")

    # three orthogonal complementary judges (reproducible personas)
    profiles = [
                   {"label": "conservative", "strictness": 1.15, "focus": ["realism", "tech_depth"],
                    "weights": {"realism": 0.30, "tech_depth": 0.25, "must_have_coverage": 0.20,
                                "impact_evidence": 0.10,
                                "recency_seniority": 0.10, "ats_readability": 0.05},
                    "view": "must_haves_only"},
                   {"label": "balanced", "strictness": 1.00, "focus": [],
                    "weights": {"realism": 0.15, "tech_depth": 0.15, "must_have_coverage": 0.25,
                                "impact_evidence": 0.20,
                                "recency_seniority": 0.15, "ats_readability": 0.10},
                    "view": "full"},
                   {"label": "optimistic", "strictness": 0.90, "focus": ["impact_evidence", "ats_readability"],
                    "weights": {"realism": 0.10, "tech_depth": 0.15, "must_have_coverage": 0.20,
                                "impact_evidence": 0.30,
                                "recency_seniority": 0.10, "ats_readability": 0.15},
                    "view": "resp_only"},
               ][:n_judges]

    # === One-call committee ===
    committee = run_recruiter_committee_once(jd_desc, cv_text, profiles, jd_json)

    # If parsing failed or empty, fall back to per-judge calls (rare)
    if not committee or len(committee) < n_judges:
        # Build orthogonal judge profiles: weights + views + calibration tokens
        prof_lines = []
        scores = []
        for i, prof in enumerate(profiles):
            v = prof.get("view", "full")
            j_id = f"judge-{i}"
            cal = judge_cal_token(j_id, jd_json.get("id", "jd-unk"), cv.get("id", "cv-unk"), cfg.seed)
            prof_lines.append(
                " - judge_id: judge-{i}; style: {label}; strictness: {s}; emphasis: {focus}; weights: {w}; view: {v}; calibration_token: {cal}"
                .format(i=i, label=prof.get("label", "balanced"), s=prof.get("strictness", 1.0),
                        focus=", ".join(prof.get("focus", [])) or "none", w=prof.get("weights", {}), v=v, cal=cal)
            )
            s = run_recruiter_agent_json(jd_desc, cv_text, judge_id=f"judge-{i}", profile=prof)
            scores.append({
                "cv_id": cv["id"], "jd_id": jd_json["id"], "judge_id": s["judge_id"],
                "subscores": s["subscores"], "overall": float(s["overall"])
            })
        return scores
    # Use committee results directly (NO extra LLM calls)
    out = []
    for i, s in enumerate(committee[:n_judges]):
        subs = s["subscores"]
        prof_w = profiles[i]["weights"]  # weights you passed in
        out.append({
            "cv_id": cv["id"], "jd_id": jd_json["id"],
            "judge_id": s.get("judge_id", f"judge-{i}"),
            "subscores": subs,
            "overall": _weighted_overall(subs, prof_w)
        })
    return out


# Helpers
# -----------------------------
def _markdown_to_text(md: str) -> str:
    # very light markdown → text (keep it simple for now)
    text = re.sub(r"`{1,3}.*?`{1,3}", "", md, flags=re.DOTALL)  # remove code blocks/inline
    text = re.sub(r"!\[.*?]\(.*?\)", "", text)  # images
    text = re.sub(r"\[([^]]+)]\([^)]+\)", r"\1", text)  # links → anchor text
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)  # headings
    text = re.sub(r"\*\*|\*", "", text)  # bold/italics
    return text.strip()


def _extract_list_after_heading(md: str, heading_regex: str) -> list[str]:
    """
    Grab bullet list lines immediately after a heading match (until blank line or next heading).
    """
    import re
    m = re.search(heading_regex, md, flags=re.IGNORECASE)
    if not m: return []
    tail = md[m.end():]
    lines = []
    for line in tail.splitlines():
        if re.match(r"^\s*#{1,6}\s", line):  # next section
            break
        if not line.strip():  # blank block break
            if lines: break
            continue
        if re.match(r"^\s*[-*•]\s+", line):
            lines.append(re.sub(r"^\s*[-*•]\s+", "", line).strip())
    return lines


def _print_prompt_tokens(model, prompt: str, label: str):
    try:
        ct = model.count_tokens(prompt)
        total = getattr(ct, "total_tokens", None) or ct["total_tokens"]
        print(f"[tokens] {label}: prompt_total={total}")
    except Exception as e:
        print(f"[tokens] {label}: count failed ({e})")


def _normalize_job_json(jd_json: dict, jd_md: str) -> dict:
    # Accept both camelCase and snake_case inside "qualifications"
    quals = jd_json.get("qualifications") or {}
    musts = jd_json.get("must_haves") or quals.get("must_haves") or quals.get("mustHave") or []
    nices = jd_json.get("nice_to_haves") or quals.get("nice_to_haves") or quals.get("niceToHave") or []
    resps = jd_json.get("responsibilities") or []

    # Last-ditch: mine Markdown sections if lists are still empty
    if not musts:
        musts = _extract_list_after_heading(jd_md, r"what you[’']?ll bring.*?\(?.*must.?have.*?\)?\s*$")
    if not nices:
        nices = _extract_list_after_heading(jd_md, r"bonus points|nice.?to.?have\s*$")
    if not resps:
        resps = _extract_list_after_heading(jd_md, r"what you[’']?ll do|responsibilit(y|ies)\s*$")

    jd_json["must_haves"] = [s for s in (musts or []) if s]
    jd_json["nice_to_haves"] = [s for s in (nices or []) if s]
    jd_json["responsibilities"] = [s for s in (resps or []) if s]
    return jd_json


SCORE_KEYS = ["must_have_coverage", "impact_evidence", "recency_seniority", "tech_depth", "ats_readability", "realism"]


def _to_dict(x):
    if hasattr(x, "model_dump"):
        return x.model_dump(mode="python")
    if hasattr(x, "dict"):
        return x.dict()
    return dict(x) if isinstance(x, dict) else x  # pass through for strings


def judge_cal_token(judge_id: str, jd_id: str, cv_id: str, seed: int) -> int:
    s = f"{judge_id}|{jd_id}|{cv_id}|{seed}"
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % 9000 + 1000  # 1000–9999


def _weighted_overall(subs: dict[str, float], weights: dict[str, float]) -> float:
    # normalize weights in case they don’t sum to 1.0
    z = sum(weights.get(k, 0.0) for k in SCORE_KEYS) or 1.0
    return float(sum((weights.get(k, 0.0) / z) * float(subs.get(k, 0.0)) for k in SCORE_KEYS))
