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


# Utilities
# -----------------------------
def load_prompt(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"ERROR: Prompt file not found at {file_path}"


def parse_multi_part_response(response_text: str) -> Dict[str, Any]:
    """
    Robustly split a response that may contain:
      - Part 1: Markdown resume
      - Part 2: '## OUTPUT — PART 2 (Structured JSON)' + minified JSON
      - (optionally) code-fenced JSON first, or JSON-only
    Returns: {'markdown': str, 'json': dict}
    """
    parts = {"markdown": "", "json": {}}
    s = response_text.strip()
    sep = "## OUTPUT — PART 2 (Structured JSON)"

    try:
        # 1) Preferred path: use the canonical separator header
        if sep in s:
            md, rest = s.split(sep, 1)
            parts["markdown"] = md.strip()
            json_str = rest
        else:
            # 2) Fallback: strip code fences to search for the first JSON object anywhere
            no_fence = re.sub(r"```json|```", "", s)
            m = re.search(r"\{.*\}", no_fence, re.DOTALL)
            if not m:
                # No JSON found → treat whole thing as markdown
                parts["markdown"] = s
                return parts
            json_obj = m.group(0)
            # markdown is everything BEFORE that JSON object (from the original string)
            parts["markdown"] = s[: s.find(json_obj)].strip()
            json_str = json_obj

        # 3) Clean up and parse JSON
        json_str = re.sub(r"```json|```", "", json_str).strip()
        last_brace = json_str.rfind("}")
        if last_brace != -1:
            json_str = json_str[: last_brace + 1]
        parts["json"] = json.loads(json_str) if json_str else {}

    except Exception:
        # As a last resort: leave markdown as-is and try YAML→dict (optional)
        try:
            y = yaml.safe_load(s)
            if isinstance(y, dict):
                parts["json"] = y
        except Exception:
            pass

    return parts


def _model() -> genai.GenerativeModel:
    return genai.GenerativeModel(MODEL_NAME)


def _sleep():
    # simple throttle guard
    time.sleep(RPM_SLEEP_SEC)


# --- Curly-safe formatter: only {role_concept} etc. are substituted; all other { } stay literal.
def safe_curly_format(template: str, vars: dict) -> str:
    # 1) temporarily replace our placeholders {key} with sentinels
    keys = sorted(vars.keys(), key=len, reverse=True)
    sent = {k: f"@@__{k}__@@" for k in keys}
    for k, s in sent.items():
        template = re.sub(r"\{" + re.escape(k) + r"\}", s, template)
    # 2) escape all remaining braces so JSON stays intact
    template = template.replace("{", "{{").replace("}", "}}")
    # 3) restore placeholders and format
    for k, s in sent.items():
        template = template.replace(s, "{" + k + "}")
    return template.format(**vars)


# -----------------------------
# Your original single-agent helpers (kept, adjusted)
# -----------------------------
def run_job_agent(job_details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns {'markdown': <str>, 'json': <dict>} from the job agent prompt.
    """
    model = _model()
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
    resp = model.generate_content(prompt)
    _sleep()
    return parse_multi_part_response(resp.text)


def run_candidate_agent(candidate_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns {'markdown': <str>, 'json': <dict>} from the candidate agent prompt.
    """
    model = _model()
    prompt_template = load_prompt("src/prompts/candidate_agent_prompt.txt")

    defaults = {
        "persona": "", "job_markdown": "", "job_json": "null",
        "recruiter_feedback": "null", "generation": 1, "mutation_seed": "null",
        "max_pages": 1, "locale": "en-US", "salary_expectation": "null",
        "legal": "", "redact_pii": "true"
    }
    defaults.update(candidate_inputs)

    prompt = safe_curly_format(prompt_template, defaults)
    resp = model.generate_content(prompt)
    _sleep()
    return parse_multi_part_response(resp.text)


def run_recruiter_agent_json(job_description: str, cv_text: str, judge_id: str = "judge-0", profile: dict | None = None) -> Dict[str, Any]:
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
    model = _model()
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
    prompt = safe_curly_format(prompt_template, vars_dict) + persona_hint + contract

    gen_cfg = GenerationConfig(
        temperature=0.35 + 0.25 * rng.random(),  # 0.35–0.60
        top_p=0.9,
        top_k=50,
        candidate_count=1,
        max_output_tokens=1024,
    )

    resp = model.generate_content(prompt, generation_config=gen_cfg)
    _sleep()

    # Try to parse JSON directly; fallback to extracting first valid JSON object
    txt = resp.text.strip()
    try:
        js = json.loads(txt)
    except Exception:
        # Fallback: find first JSON object
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        js = json.loads(m.group(0)) if m else {"subscores": {}, "overall": 0.0}

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


# -----------------------------
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
    seniorities = ["junior", "mid", "senior"]

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
    base = {
        "role_concept": "Software/ML roles across domains",
        "seniority": "Mixed",
        "location_mode": "Remote",
        "employment_type": "Full-time",
    }
    jobs = []
    for j in range(k):
        jd_parts = run_job_agent(base)
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
        jobs.append(jd_json)
    return jobs


def tailor_cv(persona: Dict[str, Any], jd: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calls your candidate agent and converts the response into a CV dict
    that matches the CVDoc schema keys used in generate_epoch.py.
    """
    job_md = jd.get("raw_text", "")
    style = persona.get("constraints", {}).get("style_profile", {})
    candidate_inputs = {
        "persona": persona.get("core_story", ""),
        "job_markdown": job_md,
        "job_json": json.dumps(jd, ensure_ascii=False),
        "generation": persona.get("id", "gen"),
        "mutation_seed": str(uuid.uuid4()),
        "style_profile": style
    }
    cv_parts = run_candidate_agent(candidate_inputs)
    cv_md = cv_parts.get("markdown", "")
    cv_json = cv_parts.get("json", {}) or {}

    # Build a schema-friendly CV dict (safe defaults)
    cv = {
        "id": f"cv-{uuid.uuid4().hex[:8]}",
        "persona_id": persona.get("id"),
        "jd_id": jd.get("id"),
        "role_claim": cv_json.get("role_claim") or jd.get("title") or persona.get("role_seed", "Unknown"),
        "seniority_claim": cv_json.get("seniority_claim") or persona.get("seniority"),
        "contacts": cv_json.get("contacts") or {"name": None, "email": None, "phone": None, "links": None},
        "skills": cv_json.get("skills") or [],
        "sections": cv_json.get("sections") or [{"header": "Summary", "bullets": [], "evidence": []}],
        "raw_markdown": cv_md,
        "raw_text": _markdown_to_text(cv_md),
        "render_pdf_path": None
    }
    return cv


def score_cv_committee(cv: Dict[str, Any], jd: Dict[str, Any], n_judges: int = 3) -> List[Dict[str, Any]]:
    """
    Calls the recruiter agent multiple times with slight style hints (committee)
    and returns a list of dicts: each with 'judge_id', 'subscores', 'overall' in [0,1].
    """
    jd_desc = jd.get("raw_text", "")
    cv_text = cv.get("raw_markdown") or cv.get("raw_text", "")

    # three complementary judges (reproducible personas)
    profiles = [
                   {"strictness": 1.15, "focus": ["realism", "tech_depth"], "label": "conservative"},
                   {"strictness": 1.00, "focus": [], "label": "balanced"},
                   {"strictness": 0.90, "focus": ["impact_evidence", "ats_readability"], "label": "optimistic"},
               ][:n_judges]

    scores: List[Dict[str, Any]] = []
    for i, prof in enumerate(profiles):
        s = run_recruiter_agent_json(jd_desc, cv_text, judge_id=f"judge-{i}", profile=prof)
        scores.append({
            "cv_id": cv.get("id"),
            "jd_id": jd.get("id"),
            "judge_id": s.get("judge_id", f"judge-{i}"),
            "subscores": s.get("subscores", {}),
            "overall": float(s.get("overall", 0.0))
        })
    return scores


# -----------------------------
# Helpers
# -----------------------------
def _markdown_to_text(md: str) -> str:
    # very light markdown → text (keep it simple for now)
    text = re.sub(r"`{1,3}.*?`{1,3}", "", md, flags=re.DOTALL)  # remove code blocks/inline
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)  # images
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # links → anchor text
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)  # headings
    text = re.sub(r"\*\*|\*", "", text)  # bold/italics
    return text.strip()


# -----------------------------
# Demo run (optional)
# -----------------------------
if __name__ == "__main__":
    # Minimal smoke run (single JD, single persona, single committee)
    persona = {
        "id": "persona-demo-0",
        "core_story": "Data Scientist with 6 years in recommender systems (PyTorch/HF).",
        "role_seed": "Data Scientist",
        "seniority": "senior",
        "domain": "ecommerce",
        "skills_seed": ["Python", "PyTorch", "SQL"],
        "constraints": {}
    }

    jd = generate_jobs(generation=0, n_jobs=1)[0]
    cv = tailor_cv(persona, jd)
    scores = score_cv_committee(cv, jd, n_judges=3)

    print("\n=== JD (markdown head) ===")
    print(jd.get("raw_text", "")[:2000], "...\n") # prints only the first 400 characters and then literally appends an ellipsis.
    print("=== CV (markdown head) ===")
    print(cv.get("raw_markdown", "")[:2000], "...\n")
    print("=== Committee scores ===")
    print(json.dumps(scores, indent=2))
