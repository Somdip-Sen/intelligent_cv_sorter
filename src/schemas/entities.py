# Define exact shapes for what the LLM must return.If LLM output deviates, Pydantic.model_validate(...) throws error

from pydantic import BaseModel, Field, HttpUrl, EmailStr, validator
# Ref: Pydantic models, which inherit from BaseModel, allow you to define the structure of your data using Python type hints.
from typing import List, Optional, Dict
from datetime import date


class Persona(BaseModel):
    """who the candidate is (core story, skills seed, role/seniority/domain)."""
    id: str
    core_story: str  # e.g., "Java dev, 5y, fintech"
    role_seed: str
    seniority: str
    domain: str
    skills_seed: List[str]
    constraints: Dict[str, str] = {}


class JobJD(BaseModel):
    """the job object (title, must_haves, responsibilities, raw_text)."""
    id: str
    title: str
    company_stage: Optional[str] = None
    domain: str
    location: Optional[str] = None
    must_haves: List[str]
    nice_to_haves: List[str] = []
    responsibilities: List[str] = []
    salary_hint: Optional[str] = None
    raw_text: str


class EvidenceLink(BaseModel):
    source: str  # "persona" or "jd"
    key: str  # e.g., "skills_seed[PyTorch]" or "must_haves[cloud]"


class CVSection(BaseModel):
    header: str
    bullets: List[str]
    evidence: List[EvidenceLink] = []


class CVDoc(BaseModel):
    """the produced CV (contacts/skills/sections/raw_text/markdown)."""
    id: str
    persona_id: str
    jd_id: str
    role_claim: str
    seniority_claim: Optional[str] = None
    contacts: Dict[str, Optional[str]] = {"name": None, "email": None, "phone": None, "links": None}
    skills: List[str] = []
    sections: List[CVSection]
    raw_markdown: str
    raw_text: str
    render_pdf_path: Optional[str] = None


class RecruiterScore(BaseModel):
    """judge subscores + overall score."""
    cv_id: str
    jd_id: str
    judge_id: str
    subscores: Dict[str, float]  # {"must_have_coverage":0.9, "impact":0.7, ...}
    overall: float


class SampleRecord(BaseModel):
    """one training row combining persona + JD + CV + scores + meta."""
    generation: int
    cv: CVDoc
    jd: JobJD
    persona: Persona
    scores: List[RecruiterScore]
    meta: Dict[str, str] = {}
