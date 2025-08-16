from src.schemas.entities import Persona, JobJD, CVDoc, RecruiterScore, SampleRecord


def test_min_fields():
    Persona(id="p1", core_story="java dev", role_seed="SWE", seniority="mid", domain="fintech", skills_seed=["java"])
    JobJD(id="j1", title="SWE", domain="fintech", must_haves=["java"], raw_text="...")
