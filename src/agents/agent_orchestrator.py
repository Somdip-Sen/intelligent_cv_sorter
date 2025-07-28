import os
import re
import json
import time
import google.generativeai as genai
from dotenv import load_dotenv
from generate_structured_data import load_prompt

# --- Configuration & Setup ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file.")
genai.configure(api_key=api_key)

def run_job_agent(job_details: dict):
    """Takes a persona and tailors a CV to a specific job description.
    INPUTS
        - role_concept: {role_concept}
        - company_profile (optional): <mission, stage, industry, size, funding, geography>
        - team_context (optional): <org, manager title, team size, cross-functional partners>
        - location_mode:  <Onsite | Hybrid | Remote> plus locations/time zones if known
        - seniority: <Junior | Mid | Senior | Staff | Principal | Manager | Director | VP>
        - domain: <e.g., fintech risk, logistics, LLM infra, growth marketing>
        - employment_type:  <Full-time | Part-time | Contract | Internship>
        - compensation_policy (optional): currency, base range, variable/equity, benefits summary
        - visa_policy (optional): sponsorship available? yes/no
        - clearance (optional): security clearance requirements, if any
        - travel (optional):  % travel
        - application_process (optional): steps and target timelines
        - equal_opportunity_statement (optional): if absent, generate a standard EEO/Accessibility statement
        - valid_through (optional): ISO 8601 close date
    """

    print("--- CANDIDATE AGENT: Activated ---")
    model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
    prompt_template = load_prompt("prompts/candidate_agent_prompt.txt")

    # This takes all the details from our dictionary and formats them into a string.
    all_possible_inputs = {
        "role_concept": "", "company_profile": "", "team_context": "",
        "location_mode": "", "seniority": "", "domain": "",
        "employment_type": "", "compensation_policy": "", "visa_policy": "",
        "clearance": "", "travel": "", "application_process": "",
        "equal_opportunity_statement": "", "valid_through": ""
    }
    all_possible_inputs.update(job_details)
    prompt = prompt_template.format(**all_possible_inputs) # dictionary unpacking to fill all placeholders

    response = model.generate_content(prompt)
    print("--- CANDIDATE AGENT: CV Tailored ---")
    return response.text

def run_candidate_agent(role_concept: str):
    """Generates a realistic job description for a given role concept."""
    print("--- JOB AGENT: Activated ---")
    model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
    prompt_template = load_prompt("src/prompts/job_agent_prompt.txt")
    prompt = prompt_template.format(job_description=job_description, cv_text=cv_text)
    response = model.generate_content(prompt)
    print("--- JOB AGENT: Job Description Generated ---")
    return response.text


def run_recruiter_agent(job_description: str, cv_text: str):
    """Evaluates a CV against a job description and provides a score."""
    print("--- RECRUITER AGENT: Activated ---")
    model = genai.GenerativeModel('models/gemini-1.5-flash-latest')  # Using Flash for dev speed
    prompt_template = load_prompt("prompts/recruiter_agent_prompt.txt")
    prompt = prompt_template.format(job_description=job_description, cv_text=cv_text)

    response = model.generate_content(prompt)
    print("--- RECRUITER AGENT: Evaluation Complete ---")
    try:
        # Extract the number from the response string "Score: 8/10"
        score = re.search(r'\d+', response.text).group()
        return int(score)
    except (AttributeError, ValueError):
        print(f"--- RECRUITER AGENT: Could not parse score from response: {response.text} ---")
        return 0
    #------------



# --- Agent Definitions ---


def run_recruiter_agent(job_description: str, cv_text: str):
    """Evaluates a CV against a job description and provides a score."""
    print("--- RECRUITER AGENT: Activated ---")
    model = genai.GenerativeModel('models/gemini-1.5-flash-latest')  # Pro model for better evaluation
    prompt = f"""
    You are a senior technical recruiter. Evaluate the following CV based on its relevance and fit for the provided Job Description.
    Provide a score from 1 to 10, where 1 is a poor match and 10 is a perfect match.
    Consider the relevance of skills, experience, and the overall quality of the CV.
    Respond ONLY with the score in the format "Score: [number]/10".

    **JOB DESCRIPTION:**
    ---
    {job_description}
    ---

    **CANDIDATE CV:**
    ---
    {cv_text}
    ---
    """
    response = model.generate_content(prompt)
    print("--- RECRUITER AGENT: Evaluation Complete ---")
    try:
        # Extract the number from the response string "Score: 8/10"
        score = re.search(r'\d+', response.text).group()
        return int(score)
    except (AttributeError, ValueError):
        print(f"--- RECRUITER AGENT: Could not parse score from response: {response.text} ---")
        return 0


# --- Orchestration ---
if __name__ == "__main__":
    job_concept = "Mid-Level Python Backend Developer for a fintech startup"
    candidate_persona = "A software engineer with 4 years of experience, primarily using Java and Spring, but with recent personal projects in Python and Django."

    # 1. Job Agent creates an opportunity
    jd_text = run_job_agent(job_concept)
    print("\n\n=============== JOB DESCRIPTION ===============\n")
    print(jd_text)

    time.sleep(1.1)  # Throttling as in free tier gemini gives 60 request per minute

    # 2. Candidate Agent applies for the job
    candidate_cv = run_candidate_agent(jd_text, candidate_persona)
    print("\n\n================= TAILORED CV =================\n")
    print(candidate_cv)

    time.sleep(1.1)  # Throttling for PRO model(2 RPM) is much larger

    # 3. Recruiter Agent scores the application
    fitness_score = run_recruiter_agent(jd_text, candidate_cv)
    print("\n\n================== EVALUATION ==================\n")
    print(f"FITNESS SCORE: {fitness_score}/10")
