"""
inputs: Subscores from the Recruiter agent committee:
must_have_coverage, impact_evidence, recency_seniority, tech_depth, ats_readability, realism (each 0–1).
Penalties we compute:
credential_violation, date_incoherence, keyword_stuffing, dup_similarity (each 0–1).
Novelty (0–1): reward for being unlike others.
"""
from typing import List, Dict
import numpy as np

WEIGHTS = {
    "must_have_coverage": 0.40,
    "impact_evidence": 0.20,
    "recency_seniority": 0.15,
    "tech_depth": 0.10,
    "ats_readability": 0.05,
    "realism": 0.10,
}

PENALTIES = {
    "credential_violation": 0.50,
    "date_incoherence": 0.30,
    "keyword_stuffing": 0.25,
    "dup_similarity": 0.40,
}


def aggregate_committee(scores: List[Dict[str, float]]) -> Dict[str, float]:
    """averages subscores across multiple recruiter judges (reduces bias)."""
    # mean of each subscore across recruiter judges
    keys = set().union(*[s.keys() for s in scores])
    return {k: float(np.mean([s.get(k, 0.0) for s in scores])) for k in keys}


def fitness(subscores: Dict[str, float], penalties: Dict[str, float], novelty: float) -> float:
    """weighted sum of subscores minus weighted penalties plus a small novelty bonus.
    This gives a single fitness number ∈ [0,1] for selection."""
    base = sum(WEIGHTS[k] * subscores.get(k, 0.0) for k in WEIGHTS)
    penalty = sum(PENALTIES[k] * penalties.get(k, 0.0) for k in PENALTIES)
    # novelty in [0,1]; encourage diversity
    return max(0.0, min(1.0, base - penalty + 0.10 * novelty))
