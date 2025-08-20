"""
inputs: Subscores from the Recruiter agent committee:
must_have_coverage, impact_evidence, recency_seniority, tech_depth, ats_readability, realism (each 0–1).
Penalties we compute:
credential_violation, date_incoherence, keyword_stuffing, dup_similarity (each 0–1).
Novelty (0–1): reward for being unlike others.
"""
from typing import List, Dict, Optional
import numpy as np
from src.config.settings import load_cfg

cfg = load_cfg()  # or load_cfg(cli_seed=args.seed)

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


def _weighted_subscores_minus_penalties(subscores: Dict[str, float],
                                        penalties: Dict[str, float],
                                        cfg: Optional[object] = None, ) -> float:
    """
    Returns base - penalty.
    - subscores/penalties are expected in [0,1].
    - Optional weights from cfg.fitness.subscore_weights / penalty_weights (or uniform).
    - We normalize by sum of weights so base, penalty each stay in [0,1] if inputs are.
    """
    # Optional weights on cfg: cfg.fitness.subscore_weights / penalty_weights
    sw = getattr(getattr(cfg, "fitness", None), "subscore_weights", None) if cfg else None
    pw = getattr(getattr(cfg, "fitness", None), "penalty_weights", None) if cfg else None

    if not sw:
        sw = {k: 1.0 for k in subscores.keys()}
    if not pw:
        pw = {k: 1.0 for k in penalties.keys()}

    sw_sum = sum(float(v) for v in sw.values()) or 1.0
    pw_sum = sum(float(v) for v in pw.values()) or 1.0

    base = sum(float(sw.get(k, 1.0)) * float(subscores.get(k, 0.0)) for k in subscores) / sw_sum
    pen = sum(float(pw.get(k, 1.0)) * float(penalties.get(k, 0.0)) for k in penalties) / pw_sum
    return float(base) - float(pen)  # in [-1, 1], typically in [0,1] if pen <= base


def fitness(
    subscores: Dict[str, float],
    penalties: Dict[str, float],
    novelty: float,
    cfg: Optional[object] = None,
    novelty_coeff_default: float = 0.10,
) -> float:

    """weighted sum of subscores minus weighted penalties plus a small novelty bonus.
       This gives a single fitness number ∈ [0,1] for selection,
       with λ_novel configurable via cfg.fitness.novelty_coeff.
    """
    # λ_novel
    novelty_coeff = novelty_coeff_default
    if cfg and getattr(cfg, "fitness", None) and hasattr(cfg.fitness, "novelty_coeff"):
        novelty_coeff = float(cfg.fitness.novelty_coeff) # novelty in [0,1]; encourage diversity

    core = _weighted_subscores_minus_penalties(subscores, penalties, cfg)  # base - penalty
    raw = core + novelty_coeff * float(novelty)
    return max(0.0, min(1.0, float(raw)))