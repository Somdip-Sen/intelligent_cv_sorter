"""
Purpose: add "style DNA" to each persona + simple mutate/crossover for evolution.
"""
from __future__ import annotations
import random
from typing import Dict, List

STYLE_CHOICES: Dict[str, List[str]] = {
    "tone": ["formal", "concise", "story"],
    "layout": ["ats", "two_column", "narrative"],
    "bullet_len": ["short", "mixed", "long"],
    "metrics_density": ["low", "med", "high"],
    "skills_format": ["inline", "table", "taglist"],
}


def gen_style_profile(rng: random.Random) -> Dict[str, str]:
    """Sample a fresh style profile."""
    return {k: rng.choice(v) for k, v in STYLE_CHOICES.items()}


def mutate_style(profile: Dict[str, str], rng: random.Random, rate: float = 0.2) -> Dict[str, str]:
    """Flip ~rate fraction of style knobs to a different valid value."""
    out = dict(profile)
    for k, options in STYLE_CHOICES.items():
        if rng.random() < rate:
            choices = [o for o in options if o != out[k]]
            out[k] = rng.choice(choices)
    return out


def crossover_style(a: Dict[str, str], b: Dict[str, str], rng: random.Random) -> Dict[str, str]:
    """Child takes a random subset from parent A/B (no invalid combos)."""
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in STYLE_CHOICES}

# ------------------------------------------------------------
# Usage integration (minimal):

# 1) When creating personas (e.g., in agent_orchestrator.generate_personas):
# rng = random.Random(seed + generation)
# persona["style_profile"] = gen_style_profile(rng)

# 2) When tailoring a CV (in tailor_cv):
# candidate_inputs["style_profile"] = persona["style_profile"]  # pass through to the prompt
# (Prompt side: “Honor style_profile: tone/layout/bullet_len/metrics_density/skills_format.”)

# 3) When breeding next generation (pseudo):
# children = []
# for mom, dad in selected_pairs:
#     child_style = mutate_style(crossover_style(mom["style_profile"], dad["style_profile"], rng), rng, rate=0.2)
#     child = {**template_persona_from(mom, dad), "style_profile": child_style}
#     children.append(child)
