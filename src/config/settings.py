import os, yaml
from pydantic import BaseModel, ValidationError
from typing import Dict


class RetrievalCfg(BaseModel):
    top_k: int = 5


class FitnessCfg(BaseModel):
    novelty_coeff: float = 0.10
    # optional: expose weights later if you want to A/B
    # subscore_weights: Dict[str, float] = {}
    # penalty_weights: Dict[str, float]  = {}


class DedupCfg(BaseModel):
    lsh_threshold: float = 0.85
    shingle_k: int = 5
    num_perm: int = 128


class AppCfg(BaseModel):
    seed: int
    generation_prime: int
    retrieval: RetrievalCfg = RetrievalCfg()
    fitness: FitnessCfg = FitnessCfg()
    dedup: DedupCfg = DedupCfg()


def load_cfg(path: str = "configs/app.yaml", cli_seed: int | None = None) -> AppCfg:
    data = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    # ENV overrides (no defaults)
    env = {}
    if "APP_SEED" in os.environ:
        env["seed"] = int(os.environ["APP_SEED"])
    if "APP_GENERATION_PRIME" in os.environ:
        env["generation_prime"] = int(os.environ["APP_GENERATION_PRIME"])

    # CLI override (highest precedence)
    if cli_seed is not None:
        env["seed"] = int(cli_seed)

    merged = {**data, **env}
    try:
        return AppCfg.model_validate(merged)
    except ValidationError as e:
        raise RuntimeError(
            f"Missing required config keys in {path}. "
            f"Provide 'seed' and 'generation_prime' in YAML, or override via ENV "
            f"(APP_SEED, APP_GENERATION_PRIME) / CLI."
        ) from e
