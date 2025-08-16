"""
“run context” helpers. It gives us two tiny utilities you wrap around a run
so it’s repeatable and logged:
set_seed(...) → makes randomness deterministic.
mlflow_run(...) → a context manager that opens/closes an MLflow run so you can log metrics/artifacts cleanly with mlflow.log_metric(...).
"""
import os, random, numpy as np
from contextlib import contextmanager
import mlflow


def set_seed(seed: int = 42):
    # makes randomness repeatable with seed(reproducibility)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@contextmanager
def mlflow_run(name: str, tags: dict = None):
    mlflow.set_experiment("synthetic_cv_generation")
    with mlflow.start_run(run_name=name):  # name -> e.g., gen-0, gen-1, ...
        # ... generate JDs, CVs, scores ...
        if tags:  # e.g., {"stage": "generation", ...}
            [mlflow.set_tag(k, v) for k, v in tags.items()]
            # mlflow.log_metric("n_cv", len(cvs))
            # mlflow.log_metric("dup_rate", dup_rate)
            # mlflow.log_artifact("data/synth/raw/gen_0_all.jsonl")
        yield
