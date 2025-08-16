# Run CLI -> python -m src.pipeline.generate_epoch --output data/synth/raw --generation 0
# or with Env override -> APP_SEED=123 APP_GENERATION_PRIME=10009 python -m src.pipeline.generate_epoch --output data/synth/raw --generation 0

from typing import List, Tuple
from pathlib import Path
import json, uuid
from pydantic import ValidationError

from src.schemas.entities import Persona, JobJD, CVDoc, RecruiterScore, SampleRecord
from src.scoring.fitness import aggregate_committee, fitness
from src.utils.similarity import lsh_index, dup_similarity
from src.utils.run_ctx import mlflow_run, set_seed
from src.utils.storage import save_jsonl  # implement simple writer
from src.config.settings import load_cfg

# NOTE: adapt these to your agent_orchestrator functions
from src.agents.agent_orchestrator import generate_personas, generate_jobs, tailor_cv, score_cv_committee

cfg = load_cfg()  # or load_cfg(cli_seed=args.seed)
set_seed(cfg.seed)  # your existing helper


def run_generation(output_dir: str, generation: int, top_k: float = 0.20):
    set_seed(cfg.seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    personas = [Persona.model_validate(p) for p in generate_personas(generation)]
    jobs = [JobJD.model_validate(j) for j in generate_jobs(generation)]

    samples, sigs = [], []
    # draft CVs
    for jd in jobs:
        for pe in personas:
            cv_raw = tailor_cv(pe, jd)  # returns dict; ensure keys match CVDoc
            cv = CVDoc.model_validate(cv_raw)
            judges = score_cv_committee(cv, jd)  # list of dicts with subscores
            rs = [RecruiterScore.model_validate(s) for s in judges]
            subs = aggregate_committee([r.subscores for r in rs])

            samples.append((cv, jd, pe, rs, subs))
            sigs.append((cv.id, cv.raw_text))

    # build LSH for novelty/dup checks
    lsh = lsh_index([(sid, txt) for sid, txt in sigs], threshold=0.85)

    # compute penalties + fitness
    records = []
    for cv, jd, pe, rs, subs in samples:
        penalties = {
            "credential_violation": float("phd" in cv.raw_text.lower() and "phd" not in pe.core_story.lower()),
            "date_incoherence": 0.0,  # TODO: implement date checker
            "keyword_stuffing": float(sum(cv.raw_text.lower().count(t.lower()) for t in jd.must_haves) > 30),
            "dup_similarity": dup_similarity(lsh, cv.id, cv.raw_text),
        }
        novelty = 1.0 - penalties["dup_similarity"]  # so duplicates get lower fitness.
        fit = fitness(subs, penalties, novelty)
        rec = SampleRecord(
            generation=generation, cv=cv, jd=jd, persona=pe, scores=rs,
            meta={"fitness": f"{fit:.4f}", "novelty": f"{novelty:.3f}"}
        )
        records.append((fit, rec))

    # select survivors (top_k) and persist ALL + survivors
    records.sort(key=lambda x: x[0], reverse=True)
    survivors = [r for _, r in records[: max(1, int(len(records) * top_k))]]

    # write artifacts
    save_jsonl(out / f"gen_{generation}_all.jsonl", [r.model_dump(mode="json") for _, r in records])
    save_jsonl(out / f"gen_{generation}_survivors.jsonl", [r.model_dump(mode="json") for r in survivors])

    # MLflow logging
    with mlflow_run(name=f"gen-{generation}", tags={"stage": "generation"}):
        import mlflow
        mlflow.log_metric("n_personas", len(personas))
        mlflow.log_metric("n_jobs", len(jobs))
        mlflow.log_metric("n_cv", len(samples))
        mlflow.log_metric(
            "dup_rate",
            sum(float(r.meta["novelty"]) < 0.5 for _, r in records) / len(records)
        )
        if records:
            mlflow.log_metric("fitness_p95", sorted([f for f, _ in records])[-max(1, int(0.05 * len(records)))])
