# Run CLI -> python -m src.pipeline.generate_epoch --output data/synth/raw --generation 0
# or with Env override -> APP_SEED=123 APP_GENERATION_PRIME=10009 python -m src.pipeline.generate_epoch --output data/synth/raw --generation 0
import hashlib
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

# NOTE: adapt these to your agent_orchestrator functions
from src.agents.agent_orchestrator import generate_personas, generate_jobs, tailor_cv, score_cv_committee
from src.config.settings import load_cfg
from src.schemas.entities import Persona, JobJD, CVDoc, RecruiterScore, SampleRecord
from src.scoring.fitness import aggregate_committee, fitness
from src.utils.run_ctx import mlflow_run, set_seed
from src.utils.similarity import lsh_index, dup_similarity
from src.utils.storage import save_jsonl  # implement simple writer

cfg = load_cfg()  # or load_cfg(cli_seed=args.seed)
set_seed(cfg.seed)  # your existing helper
MAX_WORKERS = int(os.getenv("GENAI_MAX_CONCURRENCY", "8"))


def run_generation(output_dir: str, generation: int, top_k: float = 0.20):
    set_seed(cfg.seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    personas = [Persona.model_validate(p) for p in generate_personas(generation)]
    rng = random.Random(cfg.seed * 10 + generation)
    rng.shuffle(personas)
    jobs = [JobJD.model_validate(j) for j in generate_jobs(generation)]

    # ------ serial code ----
    # samples, sigs = [], []
    # # draft CVs
    # for jd in jobs:
    #     for pe in personas:
    #         cv_raw = tailor_cv(pe, jd)  # returns dict; ensure keys match CVDoc
    #         cv = CVDoc.model_validate(cv_raw)
    #         judges = score_cv_committee(cv, jd)  # list of dicts with subscores
    #         rs = [RecruiterScore.model_validate(s) for s in judges]
    #         subs = aggregate_committee([r.subscores for r in rs])
    #
    #         samples.append((cv, jd, pe, rs, subs))
    #         sigs.append((cv.id, cv.raw_text))
    # ----

    # ------ parallel version code ----
    def _build_sample(pe, jd):
        cv_raw = tailor_cv(pe, jd)
        cv = CVDoc.model_validate(cv_raw)
        judges = score_cv_committee(cv, jd)
        rs = [RecruiterScore.model_validate(s) for s in judges]
        subs = aggregate_committee([r.subscores for r in rs])
        return (cv, jd, pe, rs, subs), (cv.id, cv.raw_text)

    samples, sigs = [], []
    errors = 0
    # deterministic one-time shuffle (prevents same item always going first)
    _rng = random.Random(f"permute|{generation}|{cfg.seed}")
    jobs_shuf = jobs[:]
    _rng.shuffle(jobs_shuf)
    personas_shuf = personas[:]
    _rng.shuffle(personas_shuf)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(_build_sample, pe, jd) for jd in jobs_shuf for pe in personas_shuf]
        for fut in as_completed(futs):
            try:
                sample, sig = fut.result()
                samples.append(sample)
                sigs.append(sig)
            except Exception as e:
                import traceback
                print("[parallel] task failed:", repr(e))
                traceback.print_exc()  # <-- shows exact file:line

    # optional: stable output order
    samples.sort(key=lambda t: t[0].id)
    sigs.sort(key=lambda s: s[0])

    # build LSH for novelty/dup checks
    lsh = lsh_index([(sid, txt) for sid, txt in sigs], threshold=0.35)

    # compute penalties + fitness
    records = []
    for cv, jd, pe, rs, subscores in samples:
        penalties = {
            "credential_violation": float("phd" in cv.raw_text.lower() and "phd" not in pe.core_story.lower()),
            "date_incoherence": 0.0,  # TODO: implement date checker
            "keyword_stuffing": float(sum(cv.raw_text.lower().count(t.lower()) for t in jd.must_haves) > 30),
            # Use a non-triggerable threshold so we always get the continuous max est. similarity
            "dup_similarity": dup_similarity(lsh, cv.id, cv.raw_text, threshold=1.01)
        }
        novelty = round(1.0 - float(penalties["dup_similarity"]), 4)  # so duplicates get lower fitness.
        fit = float(fitness(subscores, penalties, novelty))  # ensure float if fitness returned string

        rec = SampleRecord(
            generation=generation, cv=cv, jd=jd, persona=pe, scores=rs,
            meta={"fitness": round(fit, 4), "novelty": round(novelty, 5)}
        )
        records.append((fit, rec))
        # existing shuffle + jitter stay as-is
        rng = random.Random(f"tiebrk|{generation}|{cfg.seed}")
        rng.shuffle(records)

        def _jitter(cv_id: str) -> float:
            h = int(hashlib.sha256(cv_id.encode()).hexdigest(), 16)
            return (h % 1000) / 1e7

        # Optional fairness: pick top-1 per JD with a fitness floor
        if os.getenv("PER_JD_TOP", "0") == "1":
            from collections import defaultdict
            floor = float(os.getenv("FITNESS_FLOOR", "0.0"))
            by_jd = defaultdict(list)
            for f, r in records:
                by_jd[r.jd.id].append((f, r))
            survivors = []
            for _, lst in by_jd.items():
                lst.sort(key=lambda x: (x[0], float(x[1].meta.get("novelty", 0.0))), reverse=True)
                if lst and lst[0][0] >= floor:
                    survivors.append(lst[0][1])
        else:
            # select survivors (top_k) and persist ALL + survivors
            rng = random.Random(f"tiebrk|{generation}|{cfg.seed}")
            tie_jit = {r.cv.id: rng.random() for _, r in records}  # one tiny jitter per CV (reproducible per gen)
            EPS = float(os.getenv("FITNESS_TIE_EPS", "0.02"))  # treat scores within 0.02 as "near-ties"

            def sort_key(item):
                fit, rec = item
                nov = float(rec.meta.get("novelty", 0.0))
                # round fitness into EPS buckets so near-ties prefer higher novelty, then jitter
                fit_bucket = round(fit / EPS, 0)
                return fit_bucket, nov, tie_jit[rec.cv.id]

            records.sort(key=sort_key, reverse=True)
            survivors = [r for _, r in records[: max(1, int(len(records) * top_k))]]

    # write artifacts
    save_jsonl(out / f"gen_{generation}_all.jsonl", [r.model_dump(mode="json") for _, r in records])
    save_jsonl(out / f"gen_{generation}_survivors.jsonl", [r.model_dump(mode="json") for r in survivors])

    # --- committee dispersion (L1 distance across judges' subscores) ---
    def _l1(a: dict, b: dict) -> float:
        keys = set(a) | set(b)
        return sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in keys)

    cl1 = []
    for _, rec in records:  # records is List[Tuple[fitness: float, SampleRecord]]
        # each rec.scores is List[RecruiterScore]; take their subscores dicts
        subs = [s.subscores for s in rec.scores]
        for i in range(len(subs)):
            for j in range(i + 1, len(subs)):
                cl1.append(_l1(subs[i], subs[j]))

    # optional: crude parse_fail_rate proxy (expected pairs vs realized samples)
    n_expected = len(personas) * len(jobs)
    parse_fail_rate = 1.0 - (len(samples) / max(1, n_expected))

    # MLflow logging
    with mlflow_run(name=f"gen-{generation}", tags={"stage": "generation"}):
        import mlflow
        mlflow.log_metric("n_personas", len(personas))
        mlflow.log_metric("n_jobs", len(jobs))
        mlflow.log_metric("n_cv", len(samples))
        if records:
            mlflow.log_metric(
                "dup_rate",
                sum(float(r.meta["novelty"]) < 0.5 for _, r in records) / len(records)
            )
            mlflow.log_metric(
                "fitness_p95",
                sorted([f for f, _ in records])[-max(1, int(0.05 * len(records)))]
            )
        else:
            mlflow.log_metric("dup_rate", 0.0)
            mlflow.log_metric("fitness_p95", 0.0)

        if cl1:
            mlflow.log_metric("committee_l1_mean", mean(cl1))  # diversity of judges’ views (higher = less agreement).
        else:
            mlflow.log_metric("committee_l1_mean", 0.0)
        mlflow.log_metric("parse_fail_rate",
                          parse_fail_rate)  # rough health check (you can refine later if you add try/except around parsing).
        mlflow.log_metric("parallel_errors", errors)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--top_k", type=float, default=0.20)
    parser.add_argument("--PRINT_TOKENS_COUNT", type=int, default=0,
                        help="1 - if you want to print the token count per call")
    parser.add_argument("--FITNESS_FLOOR", type=float, default=0.0)  # PER_JD_TOP=1 FITNESS_FLOOR=0.25
    parser.add_argument("--PER_JD_TOP", type=int, default=1)  # else 0
    args = parser.parse_args()

    run_generation(output_dir=args.output, generation=args.generation, top_k=args.top_k)
