# src/utils/cache.py
import hashlib
from pathlib import Path
import os, time, json
import threading, uuid, tempfile

_DEBUG_N = int(os.getenv("DEBUG_SAMPLE_N", "5"))
_DEBUG_CNT = 0
_MAX = int(os.getenv("GENAI_MAX_CONCURRENCY", "8"))
_SEM = threading.BoundedSemaphore(_MAX)


def debug_dump(agent: str, text: str, parsed_json: dict | None, finish_reason=None):
    global _DEBUG_CNT
    if os.getenv("DEBUG_GEN") != "1" or _DEBUG_CNT >= _DEBUG_N:
        return
    os.makedirs("logs/raw", exist_ok=True)
    stamp = int(time.time() * 1000)
    ok = bool(parsed_json)
    path = f"logs/raw/{stamp}_{agent}_ok{int(ok)}_fin{finish_reason or 'NA'}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
        if parsed_json is not None:
            f.write("\n\n---PARSED_JSON_PREVIEW---\n")
            f.write(json.dumps(parsed_json, ensure_ascii=False)[:2000])
    _DEBUG_CNT += 1


CACHE_DIR = Path(".cache/agents")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _resp_text(resp) -> str:
    if resp is None:
        return ""
    try:
        return resp.text or ""  # works only when finish_reason == STOP and parts exist
    except Exception:
        pass
    texts = []
    for cand in getattr(resp, "candidates", []) or []:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) if content else None
        if parts:
            for p in parts:
                t = getattr(p, "text", None)
                if t:
                    texts.append(t)
    # --- NEW: write a small diag so we know why it's empty ---
    os.makedirs("logs/raw", exist_ok=True)

    # Pull safe fields from prompt_feedback
    pf = getattr(resp, "prompt_feedback", None)
    pf_block = getattr(pf, "block_reason", None) if pf else None
    pf_safety = []
    if pf:
        for sr in (getattr(pf, "safety_ratings", []) or []):
            pf_safety.append({
                "category": str(getattr(sr, "category", None)),
                "probability": str(getattr(sr, "probability", None)),
            })

    diag = {
        "ts_ms": int(time.time() * 1000),
        "finish_reasons": [getattr(c, "finish_reason", None) for c in (getattr(resp, "candidates", []) or [])],
        "prompt_block_reason": str(pf_block) if pf_block is not None else None,
        "prompt_safety": pf_safety,
    }

    with open("logs/raw/_diag.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(diag, ensure_ascii=False, default=str) + "\n")

    print(f"[genai-diag] finish={diag['finish_reasons']} block={pf_block}")
    return ("\n".join(texts) or "").strip()


def _key(model_name: str, prompt: str, gen_cfg: dict | None) -> Path:
    h = hashlib.sha256()
    h.update(model_name.encode());
    h.update(b"\n")
    h.update(prompt.encode());
    h.update(b"\n")
    if gen_cfg: h.update(json.dumps(gen_cfg, sort_keys=True).encode())
    return CACHE_DIR / (h.hexdigest() + ".json")


def call_with_cache(model, prompt: str, generation_config: dict | None = None, retries=3, backoff=1.5):
    label = getattr(model, "model_name", None) or getattr(model, "_cache_model_label", None) or "unknown"
    p = _key(label, prompt, generation_config)
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        data.setdefault("cached", True)
        # diag
        os.makedirs("logs/raw", exist_ok=True)
        with open("logs/raw/_diag.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts_ms": int(time.time() * 1000),
                "source": "cache",
                "text_len": len((data.get("text") or "")),
                "finish_reason": "CACHE"
                }) + "\n")
        return data
    # simple retry
    delay = 0.8
    for a in range(retries):
        try:
            with _SEM:
                resp = (model.generate_content(prompt, generation_config=generation_config)
                        if generation_config else model.generate_content(prompt))
            text = (_resp_text(resp) or "").strip()

            # diag + build + cache (single version)
            os.makedirs("logs/raw", exist_ok=True)
            finish = getattr(resp.candidates[0], "finish_reason", None) if getattr(resp, "candidates", None) else None
            data = {"text": text, "finish_reason": finish, "cached": False}
            # only cache non-empty responses
            if (text or "").strip():
                tmp = Path(str(p) + f".tmp.{uuid.uuid4().hex}")
                tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                os.replace(tmp, p)
            return data

        except Exception as e:
            if a == retries - 1: raise
            time.sleep(delay);
            delay *= backoff
