# duplicates/novelty are detected in this file,
# Build LSH index for all CV texts in a generation.
"""
Method: MinHash + LSH
how to run:
idx = lsh_index(samples=[], threshold=0.85, num_perm=512, shingle_k=5)
Insert as you generate CVs: idx.insert(sample_id, cv_text)
When scoring a new CV:
dup_sim = dup_similarity(idx, sample_id, cv_text)  # 1.0 if duplicate cluster hit
novelty = 1.0 - dup_sim
"""
# --- similarity.py (hybrid: datasketch if available, else builtin) ---

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set, Tuple, Optional, Any
import hashlib
import random
import re

# -------- try datasketch backend --------
_HAS_DATASKETCH = False
try:
    from datasketch import MinHash as DSMinHash, MinHashLSH as DSMinHashLSH

    _HAS_DATASKETCH = True
except Exception:
    _HAS_DATASKETCH = False


# ---------- shared helpers ----------
def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokenize_words(text: str) -> List[str]:
    # Simple word tokenizer: words & numbers
    return re.findall(r"[a-z0-9]+", text.lower())


def _k_shingles(tokens: List[str], k: int) -> Set[str]:
    if k <= 1:
        return set(tokens)
    return {" ".join(tokens[i: i + k]) for i in range(max(0, len(tokens) - k + 1))}


# ---------- datasketch backend ----------
@dataclass
class _DSIndex:
    threshold: float
    num_perm: int
    shingle_k: int
    seed: int = 1337
    lsh: Any = field(init=False)
    minhashes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # MinHashLSH threshold equals approximate Jaccard cutoff
        self.lsh = DSMinHashLSH(threshold=float(self.threshold), num_perm=int(self.num_perm))

    def _minhash_for_text(self, text: str) -> Any:
        mh = DSMinHash(num_perm=int(self.num_perm), seed=int(self.seed))
        for s in _k_shingles(_tokenize_words(_normalize_text(text)), self.shingle_k) or {"__EMPTY__"}:
            mh.update(s.encode("utf-8"))
        return mh

    def insert(self, sid: str, text: str) -> None:
        mh = self._minhash_for_text(text)
        self.minhashes[str(sid)] = mh
        self.lsh.insert(str(sid), mh)

    def query_candidates(self, text_or_mh: Any) -> List[str]:
        """Return candidate IDs that share at least one band bucket."""
        if hasattr(text_or_mh, "hashvalues"):
            mh = text_or_mh
        else:
            mh = self._minhash_for_text(str(text_or_mh))
        return list(self.lsh.query(mh))

    @staticmethod
    def est_jaccard(mh_a: Any, mh_b: Any) -> float:
        """Estimate Jaccard via signature agreement rate."""
        # datasketch exposes exact MinHash Jaccard estimate
        return float(mh_a.jaccard(mh_b))


# ---------- builtin backend ----------
# Large prime (fits in 64-bit). 2^61 - 1 is a Mersenne prime.
_LARGE_PRIME = (1 << 61) - 1


def _hash64(s: str) -> int:
    # Stable 64-bit hash via blake2b digest_size=8
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


def _make_permutations(num_perm: int, seed: int = 1337) -> List[Tuple[int, int]]:
    rnd = random.Random(seed)
    return [(rnd.randrange(1, _LARGE_PRIME - 1), rnd.randrange(0, _LARGE_PRIME - 1)) for _ in range(num_perm)]


def _minhash_signature(shingle_hashes: Iterable[int], perms: List[Tuple[int, int]]) -> List[int]:
    sig = []
    for a, b in perms:
        mn = _LARGE_PRIME - 1
        for h in shingle_hashes:
            # (a*h + b) % p
            v = (a * h + b) % _LARGE_PRIME
            if v < mn:
                mn = v
        sig.append(mn)
    return sig


def _select_bands(num_perm: int) -> Tuple[int, int]:
    """
        Choose (#bands, rows_per_band) so that bands * rows = num_perm.
        Defaults prefer 32x4 if possible; else 16x8; else fall back to 8x(num_perm/8).
        """
    for b, r in [(32, 4), (16, 8), (8, 16), (4, 32), (64, 2), (2, 64)]:
        if b * r == num_perm:
            return b, r
    # Fallback: use greatest divisor near 32
    for b in range(32, 1, -1):
        if num_perm % b == 0:
            return b, num_perm // b
    return num_perm, 1  # degenerate


@dataclass
class _BuiltinIndex:
    threshold: float
    num_perm: int
    shingle_k: int
    seed: int = 1337
    bands: int = field(init=False)
    rows_per_band: int = field(init=False)
    perms: List[Tuple[int, int]] = field(init=False)
    buckets: List[Dict[int, Set[str]]] = field(init=False)  # one dict per band
    signatures: Dict[str, List[int]] = field(default_factory=dict)

    def __post_init__(self):
        self.perms = _make_permutations(self.num_perm, self.seed)
        b, r = _select_bands(self.num_perm)
        self.bands, self.rows_per_band = b, r
        self.buckets = [dict() for _ in range(self.bands)]

    def _signature_for_text(self, text: str) -> List[int]:
        tokens = _tokenize_words(_normalize_text(text))
        # Avoid empty set: use a dummy shingle to keep signature defined
        shingles = _k_shingles(tokens, self.shingle_k) or {"__EMPTY__"}
        shash = [_hash64(s) for s in shingles]
        return _minhash_signature(shash, self.perms)

    def _band_hashes(self, sig: List[int]) -> List[int]:
        """Hash each band (tuple of rows) into an int key."""
        keys = []
        start = 0
        for _ in range(self.bands):
            end = start + self.rows_per_band
            chunk = sig[start:end]
            # Stable hash of the band (bytes of ints)
            h = hashlib.blake2b(b"".join(int(x).to_bytes(8, "big") for x in chunk), digest_size=8).digest()
            keys.append(int.from_bytes(h, "big"))
            start = end
        return keys

    def insert(self, sid: str, text: str) -> None:
        sig = self._signature_for_text(text)
        self.signatures[str(sid)] = sig
        for band_idx, key in enumerate(self._band_hashes(sig)):
            bucket = self.buckets[band_idx].setdefault(key, set())
            bucket.add(str(sid))

    def query_candidates_by_sig(self, sig: List[int]) -> Set[str]:
        cands: Set[str] = set()
        for band_idx, key in enumerate(self._band_hashes(sig)):
            bucket = self.buckets[band_idx].get(key)
            if bucket:
                cands.update(bucket)
        return cands

    @staticmethod
    def est_jaccard(sig_a: List[int], sig_b: List[int]) -> float:
        if not sig_a or not sig_b or len(sig_a) != len(sig_b):
            return 0.0
        eq = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
        return float(eq) / float(len(sig_a))


# ---------- public factory & API ----------
@dataclass
class LSHIndex:
    """Thin facade so callers don't care which backend is used."""
    backend: str
    impl: Any
    threshold: float
    num_perm: int
    shingle_k: int

    def insert(self, sid: str, text: str) -> None:
        self.impl.insert(sid, text)


def lsh_index(
        samples: Iterable[Tuple[str, str]],
        threshold: float = 0.85,
        num_perm: int = 512,
        shingle_k: int = 5,
        seed: int = 1337,
        backend: str = "auto",  # "auto" | "datasketch" | "builtin"
) -> LSHIndex:
    """
        Build an LSH index from (sid, text) samples.
        """
    use_ds = (backend == "datasketch") or (backend == "auto" and _HAS_DATASKETCH)
    if use_ds:
        idx_impl = _DSIndex(threshold=threshold, num_perm=num_perm, shingle_k=shingle_k, seed=seed)
    else:
        idx_impl = _BuiltinIndex(threshold=threshold, num_perm=num_perm, shingle_k=shingle_k, seed=seed)

    idx = LSHIndex(
        backend=("datasketch" if use_ds else "builtin"),
        impl=idx_impl,
        threshold=float(threshold),
        num_perm=int(num_perm),
        shingle_k=int(shingle_k),
    )
    for sid, text in samples:
        idx.insert(str(sid), text)
    return idx


def dup_similarity(
        lsh: LSHIndex,
        sid: str,
        text: str,
        threshold: Optional[float] = None,
) -> float:
    """
    Return the maximum estimated Jaccard similarity between `text` and any
    candidate already in the LSH index (excluding `sid`), using MinHash-LSH.

    Semantics for your pipeline:
      - If any candidate ≥ threshold, return 1.0  (treat as duplicate cluster hit).
      - Else return the maximum estimated similarity in [0, 1).

    purpose: flags 1.0 if the CV is very similar to an existing one (same cluster), else 0.0.
    """
    th = float(threshold if threshold is not None else lsh.threshold)

    if lsh.backend == "datasketch":
        # datasketch path
        impl: _DSIndex = lsh.impl  # type: ignore
        mh_q = impl._minhash_for_text(text)
        cands = [c for c in impl.query_candidates(mh_q) if str(c) != str(sid)]
        pool = cands if cands else [c for c in impl.minhashes.keys() if str(c) != str(sid)]  # fallback scan
        best = 0.0
        for cid in pool:
            mh_c = impl.minhashes.get(cid)
            if mh_c is None:
                continue
            est = _DSIndex.est_jaccard(mh_q, mh_c)  # max estimated Jaccard similarity
            if est >= th:
                return 1.0
            if est > best:
                best = est
        return float(best)

    else:
        # builtin path
        impl: _BuiltinIndex = lsh.impl  # type: ignore
        sig_q = impl._signature_for_text(text)
        cands = {c for c in impl.query_candidates_by_sig(sig_q) if str(c) != str(sid)}
        best = 0.0
        for cid in cands:
            sig_c = impl.signatures.get(cid)
            if not sig_c:
                continue
            est = _BuiltinIndex.est_jaccard(sig_q, sig_c)
            if est >= th:
                return 1.0
            if est > best:
                best = est
        return float(best)
