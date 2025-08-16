# duplicates/novelty are detected in this file,
# Build LSH index for all CV texts in a generation.
"""
Method: MinHash + LSH

"""

import re
from datasketch import MinHash, MinHashLSH

_WORD = re.compile(r"[A-Za-z0-9]+")


def shingles(text: str, k: int = 5):
    toks = _WORD.findall(text.lower())
    return {" ".join(toks[i:i + k]) for i in range(max(1, len(toks) - k + 1))}


def minhash_sig(text: str, num_perm: int = 128) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    for s in shingles(text):
        mh.update(s.encode("utf-8"))
    return mh


def lsh_index(samples, threshold=0.85, num_perm=128):
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for sid, txt in samples:
        mh = minhash_sig(txt, num_perm)
        lsh.insert(sid, mh)
    return lsh


def dup_similarity(lsh, sid: str, text: str, num_perm=128) -> float:
    """flags 1.0 if the CV is very similar to an existing one (same cluster), else 0.0."""
    mh = minhash_sig(text, num_perm)
    cands = lsh.query(mh)
    return 1.0 if any(c != sid for c in cands) else 0.0
    # penalty flag; or compute max Jaccard est
