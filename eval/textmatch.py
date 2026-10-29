"""
Fuzzy text matching used to decide whether a retrieved chunk contains gold evidence.

Each pipeline chunks the paper differently (and parses it with a different
library), so exact string matching is useless. Instead we compare sets of word
trigrams after normalisation (NFKC, lowercase, de-hyphenation, punctuation
stripped). A chunk "contains" an evidence paragraph when enough of the trigrams
overlap — measured against the SMALLER of the two, so a chunk that is a fragment
of a long evidence paragraph still counts, and a long chunk that wraps a short
paragraph does too.
"""

from __future__ import annotations

import re
import unicodedata

RELEVANCE_THRESHOLD = 0.5

_WORD = re.compile(r"[a-z0-9]+")
_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")


def normalize_words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    return _WORD.findall(text.lower())


# QASPER's text came from LaTeX source: citations, inline math and cross-refs are
# replaced by placeholders (BIBREF10, INLINEFORM0, …) that never appear in the
# PDF. Treat them as breaks so no trigram spans one.
_PLACEHOLDER = re.compile(r"^(inlineform|displayform|bibref|secref|tabref|figref)\d*$")


def trigrams(text: str) -> set[tuple[str, str, str]]:
    grams: set[tuple[str, str, str]] = set()
    seg: list[str] = []
    for w in normalize_words(text) + ["inlineform"]:
        if _PLACEHOLDER.match(w):
            grams.update((seg[i], seg[i + 1], seg[i + 2]) for i in range(len(seg) - 2))
            seg = []
        else:
            seg.append(w)
    return grams


def match_score(
    evidence: str,
    text: str,
    ev_grams: set | None = None,
    text_grams: set | None = None,
    contain_only: bool = False,
) -> float:
    """
    contain_only=True  → share of the evidence found in `text` (used to locate
                         evidence on a PDF page).
    contain_only=False → overlap relative to the smaller side (used to judge a
                         retrieved chunk against an evidence paragraph).
    """
    e = ev_grams if ev_grams is not None else trigrams(evidence)
    t = text_grams if text_grams is not None else trigrams(text)
    if not e or not t:
        return 0.0
    inter = len(e & t)
    denom = len(e) if contain_only else min(len(e), len(t))
    return inter / denom


MIN_SHARED = 10  # a tiny chunk can't be "relevant" on a handful of shared trigrams


def is_relevant(chunk_grams: set, evidence_grams: set) -> bool:
    if not chunk_grams or not evidence_grams:
        return False
    inter = len(chunk_grams & evidence_grams)
    if inter < MIN_SHARED and inter < 0.9 * len(evidence_grams):
        return False
    return inter / min(len(chunk_grams), len(evidence_grams)) >= RELEVANCE_THRESHOLD
