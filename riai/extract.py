"""
Topic extraction from raw text titles.
Pipeline: title (already clean for Wikipedia) -> spaCy NER -> RAKE/YAKE fallback.
Returns a list of candidate topic strings, ranked by confidence.
"""
from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)

_nlp = None  # lazy-loaded


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except Exception as exc:
            log.warning("spaCy load failed (%s); NER disabled", exc)
            _nlp = False
    return _nlp if _nlp else None


_STOP_ENTITIES = frozenset({
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
    "BBC", "Reuters", "AP", "CNN", "Fox", "NPR",
})

_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "EVENT", "FAC", "PRODUCT", "WORK_OF_ART", "LAW", "NORP"}


def extract_topics(title: str, source: str = "unknown") -> list[dict[str, Any]]:
    """
    Extract candidate topics from a title string.
    Returns list of {"text": str, "method": str, "confidence": float}.
    Ordered from highest to lowest confidence.
    """
    title = _clean(title)
    if not title:
        return []

    candidates: list[dict[str, Any]] = []

    # 1. Wikipedia titles are already canonical — use directly at max confidence
    if source == "wikipedia":
        candidates.append({"text": title, "method": "wikipedia_title", "confidence": 1.0})
        return candidates

    # 2. spaCy NER
    nlp = _get_nlp()
    if nlp:
        try:
            doc = nlp(title[:512])  # cap length for speed
            for ent in doc.ents:
                if ent.label_ not in _ENTITY_LABELS:
                    continue
                text = ent.text.strip()
                if len(text) < 2 or text in _STOP_ENTITIES:
                    continue
                candidates.append({
                    "text": text,
                    "method": f"spacy:{ent.label_}",
                    "confidence": 0.85,
                })
        except Exception as exc:
            log.debug("spaCy extraction error: %s", exc)

    # 3. YAKE keyword extraction fallback (no heavy deps)
    try:
        kws = _yake_extract(title)
        for kw in kws:
            if not any(c["text"].lower() == kw.lower() for c in candidates):
                candidates.append({"text": kw, "method": "yake", "confidence": 0.6})
    except Exception as exc:
        log.debug("YAKE extraction error: %s", exc)

    # 4. If nothing found, use the cleaned title itself
    if not candidates:
        candidates.append({"text": title, "method": "title_fallback", "confidence": 0.4})

    return candidates


def _clean(text: str) -> str:
    """Strip HTML tags and normalise whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _yake_extract(text: str, top_n: int = 3) -> list[str]:
    """
    Minimal YAKE-style extraction using co-occurrence frequencies.
    Not the full YAKE algorithm, but a cheap approximation that avoids
    adding the yake package dependency.
    """
    try:
        import yake
        kw_extractor = yake.KeywordExtractor(
            lan="en", n=3, dedupLim=0.7, top=top_n, features=None
        )
        keywords = kw_extractor.extract_keywords(text)
        return [kw for kw, _score in keywords]
    except ImportError:
        pass

    # Manual fallback: extract capitalized noun phrases
    tokens = text.split()
    phrases: list[str] = []
    current: list[str] = []
    for tok in tokens:
        if tok[0].isupper() and re.match(r"^[A-Z][a-zA-Z\-']+$", tok):
            current.append(tok)
        else:
            if len(current) >= 1:
                phrases.append(" ".join(current))
            current = []
    if current:
        phrases.append(" ".join(current))

    # Deduplicate, prefer longer phrases
    seen: set[str] = set()
    result: list[str] = []
    for p in sorted(phrases, key=len, reverse=True):
        if p.lower() not in seen and len(p) > 2:
            seen.add(p.lower())
            result.append(p)
        if len(result) >= top_n:
            break
    return result
