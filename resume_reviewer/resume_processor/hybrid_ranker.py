"""
Hybrid Retrieve-and-Rerank Pipeline
====================================
BM25 (lexical) → SBERT (semantic) → RRF fusion → Cross-Encoder reranking.

Design decisions (from thesis audit):
  • CE score only — no penalty gates, no boosts.
  • Two-tier scoring: top-K get CE reranking, rest keep RRF score.
  • Pure IR pipeline: no skill-matching heuristics.

Public API
----------
    ranker = HybridRanker()          # lazy-loads models
    ranker.rank(parsed_resumes,      # List[ParsedResume]
                job_description,     # raw JD text
                top_k=10)            # how many to CE-rerank
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RRF_K = 60                # Reciprocal Rank Fusion constant
CE_TOP_K_DEFAULT = 10     # Candidates sent to Cross-Encoder
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
CE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ---------------------------------------------------------------------------
# Lazy singleton helpers (avoid re-loading 300 MB of weights every call)
# ---------------------------------------------------------------------------
_sbert_model = None
_ce_model = None


def _get_sbert():
    """Lazy-load the SBERT bi-encoder (singleton)."""
    global _sbert_model
    if _sbert_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"[HYBRID] Loading SBERT model: {SBERT_MODEL_NAME}")
            _sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
            logger.info("[HYBRID] SBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"[HYBRID] SBERT unavailable: {e}")
            _sbert_model = None
    return _sbert_model


def _get_ce():
    """Lazy-load the Cross-Encoder (singleton)."""
    global _ce_model
    if _ce_model is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"[HYBRID] Loading Cross-Encoder model: {CE_MODEL_NAME}")
            _ce_model = CrossEncoder(CE_MODEL_NAME)
            logger.info("[HYBRID] Cross-Encoder model loaded successfully")
        except Exception as e:
            logger.warning(f"[HYBRID] Cross-Encoder unavailable: {e}")
            _ce_model = None
    return _ce_model


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _resume_to_text(resume) -> str:
    """Build a single flat text string from a ParsedResume for retrieval."""
    parts: list[str] = []
    for key in ("experience", "skills", "education", "misc"):
        section_text = (resume.sections.get(key) or "").strip()
        if section_text:
            parts.append(section_text)
    # Also include raw_text if sections are empty (fallback parses)
    if not parts and isinstance(resume.parsed, dict):
        raw = (resume.parsed.get("raw_text") or "").strip()
        if raw:
            parts.append(raw)
    return " ".join(parts)


def _tokenize(text: str) -> List[str]:
    """Whitespace + punctuation tokenizer for BM25."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


# ---------------------------------------------------------------------------
# Stage 1: BM25 retrieval
# ---------------------------------------------------------------------------

def _bm25_rank(
    corpus_tokens: List[List[str]],
    query_tokens: List[str],
) -> List[Tuple[int, float]]:
    """
    Return (index, score) pairs sorted descending by BM25 score.

    Uses rank_bm25.BM25Okapi if available, otherwise falls back to a
    minimal pure-Python implementation so the pipeline never hard-fails.
    """
    n = len(corpus_tokens)
    if n == 0:
        return []

    try:
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi(corpus_tokens)
        scores = bm25.get_scores(query_tokens)
    except ImportError:
        logger.warning("[HYBRID] rank_bm25 not installed — using built-in BM25")
        scores = _bm25_fallback(corpus_tokens, query_tokens)

    indexed = [(i, float(scores[i])) for i in range(n)]
    indexed.sort(key=lambda x: x[1], reverse=True)
    return indexed


def _bm25_fallback(
    corpus_tokens: List[List[str]],
    query_tokens: List[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> np.ndarray:
    """Minimal BM25-Okapi for environments without rank_bm25."""
    n = len(corpus_tokens)
    dl = np.array([len(doc) for doc in corpus_tokens], dtype=float)
    avgdl = dl.mean() if n > 0 else 1.0

    # Document frequency
    df: Dict[str, int] = {}
    for doc in corpus_tokens:
        for t in set(doc):
            df[t] = df.get(t, 0) + 1

    scores = np.zeros(n, dtype=float)
    for q in query_tokens:
        if q not in df:
            continue
        idf = math.log((n - df[q] + 0.5) / (df[q] + 0.5) + 1.0)
        for i, doc in enumerate(corpus_tokens):
            tf = doc.count(q)
            denom = tf + k1 * (1.0 - b + b * dl[i] / avgdl)
            scores[i] += idf * (tf * (k1 + 1.0)) / denom if denom > 0 else 0.0
    return scores


# ---------------------------------------------------------------------------
# Stage 2: SBERT retrieval
# ---------------------------------------------------------------------------

def _sbert_rank(
    resume_texts: List[str],
    jd_text: str,
) -> List[Tuple[int, float]]:
    """
    Rank resumes by cosine similarity to the JD using SBERT.

    Returns (index, score) pairs sorted descending.
    Falls back to zero scores if model is unavailable.
    """
    n = len(resume_texts)
    if n == 0:
        return []

    model = _get_sbert()
    if model is None:
        logger.warning("[HYBRID] SBERT unavailable — all semantic scores = 0")
        return [(i, 0.0) for i in range(n)]

    try:
        all_texts = [jd_text] + resume_texts
        embeddings = model.encode(all_texts, show_progress_bar=False, normalize_embeddings=True)
        jd_emb = embeddings[0]
        resume_embs = embeddings[1:]
        # Cosine similarity (embeddings already L2-normalised)
        similarities = resume_embs @ jd_emb
        indexed = [(i, float(similarities[i])) for i in range(n)]
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed
    except Exception as e:
        logger.error(f"[HYBRID] SBERT encoding failed: {e}")
        return [(i, 0.0) for i in range(n)]


# ---------------------------------------------------------------------------
# Stage 3: Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _rrf_fuse(
    *rankings: List[Tuple[int, float]],
    k: int = RRF_K,
) -> List[Tuple[int, float]]:
    """
    Fuse multiple rankings via Reciprocal Rank Fusion.

    Each ranking is a list of (index, score) sorted descending.
    Returns (index, rrf_score) sorted descending.
    """
    rrf_scores: Dict[int, float] = {}
    for ranking in rankings:
        for rank_pos, (idx, _score) in enumerate(ranking):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank_pos + 1)

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused


# ---------------------------------------------------------------------------
# Stage 4: Cross-Encoder reranking
# ---------------------------------------------------------------------------

def _cross_encoder_rerank(
    resume_texts: List[str],
    jd_text: str,
    candidate_indices: List[int],
) -> List[Tuple[int, float]]:
    """
    Rerank a subset of candidates using the Cross-Encoder.

    Args:
        resume_texts: full corpus of resume texts (indexed by position)
        jd_text: job description text
        candidate_indices: which indices to rerank

    Returns:
        (index, ce_score) sorted descending by CE score.
    """
    if not candidate_indices:
        return []

    model = _get_ce()
    if model is None:
        logger.warning("[HYBRID] Cross-Encoder unavailable — returning RRF order")
        return [(idx, 0.0) for idx in candidate_indices]

    try:
        pairs = [(jd_text, resume_texts[idx]) for idx in candidate_indices]
        raw_scores = model.predict(pairs, show_progress_bar=False)

        # Sigmoid to map logits → [0, 1]
        def sigmoid(x):
            return 1.0 / (1.0 + math.exp(-float(x)))

        results = [
            (candidate_indices[i], sigmoid(raw_scores[i]))
            for i in range(len(candidate_indices))
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    except Exception as e:
        logger.error(f"[HYBRID] Cross-Encoder failed: {e}")
        return [(idx, 0.0) for idx in candidate_indices]


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _min_max_normalise(values: List[float]) -> List[float]:
    """Min-max normalise a list of floats to [0, 1]."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

class HybridRanker:
    """
    Hybrid Retrieve-and-Rerank pipeline.

    Usage::

        ranker = HybridRanker()
        ranker.rank(parsed_resumes, job_description, top_k=10)

    After calling ``rank()``, each resume's ``.scores`` dataclass is
    populated with the fields that the NLG and UI expect.
    """

    def __init__(self):
        # Models are lazy-loaded on first rank() call via module-level singletons
        pass

    # ------------------------------------------------------------------
    def rank(
        self,
        parsed_resumes: list,
        job_description: str,
        top_k: int = CE_TOP_K_DEFAULT,
    ) -> List[Dict[str, Any]]:
        """
        Score and rank a batch of parsed resumes against a job description.

        Mutates each resume's ``.scores`` in-place **and** returns the
        ``final_ranking`` list expected by ``_assemble_output()``.

        Pipeline:
            1. BM25 lexical retrieval
            2. SBERT semantic retrieval
            3. RRF fusion of (1) and (2)
            4. Cross-Encoder reranking on top-K from (3)
            5. Assign final_score: CE score for top-K, RRF score for the rest
        """
        n = len(parsed_resumes)
        if n == 0:
            return []

        t0 = time.perf_counter()

        # --- Build text corpus -------------------------------------------
        resume_texts = [_resume_to_text(r) for r in parsed_resumes]
        corpus_tokens = [_tokenize(t) for t in resume_texts]
        jd_tokens = _tokenize(job_description)

        # --- Stage 1: BM25 -----------------------------------------------
        logger.info("[HYBRID] Stage 1 — BM25 retrieval")
        bm25_ranking = _bm25_rank(corpus_tokens, jd_tokens)
        bm25_scores = {idx: sc for idx, sc in bm25_ranking}

        # --- Stage 2: SBERT ----------------------------------------------
        logger.info("[HYBRID] Stage 2 — SBERT retrieval")
        sbert_ranking = _sbert_rank(resume_texts, job_description)
        sbert_scores = {idx: sc for idx, sc in sbert_ranking}

        # --- Stage 3: RRF ------------------------------------------------
        logger.info("[HYBRID] Stage 3 — Reciprocal Rank Fusion")
        rrf_ranking = _rrf_fuse(bm25_ranking, sbert_ranking, k=RRF_K)
        rrf_scores = {idx: sc for idx, sc in rrf_ranking}

        # Normalise RRF scores to [0, 1] for use as final_score fallback
        rrf_vals = [sc for _, sc in rrf_ranking]
        rrf_norm_list = _min_max_normalise(rrf_vals)
        rrf_norm = {rrf_ranking[i][0]: rrf_norm_list[i] for i in range(len(rrf_ranking))}

        # --- Stage 4: Cross-Encoder on top-K -----------------------------
        top_k_actual = min(top_k, n)
        top_k_indices = [idx for idx, _ in rrf_ranking[:top_k_actual]]
        logger.info(f"[HYBRID] Stage 4 — Cross-Encoder reranking top-{top_k_actual}")
        ce_results = _cross_encoder_rerank(resume_texts, job_description, top_k_indices)
        ce_scores = {idx: sc for idx, sc in ce_results}

        # --- Normalise raw signal scores for NLG -------------------------
        all_bm25 = [bm25_scores.get(i, 0.0) for i in range(n)]
        all_sbert = [sbert_scores.get(i, 0.0) for i in range(n)]
        bm25_normed = _min_max_normalise(all_bm25)
        sbert_normed = _min_max_normalise(all_sbert)

        # --- Assign final scores -----------------------------------------
        # Two-tier: CE score for top-K, normalised RRF for the rest
        for i, resume in enumerate(parsed_resumes):
            sc = resume.scores

            # Raw signal scores
            sc.sbert_score = sbert_scores.get(i, 0.0)
            sc.ce_score = ce_scores.get(i, 0.0)
            sc.combined_tfidf = bm25_scores.get(i, 0.0)  # BM25 fills the "lexical" slot

            # Normalised scores (NLG reads these)
            sc.tfidf_norm = bm25_normed[i]
            sc.semantic_norm = sbert_normed[i]
            sc.ce_norm = ce_scores.get(i, 0.0)  # CE sigmoid already in [0,1]

            # Legacy aliases the dataclass still carries
            sc.tfidf_section_score = bm25_scores.get(i, 0.0)
            sc.tfidf_taxonomy_score = 0.0
            sc.semantic_score = sbert_scores.get(i, 0.0)
            sc.cross_encoder = ce_scores.get(i, 0.0)

            # Final score
            if i in ce_scores and ce_scores[i] > 0:
                sc.final_score = ce_scores[i]
            else:
                sc.final_score = rrf_norm.get(i, 0.0)

            sc.final_score_display = round(sc.final_score * 100, 2)
            sc.final_pre_llm = sc.final_score
            sc.final_pre_llm_display = sc.final_score_display

            # Score breakdown for NLG detailed view
            sc.score_breakdown = {
                "pipeline": "hybrid_retrieve_rerank",
                "stages": {
                    "bm25_raw": float(bm25_scores.get(i, 0.0)),
                    "bm25_norm": float(bm25_normed[i]),
                    "sbert_raw": float(sbert_scores.get(i, 0.0)),
                    "sbert_norm": float(sbert_normed[i]),
                    "rrf_raw": float(rrf_scores.get(i, 0.0)),
                    "rrf_norm": float(rrf_norm.get(i, 0.0)),
                    "ce_score": float(ce_scores.get(i, 0.0)),
                },
                "tier": "ce_reranked" if i in ce_scores and ce_scores[i] > 0 else "rrf_only",
                "final_score": float(sc.final_score),
            }

            # Rationale
            if i in ce_scores and ce_scores[i] > 0:
                sc.rationale = (
                    f"Cross-Encoder score {sc.ce_score:.3f} "
                    f"(BM25-norm {bm25_normed[i]:.2f}, SBERT {sbert_normed[i]:.2f})"
                )
            else:
                sc.rationale = (
                    f"RRF-fused score {rrf_norm.get(i, 0.0):.3f} "
                    f"(BM25-norm {bm25_normed[i]:.2f}, SBERT {sbert_normed[i]:.2f})"
                )

        # --- Build final_ranking list sorted by final_score desc ----------
        ranked_indices = sorted(range(n), key=lambda i: parsed_resumes[i].scores.final_score, reverse=True)
        final_ranking: List[Dict[str, Any]] = []
        for rank_pos, idx in enumerate(ranked_indices):
            r = parsed_resumes[idx]
            final_ranking.append({
                "id": r.id,
                "rank": rank_pos + 1,
                "reasoning": r.scores.rationale,
                "scores_snapshot": {
                    "final_score": r.scores.final_score,
                    "ce_score": r.scores.ce_score,
                    "sbert_score": r.scores.sbert_score,
                    "bm25_norm": r.scores.tfidf_norm,
                },
            })

        elapsed = time.perf_counter() - t0
        logger.info(
            f"[HYBRID] Pipeline complete: {n} resumes ranked in {elapsed:.2f}s "
            f"(CE reranked top-{top_k_actual})"
        )
        return final_ranking
