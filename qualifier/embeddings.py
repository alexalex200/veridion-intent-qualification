"""Local, non-LLM text similarity backend.

Default: TF-IDF + cosine similarity via scikit-learn. Zero network calls,
no model download, fits in milliseconds for datasets of this size, and is
fully interpretable (you can inspect which terms drove a match).

Optional upgrade: if `sentence-transformers` is installed AND the
QUALIFIER_USE_ST=1 environment variable is set, a small local sentence
embedding model is used instead/in addition, for better handling of
paraphrase-heavy queries (e.g. "fast-growing fintech competing with
traditional banks" vs. a company description that never uses those exact
words). This is opt-in because it requires a one-time model download
(~90MB) and a torch install - not something you want to force on a
grading environment.

Both backends expose the same interface: fit on a corpus of company text
blobs once, then score a query against all of them in one vectorized call
(no per-company round trips) - which is what makes this stage cheap
enough to run on every company before any expensive step.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfBackend:
    name = "tfidf"

    def __init__(self, corpus: List[str]):
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), min_df=1, max_df=0.9
        )
        self.matrix = self.vectorizer.fit_transform(corpus)

    def score(self, query_text: str) -> np.ndarray:
        query_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        return sims


class SentenceTransformerBackend:
    name = "sentence-transformers"

    def __init__(self, corpus: List[str], model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # local import: optional dep

        self.model = SentenceTransformer(model_name)
        self.doc_embeddings = self.model.encode(
            corpus, normalize_embeddings=True, show_progress_bar=False
        )

    def score(self, query_text: str) -> np.ndarray:
        query_embedding = self.model.encode(
            [query_text], normalize_embeddings=True, show_progress_bar=False
        )[0]
        return self.doc_embeddings @ query_embedding


def build_embedder(corpus: List[str]):
    """Fits an embedding backend once over the full company corpus.
    Falls back to TF-IDF if the optional dense backend isn't available
    or fails to load, so the pipeline never hard-depends on a model
    download being reachable."""
    if os.environ.get("QUALIFIER_USE_ST") == "1":
        try:
            return SentenceTransformerBackend(corpus)
        except Exception as exc:  # ImportError, download failure, etc.
            print(f"[embeddings] sentence-transformers unavailable ({exc}); "
                  f"falling back to TF-IDF.")
    return TfidfBackend(corpus)
