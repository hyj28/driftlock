"""The pinned local embedding model for skill retrieval.

PLAN 3.3 makes this a deliberate host-local setup step rather than a library
dependency: driftlock declares dependencies = [] and takes an injected callable,
so the concrete model lives here and its revision is recorded here. Installing
``sentence-transformers`` enables the genuine pinned-embedder integration test.

Host-local inference, so this is zero API cost and outside the token budget.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"

_lock = threading.Lock()
_model = None


def _load():
    global _model
    with _lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION)
        return _model


def embed(texts: Sequence[str]) -> list[list[float]]:
    model = _load()
    vectors = model.encode(
        list(texts),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return [[float(value) for value in row] for row in vectors]
