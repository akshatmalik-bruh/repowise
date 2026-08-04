"""Unit tests for GeminiEmbedder.

All tests patch google.genai via sys.modules — no real API key or network call is made.
The pattern mirrors the other embedder suites: mock the external SDK, run the real embed().
"""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from repowise.core.providers.embedding.gemini import GeminiEmbedder


# ---------------------------------------------------------------------------
# Shared fake-SDK helpers
# ---------------------------------------------------------------------------


def _fake_sdk(vectors: list[list[float]]) -> tuple[MagicMock, MagicMock]:
    """Build a minimal google.genai + google.genai.types double that returns *vectors*."""
    embeddings = [SimpleNamespace(values=v) for v in vectors]
    result = SimpleNamespace(embeddings=embeddings)

    client = MagicMock()
    client.models.embed_content.return_value = result

    genai_mod = MagicMock()
    genai_mod.Client.return_value = client

    types_mod = MagicMock()
    types_mod.EmbedContentConfig.return_value = MagicMock()
    types_mod.HttpOptions.return_value = MagicMock()

    return genai_mod, types_mod


class _PatchedSDK:
    """Context manager: temporarily injects fake google.genai modules."""

    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors
        self._orig_genai = None
        self._orig_types = None

    def __enter__(self) -> None:
        self._orig_genai = sys.modules.get("google.genai")
        self._orig_types = sys.modules.get("google.genai.types")
        genai_mod, types_mod = _fake_sdk(self.vectors)
        sys.modules["google.genai"] = genai_mod
        sys.modules["google.genai.types"] = types_mod

    def __exit__(self, *_) -> None:
        if self._orig_genai is None:
            sys.modules.pop("google.genai", None)
        else:
            sys.modules["google.genai"] = self._orig_genai
        if self._orig_types is None:
            sys.modules.pop("google.genai.types", None)
        else:
            sys.modules["google.genai.types"] = self._orig_types


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Gemini API key required"):
        GeminiEmbedder(api_key=None)


def test_api_key_from_gemini_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    emb = GeminiEmbedder()
    assert emb._api_key == "gemini-test-key"


def test_api_key_from_google_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-test-key")
    emb = GeminiEmbedder()
    assert emb._api_key == "google-test-key"


def test_default_dimensions():
    emb = GeminiEmbedder(api_key="k")
    assert emb.dimensions == 768


def test_custom_dimensions():
    emb = GeminiEmbedder(api_key="k", output_dimensionality=2048)
    assert emb.dimensions == 2048


# ---------------------------------------------------------------------------
# Embedding — happy path
# ---------------------------------------------------------------------------


async def test_embed_empty_returns_empty():
    emb = GeminiEmbedder(api_key="k")
    assert await emb.embed([]) == []


async def test_embed_returns_normalized_vectors():
    # 3-4-5 right triangle → L2 norm = 5 → normalised = [0.6, 0.8]
    with _PatchedSDK([[3.0, 4.0]]):
        emb = GeminiEmbedder(api_key="k", output_dimensionality=2)
        result = await emb.embed(["hello"])

    assert len(result) == 1
    norm = math.sqrt(sum(x * x for x in result[0]))
    assert abs(norm - 1.0) < 1e-6


async def test_embed_batch_returns_correct_count():
    with _PatchedSDK([[1.0, 0.0], [0.0, 1.0], [0.707, 0.707]]):
        emb = GeminiEmbedder(api_key="k", output_dimensionality=2)
        result = await emb.embed(["a", "b", "c"])

    assert len(result) == 3


# ---------------------------------------------------------------------------
# Width verification — the check added in fix/embedder-width-verification
# ---------------------------------------------------------------------------


async def test_embed_raises_when_api_returns_wrong_width():
    """API ignores output_dimensionality and returns a different-width vector.

    The error must name both the declared width and the actual width, and
    reference 'output_dimensionality' so the user knows what to change.
    """
    # Declared: 768. Fake SDK returns 3-wide vectors.
    with _PatchedSDK([[1.0, 0.0, 0.0]]):
        emb = GeminiEmbedder(api_key="k", output_dimensionality=768)
        with pytest.raises(ValueError, match="768") as exc_info:
            await emb.embed(["hello"])

    msg = str(exc_info.value)
    assert "3" in msg                       # actual width named
    assert "output_dimensionality" in msg   # tells user which parameter to fix


async def test_embed_raises_when_api_returns_wrong_width_custom_dim():
    """Same check holds for a non-default output_dimensionality."""
    with _PatchedSDK([[1.0, 0.0, 0.0]]):
        emb = GeminiEmbedder(api_key="k", output_dimensionality=2048)
        with pytest.raises(ValueError, match="2048") as exc_info:
            await emb.embed(["hello"])

    msg = str(exc_info.value)
    assert "3" in msg
    assert "output_dimensionality" in msg


async def test_embed_width_check_not_triggered_on_empty():
    # embed([]) returns early without touching the SDK.
    emb = GeminiEmbedder(api_key="k")
    assert await emb.embed([]) == []
