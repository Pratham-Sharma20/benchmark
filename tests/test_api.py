"""
tests/test_api.py
-----------------
Pytest tests for the LLM Benchmark API.

The tests mock `run_inference` so that no HuggingFace models are downloaded
or loaded during the test run — making the suite fast and CI-friendly.
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app, BENCHMARK_RUNS
from app.model_loader import SUPPORTED_MODELS

# ---------------------------------------------------------------------------
# Shared test client
# ---------------------------------------------------------------------------

client = TestClient(app)

# ---------------------------------------------------------------------------
# Fake inference result returned by the mock
# ---------------------------------------------------------------------------

FAKE_RESULT = {
    "generated_text": "jumps over the lazy dog",
    "latency_ms": 100.0,
    "token_count": 10,
}

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def mock_run_inference(prompt: str, model_name: str, **kwargs) -> dict:
    """Drop-in replacement for run_inference that returns instantly."""
    return FAKE_RESULT


# ===========================================================================
# GET /  – health check
# ===========================================================================

class TestRoot:
    def test_status_ok(self):
        """The root endpoint must return HTTP 200 and status='ok'."""
        response = client.get("/")
        assert response.status_code == 200

    def test_status_field(self):
        """Response body must contain {"status": "ok"}."""
        data = client.get("/").json()
        assert data["status"] == "ok"

    def test_supported_models_present(self):
        """Response must list every supported model."""
        data = client.get("/").json()
        assert "supported_models" in data
        for model in SUPPORTED_MODELS:
            assert model in data["supported_models"]


# ===========================================================================
# POST /infer
# ===========================================================================

class TestInfer:
    """All tests patch run_inference so no real model is touched."""

    @pytest.fixture(autouse=True)
    def patch_inference(self):
        with patch("app.main.run_inference", side_effect=mock_run_inference):
            yield

    # ── Status code ─────────────────────────────────────────────────────────

    def test_returns_200(self):
        response = client.post(
            "/infer",
            json={"prompt": "Hello world", "model": "distilgpt2"},
        )
        assert response.status_code == 200

    # ── Required response fields ─────────────────────────────────────────────

    def test_has_model_field(self):
        data = client.post(
            "/infer",
            json={"prompt": "Hello world", "model": "distilgpt2"},
        ).json()
        assert "model" in data

    def test_has_generated_text_field(self):
        data = client.post(
            "/infer",
            json={"prompt": "Hello world", "model": "distilgpt2"},
        ).json()
        assert "generated_text" in data

    def test_has_latency_ms_field(self):
        data = client.post(
            "/infer",
            json={"prompt": "Hello world", "model": "distilgpt2"},
        ).json()
        assert "latency_ms" in data

    def test_has_token_count_field(self):
        data = client.post(
            "/infer",
            json={"prompt": "Hello world", "model": "distilgpt2"},
        ).json()
        assert "token_count" in data

    # ── Field value correctness ──────────────────────────────────────────────

    def test_model_field_matches_request(self):
        """The returned model name must equal the one sent in the request."""
        for model_name in SUPPORTED_MODELS:
            data = client.post(
                "/infer",
                json={"prompt": "test", "model": model_name},
            ).json()
            assert data["model"] == model_name

    def test_generated_text_is_string(self):
        data = client.post(
            "/infer",
            json={"prompt": "test", "model": "distilgpt2"},
        ).json()
        assert isinstance(data["generated_text"], str)

    def test_latency_ms_is_non_negative(self):
        data = client.post(
            "/infer",
            json={"prompt": "test", "model": "distilgpt2"},
        ).json()
        assert data["latency_ms"] >= 0

    def test_token_count_is_non_negative_int(self):
        data = client.post(
            "/infer",
            json={"prompt": "test", "model": "distilgpt2"},
        ).json()
        assert isinstance(data["token_count"], int)
        assert data["token_count"] >= 0

    # ── Default model ────────────────────────────────────────────────────────

    def test_default_model_is_distilgpt2(self):
        """Omitting the model field should default to distilgpt2."""
        data = client.post(
            "/infer",
            json={"prompt": "test"},
        ).json()
        assert data["model"] == "distilgpt2"

    # ── Validation error ─────────────────────────────────────────────────────

    def test_unsupported_model_returns_422(self):
        """An unknown model name must be rejected by Pydantic (422)."""
        response = client.post(
            "/infer",
            json={"prompt": "test", "model": "gpt-4"},
        )
        assert response.status_code == 422

    def test_missing_prompt_returns_422(self):
        """Omitting the required prompt field must return 422."""
        response = client.post("/infer", json={"model": "distilgpt2"})
        assert response.status_code == 422


# ===========================================================================
# POST /benchmark
# ===========================================================================

class TestBenchmark:
    """All tests patch run_inference so no real model is touched."""

    @pytest.fixture(autouse=True)
    def patch_inference(self):
        with patch("app.main.run_inference", side_effect=mock_run_inference):
            yield

    # ── Status code ─────────────────────────────────────────────────────────

    def test_returns_200(self):
        response = client.post(
            "/benchmark",
            json={"prompt": "The quick brown fox"},
        )
        assert response.status_code == 200

    # ── Top-level response fields ────────────────────────────────────────────

    def test_has_prompt_field(self):
        data = client.post(
            "/benchmark", json={"prompt": "The quick brown fox"}
        ).json()
        assert "prompt" in data

    def test_prompt_matches_request(self):
        prompt = "The quick brown fox"
        data = client.post("/benchmark", json={"prompt": prompt}).json()
        assert data["prompt"] == prompt

    def test_has_runs_per_model_field(self):
        data = client.post(
            "/benchmark", json={"prompt": "test"}
        ).json()
        assert "runs_per_model" in data

    def test_runs_per_model_equals_benchmark_runs_constant(self):
        """runs_per_model must equal BENCHMARK_RUNS (currently 10)."""
        data = client.post(
            "/benchmark", json={"prompt": "test"}
        ).json()
        assert data["runs_per_model"] == BENCHMARK_RUNS

    def test_has_results_field(self):
        data = client.post(
            "/benchmark", json={"prompt": "test"}
        ).json()
        assert "results" in data

    # ── Results list ─────────────────────────────────────────────────────────

    def test_results_has_one_entry_per_model(self):
        """There must be exactly one result entry for every supported model."""
        data = client.post(
            "/benchmark", json={"prompt": "test"}
        ).json()
        assert len(data["results"]) == len(SUPPORTED_MODELS)

    def test_results_contain_both_models(self):
        """Both distilgpt2 and facebook/opt-125m must appear in results."""
        data = client.post(
            "/benchmark", json={"prompt": "test"}
        ).json()
        returned_models = {entry["model"] for entry in data["results"]}
        for model in SUPPORTED_MODELS:
            assert model in returned_models

    # ── Per-model stats fields ───────────────────────────────────────────────

    def test_each_result_has_p50_ms(self):
        data = client.post("/benchmark", json={"prompt": "test"}).json()
        for entry in data["results"]:
            assert "p50_ms" in entry

    def test_each_result_has_p95_ms(self):
        data = client.post("/benchmark", json={"prompt": "test"}).json()
        for entry in data["results"]:
            assert "p95_ms" in entry

    def test_each_result_has_avg_tokens_per_sec(self):
        data = client.post("/benchmark", json={"prompt": "test"}).json()
        for entry in data["results"]:
            assert "avg_tokens_per_sec" in entry

    # ── Stat value sanity ────────────────────────────────────────────────────

    def test_p50_ms_is_non_negative(self):
        data = client.post("/benchmark", json={"prompt": "test"}).json()
        for entry in data["results"]:
            assert entry["p50_ms"] >= 0

    def test_p95_ms_is_greater_or_equal_to_p50_ms(self):
        """p95 must always be >= p50 by definition of percentiles."""
        data = client.post("/benchmark", json={"prompt": "test"}).json()
        for entry in data["results"]:
            assert entry["p95_ms"] >= entry["p50_ms"]

    def test_avg_tokens_per_sec_is_non_negative(self):
        data = client.post("/benchmark", json={"prompt": "test"}).json()
        for entry in data["results"]:
            assert entry["avg_tokens_per_sec"] >= 0

    # ── Validation error ─────────────────────────────────────────────────────

    def test_missing_prompt_returns_422(self):
        response = client.post("/benchmark", json={})
        assert response.status_code == 422
