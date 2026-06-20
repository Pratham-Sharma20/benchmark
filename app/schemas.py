"""
schemas.py
----------
Pydantic models that describe the shape of every request and response body
used by the API.  Keeping them in one place makes the contract easy to read.
"""

from pydantic import BaseModel, Field
from typing import Literal


# ── /infer ────────────────────────────────────────────────────────────────────

class InferRequest(BaseModel):
    """Body accepted by POST /infer."""

    prompt: str = Field(
        ...,
        description="The text prompt to feed into the model.",
        examples=["Once upon a time"],
    )
    model: Literal["distilgpt2", "facebook/opt-125m"] = Field(
        default="distilgpt2",
        description="Which model to use for generation.",
    )


class InferResponse(BaseModel):
    """Body returned by POST /infer."""

    model: str = Field(description="Model that was used.")
    generated_text: str = Field(description="Text produced by the model.")
    latency_ms: float = Field(description="Wall-clock time for generation (ms).")
    token_count: int = Field(description="Number of tokens generated.")


# ── /benchmark ────────────────────────────────────────────────────────────────

class BenchmarkRequest(BaseModel):
    """Body accepted by POST /benchmark."""

    prompt: str = Field(
        ...,
        description="The prompt that will be run 10 times on each model.",
        examples=["The quick brown fox"],
    )


class ModelBenchmarkStats(BaseModel):
    """Per-model statistics returned by /benchmark."""

    model: str = Field(description="Model identifier.")
    p50_ms: float = Field(description="Median (50th-percentile) latency in ms.")
    p95_ms: float = Field(description="95th-percentile latency in ms.")
    avg_tokens_per_sec: float = Field(
        description="Average throughput: tokens generated per second."
    )


class BenchmarkResponse(BaseModel):
    """Body returned by POST /benchmark."""

    prompt: str = Field(description="The prompt that was benchmarked.")
    runs_per_model: int = Field(description="How many runs were executed per model.")
    results: list[ModelBenchmarkStats] = Field(
        description="One entry per model that was benchmarked."
    )
