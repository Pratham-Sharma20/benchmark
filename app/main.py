"""
main.py
-------
FastAPI application exposing two endpoints:

  POST /infer      – run a single inference and return metrics
  POST /benchmark  – run the same prompt 10 times on every supported model
                     and return aggregate statistics
"""

import asyncio
import numpy as np
from fastapi import FastAPI, HTTPException

from app.model_loader import SUPPORTED_MODELS
from app.inference import run_inference
from app.schemas import (
    InferRequest,
    InferResponse,
    BenchmarkRequest,
    BenchmarkResponse,
    ModelBenchmarkStats,
    CompareRequest,
    CompareResponse,
)

# Number of repeated runs used for the benchmark
BENCHMARK_RUNS = 10

app = FastAPI(
    title="LLM Benchmark API",
    description="Compare inference speed of distilgpt2 and facebook/opt-125m.",
    version="1.0.0",
)


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
def root():
    """Simple liveness check."""
    return {"status": "ok", "supported_models": SUPPORTED_MODELS}


# ── POST /infer ───────────────────────────────────────────────────────────────

@app.post("/infer", response_model=InferResponse, tags=["inference"])
def infer(body: InferRequest) -> InferResponse:
    """
    Run a single generation pass.

    - **prompt** – text sent to the model
    - **model**  – which model to use (`distilgpt2` or `facebook/opt-125m`)

    Returns the generated text, latency in milliseconds, and token count.
    """
    try:
        result = run_inference(prompt=body.prompt, model_name=body.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return InferResponse(
        model=body.model,
        generated_text=result["generated_text"],
        latency_ms=result["latency_ms"],
        token_count=result["token_count"],
    )


# ── POST /benchmark ───────────────────────────────────────────────────────────

@app.post("/benchmark", response_model=BenchmarkResponse, tags=["benchmark"])
def benchmark(body: BenchmarkRequest) -> BenchmarkResponse:
    """
    Run the same prompt **10 times** on *every* supported model and return
    aggregate latency statistics.

    Statistics per model:
    - **p50_ms**             – median latency (50th percentile)
    - **p95_ms**             – 95th-percentile latency
    - **avg_tokens_per_sec** – average generation throughput
    """
    model_stats: list[ModelBenchmarkStats] = []

    for model_name in SUPPORTED_MODELS:
        latencies_ms: list[float] = []
        tokens_per_sec: list[float] = []

        for run_idx in range(BENCHMARK_RUNS):
            print(f"[benchmark] {model_name}  run {run_idx + 1}/{BENCHMARK_RUNS}")
            result = run_inference(prompt=body.prompt, model_name=model_name)

            latency_ms = result["latency_ms"]
            token_count = result["token_count"]

            latencies_ms.append(latency_ms)

            # Avoid division by zero for extremely fast (empty) outputs
            if latency_ms > 0 and token_count > 0:
                tokens_per_sec.append(token_count / (latency_ms / 1_000))

        # Use numpy for percentile computation
        latencies_arr = np.array(latencies_ms)
        p50 = float(np.percentile(latencies_arr, 50))
        p95 = float(np.percentile(latencies_arr, 95))
        avg_tps = float(np.mean(tokens_per_sec)) if tokens_per_sec else 0.0

        model_stats.append(
            ModelBenchmarkStats(
                model=model_name,
                p50_ms=round(p50, 3),
                p95_ms=round(p95, 3),
                avg_tokens_per_sec=round(avg_tps, 3),
            )
        )

    return BenchmarkResponse(
        prompt=body.prompt,
        runs_per_model=BENCHMARK_RUNS,
        results=model_stats,
    )


# ── POST /compare ─────────────────────────────────────────────────────────────

@app.post("/compare", response_model=CompareResponse, tags=["inference"])
async def compare(body: CompareRequest) -> CompareResponse:
    """
    Run a prompt **concurrently** on all supported models using
    `asyncio.gather()` and return side-by-side results.

    Both models execute at the same time (via the default thread-pool executor),
    so total wall-clock time ≈ max(model_a_latency, model_b_latency) rather
    than the sum — demonstrating async I/O patterns for high-throughput serving.
    """
    loop = asyncio.get_event_loop()

    async def _infer(model_name: str) -> InferResponse:
        try:
            result = await loop.run_in_executor(
                None,  # default ThreadPoolExecutor
                lambda: run_inference(prompt=body.prompt, model_name=model_name),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return InferResponse(
            model=model_name,
            generated_text=result["generated_text"],
            latency_ms=result["latency_ms"],
            token_count=result["token_count"],
        )

    results = await asyncio.gather(*[_infer(m) for m in SUPPORTED_MODELS])

    return CompareResponse(prompt=body.prompt, results=list(results))
