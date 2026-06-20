# LLM Benchmark API

A lightweight **FastAPI** service that loads two HuggingFace causal language models — `distilgpt2` and `facebook/opt-125m` — and exposes REST endpoints for single inference and head-to-head latency benchmarking.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [API Reference](#api-reference)
- [Getting Started](#getting-started)
- [Running Tests](#running-tests)
- [CI / CD](#ci--cd)
- [Benchmark Results](#benchmark-results)
- [Dependencies](#dependencies)

---

## Features

- **In-memory model cache** — models are downloaded and loaded once per server lifetime; every subsequent request is served from RAM.
- **GPU-aware** — automatically uses CUDA if available, otherwise falls back to CPU; no config change required.
- **`POST /infer`** — run a single generation pass on either model and get back the generated text, wall-clock latency, and token count.
- **`POST /compare`** — run a prompt **concurrently** on both models with `asyncio.gather()` and get side-by-side results in roughly the time of the slower model (not the sum).
- **`POST /benchmark`** — run the same prompt **10 times** on both models and receive p50 / p95 latency and average tokens-per-second for each.
- **`GET /` Health Check** — simple liveness probe returning API status and supported models list.
- **Optimized Inference** — Uses `torch.no_grad()` and `model.eval()` to disable training behaviours and gradients, improving memory usage and speed.
- **Clean Pydantic schemas** — all request and response shapes are typed and validated automatically.
- **Interactive Documentation** — Auto-generated Swagger UI and ReDoc pages available out of the box.
- **30-test pytest suite** — no real models needed; `run_inference` is mocked so the suite runs in under 10 seconds.
- **GitHub Actions CI** — linting with `ruff` and the full test suite run on every push / PR to `main`.

---

## Project Structure

```
Benchmark/
├── app/
│   ├── __init__.py          # Marks app/ as a Python package
│   ├── model_loader.py      # Downloads & caches HuggingFace models (GPU-aware)
│   ├── inference.py         # Single inference pass + metric collection
│   ├── schemas.py           # Pydantic request / response models
│   └── main.py              # FastAPI app — routes /infer, /compare, /benchmark
│
├── tests/
│   ├── __init__.py          # Marks tests/ as a Python package
│   └── test_api.py          # 30 pytest tests (TestRoot, TestInfer, TestBenchmark)
│
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions: ruff lint + pytest on push / PR
│
├── requirements.txt         # Pinned dependencies (pip freeze output)
└── README.md                # This file
```

---

## How It Works

### Transformer Architecture & Autoregressive Decoding

Both `distilgpt2` and `facebook/opt-125m` are decoder-only transformers. At inference time they generate text **autoregressively**: the model produces one token per forward pass, appending it to the sequence and feeding the extended sequence back in until `max_new_tokens` is reached. Each forward pass runs a full attention computation over all previously seen tokens, so latency scales roughly linearly with the number of tokens generated.

This API uses **greedy decoding** (`do_sample=False`), which always picks the highest-probability next token. Greedy decoding is deterministic and has the lowest per-token latency because it skips the multinomial sampling and temperature scaling steps. Sampling strategies (top-k, nucleus/top-p) improve output diversity at the cost of extra computation and non-determinism — the right trade-off depends on whether you prioritise throughput or generation quality.

**p95 latency** (the 95th-percentile value from 10 repeated runs) matters because the median (p50) can hide outlier slowdowns caused by garbage collection, OS scheduling jitter, or the first run's JIT warm-up. A low p50 with a high p95 signals unstable tail performance — a critical signal for any production inference service where SLA breaches happen in the tail, not the average.

The `POST /compare` endpoint demonstrates a key async pattern: both models are dispatched with `asyncio.gather()` so they execute concurrently in a thread-pool executor, reducing total wall-clock time to roughly `max(t_a, t_b)` instead of `t_a + t_b`.

---

### `app/model_loader.py`

Detects the best available compute device at import time:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

Maintains a module-level dict `_cache` that maps a model name to its `(tokenizer, model)` pair.

```python
SUPPORTED_MODELS = ["distilgpt2", "facebook/opt-125m"]
```

`get_model(model_name)` checks the cache first; on a miss it calls `AutoTokenizer.from_pretrained` and `AutoModelForCausalLM.from_pretrained`, automatically moves the model to the detected device (`cuda` or `cpu`), sets the model to `eval()` mode, stores the pair, then returns it. Unsupported names raise `ValueError`.

---

### `app/inference.py`

`run_inference(prompt, model_name, max_new_tokens=50)` is the single source of truth for one generation pass:

1. Encodes the prompt with the tokenizer (`return_tensors="pt"`) and moves tensors to the model's device.
2. Uses the `torch.no_grad()` context manager to disable gradient tracking for optimal memory usage and speed.
3. Records `time.perf_counter()` before and after `model.generate(...)`.
4. Suppresses padding warnings by explicitly setting `pad_token_id=tokenizer.eos_token_id`.
5. Strips the prompt tokens from the output, decodes only the new tokens.
6. Returns `{"generated_text", "latency_ms", "token_count"}`.

Generation uses **greedy decoding** (`do_sample=False`) — deterministic and fast.

---

### `app/schemas.py`

Four Pydantic v2 models that enforce the API contract:

| Schema | Used by | Key fields |
|---|---|---|
| `InferRequest` | `POST /infer` (input) | `prompt` (str), `model` (Literal) |
| `InferResponse` | `POST /infer` (output) | `model`, `generated_text`, `latency_ms`, `token_count` |
| `BenchmarkRequest` | `POST /benchmark` (input) | `prompt` (str) |
| `BenchmarkResponse` | `POST /benchmark` (output) | `prompt`, `runs_per_model`, `results` (list of `ModelBenchmarkStats`) |

`ModelBenchmarkStats` holds `model`, `p50_ms`, `p95_ms`, `avg_tokens_per_sec` for each model in the benchmark.

---

### `app/main.py`

Creates the `FastAPI` app and wires up three routes:

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check — returns `{"status": "ok", "supported_models": [...]}` |
| `POST` | `/infer` | Single inference pass on the chosen model |
| `POST` | `/compare` | Both models run **concurrently** via `asyncio.gather()`, side-by-side results |
| `POST` | `/benchmark` | 10 runs × 2 models, returns p50 / p95 / avg tokens/sec using `numpy.percentile` |

The benchmark loops over `SUPPORTED_MODELS`, calls `run_inference` `BENCHMARK_RUNS = 10` times, safely computes token throughput avoiding division by zero, collects latencies into a NumPy array, and then computes percentiles.

---

## API Reference

### `GET /`

**Response**
```json
{
  "status": "ok",
  "supported_models": ["distilgpt2", "facebook/opt-125m"]
}
```

---

### `POST /infer`

**Request body**
```json
{
  "prompt": "Once upon a time",
  "model": "distilgpt2"
}
```

> `model` is optional and defaults to `"distilgpt2"`. Accepted values: `"distilgpt2"`, `"facebook/opt-125m"`.

**Response**
```json
{
  "model": "distilgpt2",
  "generated_text": ", there was a kingdom …",
  "latency_ms": 1432.871,
  "token_count": 50
}
```

---

### `POST /compare`

Run a prompt concurrently on **all supported models** and receive side-by-side results. Total wall-clock time ≈ the slower of the two models, not their sum.

**Request body**
```json
{
  "prompt": "Explain quantum entanglement"
}
```

**Response**
```json
{
  "prompt": "Explain quantum entanglement",
  "results": [
    {
      "model": "distilgpt2",
      "generated_text": " in terms of the quantum …",
      "latency_ms": 1412.553,
      "token_count": 50
    },
    {
      "model": "facebook/opt-125m",
      "generated_text": " as a phenomenon where …",
      "latency_ms": 1589.201,
      "token_count": 50
    }
  ]
}
```

---

### `POST /benchmark`

**Request body**
```json
{
  "prompt": "The quick brown fox"
}
```

**Response**
```json
{
  "prompt": "The quick brown fox",
  "runs_per_model": 10,
  "results": [
    {
      "model": "distilgpt2",
      "p50_ms": 1565.662,
      "p95_ms": 1614.037,
      "avg_tokens_per_sec": 31.771
    },
    {
      "model": "facebook/opt-125m",
      "p50_ms": 1596.663,
      "p95_ms": 1731.804,
      "avg_tokens_per_sec": 30.924
    }
  ]
}
```

> ⚠️ The benchmark endpoint runs **20 inference passes** (10 per model) synchronously. Expect it to take 30–60 seconds on CPU.

---

## Getting Started

### Prerequisites

- Python 3.11+
- (Recommended) a virtual environment

### 1 — Clone and create a virtual environment

```powershell
git clone <your-repo-url>
cd Benchmark
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
```

### 2 — Install dependencies

```powershell
pip install -r requirements.txt
```

> The first time a model is requested, HuggingFace will download its weights (~250 MB each). They are cached in `~/.cache/huggingface/`.

### 3 — Start the server

```powershell
uvicorn app.main:app --reload
```

The API is now available at **http://127.0.0.1:8000**.  
Interactive docs (Swagger UI): **http://127.0.0.1:8000/docs**  
Alternative docs (ReDoc): **http://127.0.0.1:8000/redoc**

### 4 — Try it

```powershell
# Health check
curl http://localhost:8000/

# Single inference
curl -X POST http://localhost:8000/infer `
  -H "Content-Type: application/json" `
  -d '{"prompt": "Once upon a time", "model": "distilgpt2"}'

# Full benchmark (both models, 10 runs each)
curl -X POST http://localhost:8000/benchmark `
  -H "Content-Type: application/json" `
  -d '{"prompt": "The quick brown fox"}'
```

---

## Running Tests

The test suite mocks `run_inference` — **no models are downloaded** during testing.

```powershell
venv\Scripts\pytest tests\test_api.py -v
```

Expected output:

```
collected 30 items

tests/test_api.py::TestRoot::test_status_ok                              PASSED
tests/test_api.py::TestRoot::test_status_field                           PASSED
tests/test_api.py::TestRoot::test_supported_models_present               PASSED
tests/test_api.py::TestInfer::test_returns_200                           PASSED
... (27 more) ...

30 passed in 8.69s
```

### Test layout

| Class | Tests | What is covered |
|---|---|---|
| `TestRoot` | 3 | `GET /` status, body shape, supported model list |
| `TestInfer` | 12 | HTTP 200, all four response fields present, field types, default model, 422 on bad model / missing prompt |
| `TestBenchmark` | 15 | HTTP 200, `prompt` echo, `runs_per_model`, both models in results, p50/p95/avg_tokens fields, p95 ≥ p50, 422 on missing prompt |

---

## CI / CD

**File:** [`.github/workflows/ci.yml`](.github/workflows/ci.yml)

The pipeline triggers on every **push** and **pull request** to `main`:

```
Checkout  →  Setup Python 3.11  →  pip install -r requirements.txt
         →  ruff check .         →  pytest
```

- **Ruff** enforces code style and catches common errors.
- **pytest** runs the full 30-test suite (no GPU / model downloads required).

---

## Benchmark Results

Results collected on CPU (greedy decoding, `max_new_tokens=50`, 10 runs each):

| Model | p50 Latency | p95 Latency | Avg Tokens/sec |
|---|---:|---:|---:|
| distilgpt2 | 1565.662 ms | 1614.037 ms | 31.771 |
| facebook/opt-125m | 1596.663 ms | 1731.804 ms | 30.924 |

> Results will vary based on hardware. A GPU will reduce latency by 10–50×.

---

## Dependencies

Key packages (see [`requirements.txt`](requirements.txt) for full pinned versions):

| Package | Version | Purpose |
|---|---|---|
| `fastapi` | 0.138.0 | Web framework |
| `uvicorn` | 0.49.0 | ASGI server |
| `transformers` | 5.12.1 | HuggingFace model loading |
| `torch` | 2.12.1 | Tensor computation / model inference |
| `numpy` | 2.4.6 | Percentile computation for benchmark stats |
| `pydantic` | 2.13.4 | Request / response schema validation |
| `pytest` | 9.1.1 | Test runner |
| `httpx` | 0.28.1 | HTTP client used by `TestClient` |
| `ruff` | 0.15.18 | Fast Python linter |