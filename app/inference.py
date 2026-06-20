"""
inference.py
------------
Core logic for running a single inference pass and collecting metrics
(latency in ms, number of tokens generated).
"""

import time
import torch
from app.model_loader import get_model


def run_inference(prompt: str, model_name: str, max_new_tokens: int = 50) -> dict:
    """
    Run one forward pass through *model_name* for the given *prompt*.

    Returns
    -------
    dict with keys:
        generated_text  : str   – the model's output (prompt stripped)
        latency_ms      : float – wall-clock time in milliseconds
        token_count     : int   – number of tokens generated
    """
    tokenizer, model = get_model(model_name)

    # Encode the prompt into token IDs
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids: torch.Tensor = inputs["input_ids"]
    input_len = input_ids.shape[-1]

    # ── Timed generation ──────────────────────────────────────────────────
    start = time.perf_counter()

    with torch.no_grad():
        output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    end = time.perf_counter()
    # ──────────────────────────────────────────────────────────────────────

    latency_ms = (end - start) * 1_000

    # Decode only the newly generated tokens (skip the prompt)
    new_token_ids = output_ids[0][input_len:]
    token_count = len(new_token_ids)
    generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)

    return {
        "generated_text": generated_text,
        "latency_ms": round(latency_ms, 3),
        "token_count": token_count,
    }
