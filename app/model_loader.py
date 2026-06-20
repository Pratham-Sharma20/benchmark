"""
model_loader.py
---------------
Loads HuggingFace causal language models and caches them in memory so they
are only downloaded / initialised once per server lifetime.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM

# Supported model identifiers
SUPPORTED_MODELS = ["distilgpt2", "facebook/opt-125m"]

# In-memory cache: { model_name: (tokenizer, model) }
_cache: dict[str, tuple] = {}


def get_model(model_name: str):
    """
    Return a (tokenizer, model) pair for *model_name*.

    The first call for a given name downloads and loads the model;
    subsequent calls return the cached objects instantly.

    Raises ValueError if *model_name* is not in SUPPORTED_MODELS.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Model '{model_name}' is not supported. "
            f"Choose from: {SUPPORTED_MODELS}"
        )

    if model_name not in _cache:
        print(f"[model_loader] Loading '{model_name}' for the first time …")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()  # disable dropout / batch-norm training behaviour
        _cache[model_name] = (tokenizer, model)
        print(f"[model_loader] '{model_name}' loaded and cached.")

    return _cache[model_name]
