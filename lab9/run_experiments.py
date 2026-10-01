"""
Runs the three transformer-architecture demos and the local-LLM-vs-Claude
comparison described in README.md.

Run: python3 run_experiments.py
"""

import time
from transformer_demos import run_translation, run_sentiment, run_generation


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
section("Encoder-decoder: machine translation (Helsinki-NLP/opus-mt-en-fr)")
SENTENCES_EN = [
    "Artificial intelligence lets machines learn from data.",
    "The warehouse robot searches for the shortest path to the goal.",
    "A Bayesian network represents conditional independence assumptions.",
]
for src, tgt in run_translation(SENTENCES_EN):
    print(f"EN: {src}")
    print(f"FR: {tgt}\n")

# ---------------------------------------------------------------------------
section("Encoder-only: sentiment analysis (distilbert-base-uncased-finetuned-sst-2-english)")
SENTIMENT_EXAMPLES = [
    "I loved this course, the labs were genuinely interesting.",
    "The assignment was confusing and the deadline was far too short.",
    "The lecture covered Bayesian networks today.",
]
for text, result in run_sentiment(SENTIMENT_EXAMPLES):
    print(f"{result['label']:<10} ({result['score']:.4f})  {text}")

# ---------------------------------------------------------------------------
section("Decoder-only: GPT-style generation (distilgpt2)")
GEN_PROMPTS = [
    "The warehouse robot",
    "Artificial intelligence is",
]
for prompt, text in run_generation(GEN_PROMPTS, max_new_tokens=30):
    print(f"Prompt: {prompt!r}")
    print(f"Continuation: {text!r}\n")

# ---------------------------------------------------------------------------
section("Local open-weights LLM vs Claude: same prompts, compared")
print("Loading Qwen/Qwen2.5-0.5B-Instruct locally via transformers "
      "(substituting for Ollama - see README.md for why)...")
from transformers import pipeline  # noqa: E402  (deferred: slow import, only needed here)

t0 = time.time()
local_llm = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")
print(f"Model loaded in {time.time() - t0:.1f}s")

COMPARISON_PROMPTS = [
    "In one sentence, what is self-attention in a transformer?",
    "In one sentence, why does A* search use a heuristic?",
    "In one sentence, what does it mean for a heuristic to be admissible?",
]

for prompt in COMPARISON_PROMPTS:
    t0 = time.time()
    out = local_llm([{"role": "user", "content": prompt}], max_new_tokens=60, do_sample=False)
    elapsed = time.time() - t0
    answer = out[0]["generated_text"][-1]["content"]
    print(f"\nPrompt: {prompt}")
    print(f"Qwen2.5-0.5B-Instruct ({elapsed:.1f}s): {answer}")
