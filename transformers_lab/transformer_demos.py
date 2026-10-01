"""
Hands-on transformer demos, one per architecture family:

  Encoder-decoder -> machine translation   (Helsinki-NLP/opus-mt-en-fr)
  Encoder-only    -> sentiment analysis    (distilbert-base-uncased-finetuned-sst-2-english)
  Decoder-only    -> GPT-style generation  (distilgpt2)

All models are small enough to run on CPU. Pipeline task names changed
between transformers versions (this environment's transformers==5.18 no
longer registers a "translation_en_to_fr" pipeline alias), so the
translation and generation models are loaded directly via their
`AutoModelFor...` classes rather than through `pipeline(...)` - the
sentiment pipeline's task name is still registered, so that one keeps
using the convenience `pipeline()` wrapper.
"""

from transformers import (
    pipeline,
    AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM,
)


def run_translation(sentences):
    """Encoder-decoder: full source sequence attended to by a decoder
    that also attends to its own previously generated tokens (cross-attention)."""
    name = "Helsinki-NLP/opus-mt-en-fr"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    results = []
    for s in sentences:
        inputs = tok(s, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=60)
        translation = tok.decode(out[0], skip_special_tokens=True)
        results.append((s, translation))
    return results


def run_sentiment(sentences):
    """Encoder-only: bidirectional self-attention produces one representation
    used for classification (no generation, no causal mask)."""
    clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return list(zip(sentences, clf(sentences)))


def run_generation(prompts, max_new_tokens=40):
    """Decoder-only: masked (causal) self-attention, next-token prediction
    repeated autoregressively - the same sample -> append -> sample again
    loop as bn_lab's n-gram model, just with a transformer computing the
    conditional distribution instead of a count table."""
    name = "distilgpt2"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    results = []
    for p in prompts:
        inputs = tok(p, return_tensors="pt")
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
        text = tok.decode(out[0], skip_special_tokens=True)
        results.append((p, text))
    return results


if __name__ == "__main__":
    print(run_translation(["Artificial intelligence lets machines learn from data."]))
    print(run_sentiment(["I loved this course."]))
    print(run_generation(["The warehouse robot"]))
