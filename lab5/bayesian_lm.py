"""
A tiny n-gram autoregressive language model, viewed as a Bayesian network.

First-order model : P(X_t | X_{t-1})               context length 1
Second-order model: P(X_t | X_{t-2}, X_{t-1})       context length 2

Both are represented as plain Python dicts of counts, with no ML library
and no pretrained model - exactly as the lab specifies. The same counting
machinery (build_counts / counts_to_probs) is reused for both orders so
that comparing them (Part XIII) compares the model, not two independently
written implementations.
"""

import random
from collections import Counter, defaultdict

START = "<START>"
END = "<END>"

DATASET = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug",
]


def tokenize(sentences, order):
    """
    Lower-case, split on whitespace, and pad with `order` <START> tokens
    and a single <END> token.

    Padding with `order` START tokens (not just one) means the context is
    always exactly `order` tokens long, even for the very first real word -
    this keeps the first-order and second-order models structurally
    identical (both just "predict next token from the last `order` tokens"),
    which is what lets the same build_counts() function serve both.
    """
    tokenized = []
    for s in sentences:
        words = s.lower().split()
        tokenized.append([START] * order + words + [END])
    return tokenized


def vocabulary(tokenized_sentences):
    vocab = set()
    for toks in tokenized_sentences:
        vocab.update(toks)
    return vocab


def build_counts(tokenized_sentences, order):
    """C(context, next_token) for context = tuple of `order` preceding tokens."""
    counts = defaultdict(Counter)
    for toks in tokenized_sentences:
        for i in range(order, len(toks)):
            context = tuple(toks[i - order:i])
            nxt = toks[i]
            counts[context][nxt] += 1
    return counts


def counts_to_probs(counts):
    """P(next | context) = C(context, next) / sum_k C(context, k)."""
    probs = {}
    for context, next_counts in counts.items():
        total = sum(next_counts.values())
        probs[context] = {w: c / total for w, c in next_counts.items()}
    return probs


class NgramModel:
    """A counting-based order-N autoregressive language model."""

    def __init__(self, sentences, order):
        self.order = order
        self.tokenized = tokenize(sentences, order)
        self.vocab = vocabulary(self.tokenized)
        self.counts = build_counts(self.tokenized, order)
        self.probs = counts_to_probs(self.counts)

    def context_key(self, context_words):
        """Build a lookup key from a list/tuple of the last `order` words."""
        return tuple(context_words[-self.order:])

    def distribution(self, context_words):
        """Full P(next | context) dict. Empty dict if context unseen."""
        key = self.context_key(context_words)
        return dict(self.probs.get(key, {}))

    def most_probable_next(self, context_words):
        dist = self.distribution(context_words)
        if not dist:
            return None
        return max(dist.items(), key=lambda kv: kv[1])[0]

    def sample_next(self, context_words, rng):
        dist = self.distribution(context_words)
        if not dist:
            return None
        words, weights = zip(*dist.items())
        return rng.choices(words, weights=weights, k=1)[0]

    def generate(self, mode="sample", max_len=20, rng=None):
        """
        mode="sample"  -> Mode B: sample from P(next | context) at each step.
        mode="greedy"  -> Mode A: always take argmax P(next | context).
        Stops when <END> is generated or max_len tokens have been produced.
        """
        rng = rng or random
        context = [START] * self.order
        generated = []
        for _ in range(max_len):
            nxt = (self.most_probable_next(context) if mode == "greedy"
                   else self.sample_next(context, rng))
            if nxt is None or nxt == END:
                break
            generated.append(nxt)
            context = context + [nxt]
        return " ".join(generated)

    def check_normalization(self, tol=1e-9):
        """Returns a list of (context, total_probability) for every context."""
        return [(ctx, sum(dist.values())) for ctx, dist in self.probs.items()]

    def num_parameters(self):
        """Total number of (context -> next_token) probability entries."""
        return sum(len(dist) for dist in self.probs.values())

    def zero_probability_continuations(self, context_words):
        """Vocabulary words that never follow this context (probability 0)."""
        dist = self.distribution(context_words)
        return sorted(w for w in self.vocab if w not in dist and w != START)


if __name__ == "__main__":
    order1 = NgramModel(DATASET, order=1)
    order2 = NgramModel(DATASET, order=2)
    print("Order-1 contexts:", len(order1.probs), "| parameters:", order1.num_parameters())
    print("Order-2 contexts:", len(order2.probs), "| parameters:", order2.num_parameters())
