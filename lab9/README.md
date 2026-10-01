# Lab 9 - Transformers: Architecture, Hands-On Use, and Local LLMs

**Course:** Undergraduate Artificial Intelligence
**Topic:** The transformer architecture, used hands-on across its three architectural families, plus a locally-run open-weights model compared against Claude
**Author:** Parth Jhalani

> `transformer_demos.py` was written with the assistance of **Claude (Anthropic)** from the
> prompt in [`prompts.txt`](prompts.txt), then run and one API mismatch was diagnosed and fixed
> (see `prompts.txt`). All output below is from [`full_output.txt`](full_output.txt)
> (`python3 run_experiments.py`).

## Files

| File | Description |
|---|---|
| `transformer_demos.py` | Encoder-decoder (translation), encoder-only (sentiment), decoder-only (generation) demos |
| `run_experiments.py` | Runs all demos plus the local-LLM-vs-Claude comparison |
| `prompts.txt` | The prompt used with the LLM, the API-mismatch fix, and the Ollama substitution decision |
| `full_output.txt` | Raw captured output of `python3 run_experiments.py` |

## How to Run

```bash
python3 -m venv ../.venv && source ../.venv/bin/activate
pip install torch transformers sentencepiece sacremoses --index-url https://download.pytorch.org/whl/cpu
python3 run_experiments.py
```

---

## Important Concepts

**Attention mechanism:** a way to produce, for each position in a sequence, a weighted combination of *value* vectors from other positions, where the weights (how much each other position matters) are computed by comparing a *query* vector (from the position asking "what do I need?") against every position's *key* vector (what each position "offers"), typically via a scaled dot product followed by softmax. This replaces the fixed, position-based mixing of convolutions or the strictly sequential mixing of RNNs with a learned, content-based, all-pairs mixing.

**Self-attention:** attention where the queries, keys, and values all come from the *same* sequence - every token looks at every other token *in that same sequence* to decide how to update its own representation. This is what lets a transformer encoder directly relate, say, a pronoun to the noun it refers to several words away, in a single layer, regardless of distance.

**Cross-attention:** attention where the queries come from one sequence (the decoder's current partial output) but the keys and values come from a *different* sequence (the encoder's output). This is literally how the decoder "looks back at the source sentence" during translation - visible as the `Multihead cross-attention` block in the architecture diagram, taking `K,V` from the encoder stack and `Q` from the decoder stack.

**Multihead attention:** running several attention "heads" in parallel, each with its own learned query/key/value projections, then concatenating their outputs. Different heads can specialise in different kinds of relationships (e.g. one head tracking syntactic agreement, another tracking coreference) rather than forcing a single attention pattern to capture everything at once.

**Positional encoding:** attention itself is permutation-invariant (it has no built-in notion of order - swapping two input tokens would, without this, just swap two rows of the output symmetrically), so a vector encoding each position (fixed sinusoidal functions in the original paper, or a learned embedding) is added to each token's input embedding, giving the model the sequence order information it otherwise has no way to recover.

**Mask (masked) attention:** restricting which positions a query is allowed to attend to. A *causal* mask (used in the decoder's "Masked multihead self-attention" block) prevents position `t` from attending to any position `>t`, which is essential for autoregressive generation - at training time, the model must not be able to "cheat" by looking at the very token it is being trained to predict next; at inference time, future tokens don't exist yet anyway.

## Hands-On: One Demo per Architecture Family

### Encoder-decoder: machine translation (`Helsinki-NLP/opus-mt-en-fr`)

```
EN: Artificial intelligence lets machines learn from data.
FR: L'intelligence artificielle permet aux machines d'apprendre des données.

EN: The warehouse robot searches for the shortest path to the goal.
FR: Le robot de l'entrepôt recherche le chemin le plus court jusqu'au but.

EN: A Bayesian network represents conditional independence assumptions.
FR: Un réseau bayésien représente des hypothèses d'indépendance conditionnelle.
```
All three translations are fluent and semantically accurate (verified by reading the French - "chemin le plus court" = "shortest path", "hypothèses d'indépendance conditionnelle" = "conditional independence assumptions"). The encoder runs bidirectional self-attention over the full English sentence at once; the decoder generates French tokens one at a time, each step using masked self-attention over the French tokens produced so far *and* cross-attention over the complete encoded English sentence - exactly the architecture diagram's two attention blocks inside the decoder stack.

### Encoder-only: sentiment analysis (`distilbert-base-uncased-finetuned-sst-2-english`)

```
POSITIVE (0.9999)  I loved this course, the labs were genuinely interesting.
NEGATIVE (0.9998)  The assignment was confusing and the deadline was far too short.
POSITIVE (0.9825)  The lecture covered Bayesian networks today.
```
The first two are correctly and confidently classified. The third is an interesting edge case: a purely factual, affectively neutral statement ("the lecture covered X today") gets classified POSITIVE with high confidence (0.98) - a useful, concrete reminder that this model was fine-tuned on a specific labelled dataset (movie reviews, SST-2) for a *binary* positive/negative task with no "neutral" option, so a neutral sentence is forced into whichever of the two classes the model finds marginally more likely, which may not reflect the sentence's actual affective content. The encoder uses bidirectional self-attention (every token can attend to every other token, before and after it - no causal mask), which is appropriate here since the whole sentence is available at once and the task is to produce one classification for it, not to generate new tokens.

### Decoder-only: GPT-style generation (`distilgpt2`)

```
Prompt: 'The warehouse robot'
Continuation: 'The warehouse robot is a robot that can be used to perform tasks such as
               cleaning, cleaning, and cleaning.'

Prompt: 'Artificial intelligence is'
Continuation: 'Artificial intelligence is a new technology that is being developed by
               researchers at the University of California, Berkeley.'
```
Both continuations are grammatical but exhibit the repetition typical of small, non-instruction-tuned base language models under greedy decoding (`do_sample=False`) - "cleaning, cleaning, and cleaning" is a clear example of the model settling into a locally-high-probability loop. This is the same generic next-token-prediction mechanism as Lab 5's hand-counted n-gram model (`sample -> append -> sample again`, stopping at a length limit), just with a transformer's masked self-attention computing a vastly richer `P(next token | previous tokens)` than a count table ever could - and, as in Lab 5's order-1 greedy generation getting stuck in a loop, greedy decoding here is also visibly prone to repetitive loops, for a related reason: at each step it deterministically picks the single highest-probability continuation, with no mechanism to avoid a token sequence it has already produced.

## Local LLM vs Claude

**On Ollama specifically:** the handout's own instructions (install Ollama, run a local GPT-type model, compare responses to GPT/Claude) were genuinely attempted - see `prompts.txt` for the full diagnosis. In short: the current Ollama Linux release is a 1.4 GB multi-backend (CUDA+ROCm+CPU) bundle against an ~18 GB free-disk budget, and the official installer additionally requires `sudo` (no passwordless sudo available in this environment). Per the pre-agreed fallback, a small open-weights **instruction-tuned** model (`Qwen/Qwen2.5-0.5B-Instruct`, ~1 GB) was run directly through `transformers` instead - the same exercise (prompt a small local open model, compare it against Claude), different serving mechanism.

Three AI/course-relevant prompts, local model's answer vs. my own (Claude's) answer to the identical prompt:

**Prompt 1: "In one sentence, what is self-attention in a transformer?"**
- Qwen2.5-0.5B-Instruct (19.0s on CPU): *"Self-attention is a mechanism within transformers that allows the model to focus on specific parts of the input sequence based on their context, improving its ability to understand and generate natural language."*
- Claude (me): *"Self-attention updates each token's representation by taking a weighted average of every token's value vector in the same sequence, where the weights come from comparing that token's query against every token's key."*
- **Assessment:** both are correct in spirit; the local model's answer is vaguer ("focus on specific parts... based on their context") and never names the query/key/value mechanism that actually defines *how* the focusing happens, while mine states the actual mechanism. Not wrong, but noticeably less precise.

**Prompt 2: "In one sentence, why does A* search use a heuristic?"**
- Qwen2.5-0.5B-Instruct (22.9s): *"A* search uses a heuristic to estimate the cost of reaching the goal from a given node in the search tree, which helps guide the algorithm towards the optimal path."*
- Claude (me): *"A* uses a heuristic h(n) estimating the remaining cost to the goal so it can prioritize expanding states that are plausibly closer to the goal first, searching far fewer states than blind search while (if h is admissible) still being guaranteed to find an optimal path."*
- **Assessment:** both are reasonably accurate here - this is a well-known, well-documented topic, and a 0.5B-parameter model handles it adequately.

**Prompt 3: "In one sentence, what does it mean for a heuristic to be admissible?"**
- Qwen2.5-0.5B-Instruct (16.7s): *"A heuristic is considered admissible if it provides an optimal or near-optimal solution within the constraints of the problem being solved."*
- Claude (me): *"A heuristic h(n) is admissible if it never overestimates the true remaining cost to the goal from n, i.e. h(n) ≤ h*(n) for every state n - this is precisely the condition that guarantees A* using it finds an optimal solution."*
- **Assessment:** this one is a genuine, concrete example of a small local model giving an **imprecise/incorrect** answer: admissibility is a specific mathematical property of the heuristic function itself (never overestimates true cost), not a vague claim that the search "provides an optimal or near-optimal solution" - that confuses the *property of the heuristic* with a *consequence of A* using an admissible heuristic*, and even that consequence is stated too loosely ("or near-optimal" dilutes the actual guarantee, which is exact optimality, not "near"). This is exactly the kind of answer an engineer must not accept uncritically - it sounds plausible (fluent, on-topic, confident) while being imprecise on the one detail (`h(n) ≤ h*(n)`) that actually defines the term, echoing this submission's running theme from Labs 4, 6, 7, and 8: a plausible-sounding answer is not the same as a correct one, whether it comes from generated code or a generated natural-language explanation.

**Takeaway:** a 0.5B-parameter local model is fast (15-25s per answer on CPU, no network dependency after the one-time download) and perfectly serviceable for well-known, frequently-documented concepts, but its answers should be treated the same way this entire submission treats any LLM output - as a draft to be checked against the actual definition, not accepted on fluency alone. This is precisely the "AI Science vs AI Engineering" / "program runs ≠ output is correct" lesson that runs through every other lab in this submission, now applied to natural-language explanations instead of generated code or generated probabilistic models.
