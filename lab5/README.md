# Lab 5 - Bayesian Networks and Autoregressive Language Models

**Course:** Undergraduate Artificial Intelligence
**Topic:** Viewing an autoregressive language model as a Bayesian network, built with LLM assistance
**Author:** Parth Jhalani

> `bayesian_lm.py` and `run_experiments.py` were written with the assistance of **Claude (Anthropic)**,
> from the prompts in [`prompts.txt`](prompts.txt), and then independently run, inspected, and tested.
> All numbers and generated sentences in this report are taken directly from
> [`full_output.txt`](full_output.txt) (`python3 run_experiments.py`, fixed seed 42), not invented.

## Files

| File | Description |
|---|---|
| `bayesian_lm.py` | `NgramModel`: counting-based order-N autoregressive LM (used for both order 1 and order 2) |
| `run_experiments.py` | Runs every experiment below (Parts IV, VII-X, XIII) |
| `prompts.txt` | The two prompts used with the LLM (Parts V and XII) |
| `full_output.txt` | Raw captured output of `python3 run_experiments.py` |
| `generated_order1.txt` / `generated_order2.txt` | 20 sampled sentences from each model (Part IX) |

## How to Run

```bash
python3 bayesian_lm.py       # quick sanity check: parameter counts for both orders
python3 run_experiments.py   # full experiment suite, reproducible (seed=42)
```

---

## Part I: From Probability to Language

**Question 1 - Why is this decomposition useful for generating text?**
The chain rule turns one intractable object - a joint distribution over an entire sequence, with exponentially many possible sequences - into a product of one-token-at-a-time conditional distributions. Each factor `P(X_t | X_1,...,X_{t-1})` only has to answer "given what's been written so far, what comes next?", which is both estimable from data (by counting what actually follows a given context) and directly usable for generation: sample `X_1`, then sample `X_2` given `X_1`, and so on. Generation becomes a sequence of small, local decisions instead of one impossible global one.

## Part II: A Bayesian Network for Text

**Question 2 - What independence assumption is being made?**
The first-order network `X_1 -> X_2 -> ... -> X_T` assumes each token is conditionally independent of all earlier tokens given only the immediately preceding one:

```
P(X_t | X_1, ..., X_{t-1}) = P(X_t | X_{t-1})    for all t.
```

This is the (first-order) Markov assumption: everything before `X_{t-1}` becomes irrelevant to predicting `X_t` once `X_{t-1}` is known.

## Part III: Build a Small Language Dataset

The dataset used is the one suggested in the handout (`DATASET` in `bayesian_lm.py`):

```
the cat sat on the mat
the cat sat on the rug
the dog sat on the mat
the dog ran to the park
the cat ran to the park
the dog sat on the rug
```

All text is lower-cased and whitespace-tokenized. `tokenize()` pads each sentence with `order` copies of `<START>` and a single `<END>` (see the design note in `prompts.txt` for why `order` copies rather than one - it's what lets one counting function serve both the first- and second-order model).

## Part IV: Constructing the Conditional Probability Table

**Question 3** - `P(next word | current word)` for `the`, `cat`, `dog`, `sat`, `ran` (from `full_output.txt`):

| Context | Distribution |
|---|---|
| `the` | cat: 0.25, dog: 0.25, mat: 0.1667, rug: 0.1667, park: 0.1667 |
| `cat` | sat: 0.6667, ran: 0.3333 |
| `dog` | sat: 0.6667, ran: 0.3333 |
| `sat` | on: 1.0 |
| `ran` | to: 1.0 |

Zero-probability transitions (observed via `zero_probability_continuations`, comparing against the full vocabulary): every word **other** than the ones listed above has probability 0 in each row - e.g. `the` is never followed by `on`, `sat`, `to`, `ran`, itself, or `<END>`; `sat` and `ran` are **deterministic** (`sat -> on` and `ran -> to` both have probability exactly 1.0, so every other word has probability 0 after them).

## Part V: Ask an LLM to Implement the Model

Prompt used: see `prompts.txt`, Prompt 1. Implemented as `NgramModel` in `bayesian_lm.py` with `order=1`.

## Part VI: Inspect the LLM-Generated Code

**Question 4 - Where are the transition counts stored?** In `build_counts()`, a `defaultdict(Counter)` mapping each context tuple to a `Counter` of the tokens observed to follow it - `NgramModel.counts`.

**Question 5 - Where is `P(X_t | X_{t-1})` computed?** In `counts_to_probs()`: for each context, every next-token count is divided by the total count for that context (`NgramModel.probs`).

**Question 6 - How does the program choose the next word? Argmax or sampling?** Both modes are implemented, selected by the `mode` argument of `NgramModel.generate()`: `mode="greedy"` always calls `most_probable_next()` (argmax over the distribution); `mode="sample"` calls `sample_next()`, which uses `random.choices(words, weights=probabilities)` to draw from the full distribution. The difference: argmax is deterministic - the same context always produces the same next word, so greedy generation from a fixed start is fully reproducible but can only ever produce one sentence per model; sampling introduces genuine randomness weighted by probability, so low-probability continuations can still occasionally appear, producing varied output across runs.

**Question 7 - What happens if the program encounters a word for which no transition has been observed?** `distribution()` looks the context up in `self.probs` with `.get(key, {})`, returning an **empty dict** for an unseen context rather than raising an error. `most_probable_next` and `sample_next` both then return `None` for an empty distribution, and `generate()` treats `None` exactly like `<END>` - it stops. (In the current dataset this only happens if `max_len` is reached mid-loop with a genuinely unseen context, since every token that can appear after `<START>` was observed during training; it is handled defensively for the general case anyway.)

## Part VII: Test the Probability Model

`check_normalization()` + `part_vii_normalization()` in `run_experiments.py` sums every context's distribution and checks it is within `1e-9` of 1.0. Result on both models (full output in `full_output.txt`, lines 18-52):

```
All contexts normalize to 1.0: True   (order-1, 11 contexts)
All contexts normalize to 1.0: True   (order-2, 15 contexts)
```

**Question 8 - If one of the totals is 0.87, what does this tell you about the implementation?** It tells you the implementation has a **bug in how the counts are normalized**, not that the model is "mostly right" - a valid probability distribution must sum to exactly 1 (up to floating-point tolerance). A total of 0.87 would mean the denominator used in `C(context, next) / total` isn't actually the sum of *all* counts for that context - for example, some observed continuations were silently excluded from the sum (perhaps due to a filtering bug or a typo'd key), while the numerator still used the full count for the one(s) that were included. This is exactly the kind of silent, plausible-looking-but-wrong behaviour that Task-3-style normalization testing is designed to catch, since a model that "mostly" generates sentible-looking text could still be using an incorrect distribution underneath.

## Part VIII: Predicting the Next Word

(see `full_output.txt`, lines 54-70, reproduced above in Part IV's table for the distributions). `argmax_w P(w | the) = 'cat'`, `argmax_w P(w | cat) = 'sat'`, `argmax_w P(w | dog) = 'sat'`, `argmax_w P(w | sat) = 'on'`, `argmax_w P(w | ran) = 'to'`.

**Question 9 - Are the most probable predictions always the same as the words you would personally expect? What does this tell you about the difference between a probability model and human linguistic expectations?** Mostly yes here, because the dataset is tiny and was hand-picked to be grammatical - `sat -> on` and `ran -> to` are forced (probability 1, only one example in the data), and `the -> cat/dog` matches intuition. But the model has **no notion of grammar, meaning, or plausibility** - it is purely a frequency count over this six-sentence corpus. If the training data happened to contain more instances of some odd pairing, the model would confidently predict it, with no sense that it's unusual. Human expectations draw on world knowledge, semantics, and a lifetime of language exposure; this model only knows "what followed what, how often, in six sentences." The two coincide here only because the toy dataset was constructed to be unambiguous.

## Part IX: Generate Text

20 sampled sentences from each model were generated and saved (`generated_order1.txt`, `generated_order2.txt`; also lines 72-121 of `full_output.txt`). Examples (order-1):

```
the cat sat on the dog ran to the cat sat on the cat sat on the dog ran to   <- ran into max_len, no <END>
the dog sat on the mat
the park
the dog sat on the rug
```

Examples (order-2):

```
the cat sat on the rug
the cat sat on the mat
the dog sat on the mat
the dog ran to the park
```

## Part X: Deterministic vs Probabilistic Generation

(`full_output.txt`, lines 123-157.) Five sentences per mode per model:

**Order-1, greedy:** all 5 runs produce the exact same sentence, and it never terminates on its own:
```
the cat sat on the cat sat on the cat sat on the cat sat on the cat sat on   (truncated at max_len=20)
```
**Order-1, sampling:** 4 distinct sentences out of 5, including very short, less coherent ones like `the park`.

**Order-2, greedy:** all 5 runs produce `the cat sat on the mat` (and this one *does* reach `<END>` and stop).

**Order-2, sampling:** 4 distinct sentences out of 5, e.g. `the cat sat on the rug`, `the dog sat on the mat`, `the dog ran to the park` - all grammatical.

**Question 10 - Compare the two sets of generated sentences. Which mode produces more variation? Why?** Sampling produces more variation in both models (4/5 distinct vs 1/5 distinct for greedy) simply because greedy is a deterministic function of the model and the fixed start context `<START>`: `argmax` always picks the same word given the same distribution, so the entire generated sequence is pinned from the first step onward - there is exactly one possible greedy sentence per model (per starting context). Sampling instead draws from the full distribution at every step, so any continuation with non-zero probability can appear, and different runs diverge as soon as a low-probability alternative happens to get picked. A striking side effect: order-1 greedy generation **never terminates naturally** - `the -> cat -> sat -> on -> the -> cat -> ...` is a perfect cycle in the first-order transition graph (since `argmax P(·|the) = cat`, `argmax P(·|cat) = sat`, `argmax P(·|sat) = on`, `argmax P(·|on) = the`), so greedy decoding loops forever and only stops because `max_len` cuts it off. This is a concrete illustration of why real language models essentially never use pure greedy decoding for open-ended generation.

## Part XI: A Second-Order Bayesian Network

**Question 11 - How does the second-order model differ from the first-order model?**
1. **Graph structure:** instead of a chain `X_1 -> X_2 -> ... -> X_T`, each node has two parents: `X_{t-2} -> X_t <- X_{t-1}`. The network now encodes a *first-order Markov assumption over pairs*, not over single tokens.
2. **Conditional probability table:** indexed by *pairs* of preceding tokens instead of single tokens - the table has rows like `P(· | the, cat)` rather than `P(· | cat)`. Measured directly: 15 distinct order-2 contexts vs 11 distinct order-1 contexts, and 19 total (context -> next) parameters vs 17 (`full_output.txt` line 162-163) on this tiny dataset.
3. **Context available for prediction:** two preceding tokens instead of one, so the model can in principle distinguish situations that look identical under the first-order model (e.g. `the cat` vs `the dog` as a context, instead of collapsing both down to just `the`).
4. **Data needed:** more. The number of possible contexts grows combinatorially with order (roughly |vocab|^order), so each individual context is seen fewer times in a fixed-size corpus - measured directly below in Part XIII, 11/15 second-order contexts have only ever been observed with exactly one possible continuation, vs 8/11 for first-order.

## Part XII: Use the LLM Again (second-order model)

Prompt used: see `prompts.txt`, Prompt 2. In practice this was implemented by generalising `NgramModel` with an `order` parameter rather than writing a second, separate class - see the design note at the end of `prompts.txt`.

## Part XIII: Comparing the Two Models

(`full_output.txt`, lines 159-181.)

| Measure | Order-1 | Order-2 |
|---|---|---|
| Distinct contexts | 11 | 15 |
| Distinct (context -> next) parameters | 17 | 19 |
| Contexts with only one possible continuation | 8 / 11 | 11 / 15 |
| Unique sentences out of 30 sampled | 13 / 30 | 6 / 30 |

**Diversity:** order-1 produces far more *unique* sentences (13/30) than order-2 (6/30) - but, as Part IX shows, several of those order-1 "unique" sentences are short fragments like `the park` or `the mat` that are not complete, well-formed echoes of the training sentences; they're unique mostly because the model is more willing to wander off-template.

**Qualitative coherence:** order-2 generations are consistently well-formed six-to-seven word sentences that closely resemble (and are sometimes identical to) a training sentence - e.g. `the dog ran to the park`, `the cat sat on the rug`. Order-1 generations are noticeably less reliable: alongside coherent sentences it also produces run-ons that loop back on themselves (`the cat sat on the dog ran to the cat sat on the cat sat on the dog ran to ...`, never reaching `<END>` within 20 tokens) and truncated fragments (`the park`, `the mat`) where the model sampled a continuation of `the` that happened to skip straight to a noun that doesn't fit the "the X sat/ran ..." pattern.

**Question 12 - Why does increasing the amount of context potentially improve prediction, and why can it simultaneously make the model harder to estimate from limited data? Relate your answer to the CPT size.**
More context lets the model distinguish situations that look identical with less context - e.g. order-2 knows that after `the cat` the only observed continuations are `sat`/`ran` just as with order-1's `cat` context, but it can *also* tell `on the` apart from `to the`, letting it track grammatical number/structure that `the` alone can't. This directly shows up as higher coherence in Part IX/XIII. The cost is combinatorial: the number of possible contexts grows roughly as `|vocab|^order`, so the same fixed amount of training data gets spread across far more distinct CPT rows - each one seen fewer times. Measured directly above: the order-2 CPT has more rows (15 vs 11) built from the *same six sentences*, so more order-2 contexts end up with only a single observed example (11/15, vs 8/11 for order-1) - i.e. the model has effectively memorized the training data rather than learned a genuine distribution, which is why its generated sentences are so close to verbatim training examples and so much less diverse.

## Part XIV: The Connection to Modern Language Models

No code required for this part - conceptual only, per the handout. The key point carried forward into the final reflection: a modern autoregressive LM still models exactly `P(X_t | X_1, ..., X_{t-1})` via the chain rule; what has changed from this lab's n-gram CPTs is purely *how* that conditional distribution is represented (a neural network instead of a hand-built count table) and *learned* (gradient-based training instead of counting), not the underlying probabilistic structure (`Probability -> Bayesian Network -> Autoregressive Model -> Language Generation`).

## Part XV: Reflection on the Role of the LLM

**Question 13 - Why is Approach B ("implement the following probabilistic model: `P(X_t|X_{t-1})`, estimated from transition counts, with sampling-based generation") preferable to Approach A ("write a Python language model for me") when constructing an intelligent system?**
- *Specifying intended behaviour*: Approach B tells the LLM exactly what distribution to estimate, how (counting), and how to use it (sampling) - this is a testable specification, not a vague goal. Approach A leaves every one of those decisions to the LLM, including ones (e.g., whether to use a library, whether "language model" means an n-gram or something else entirely) that the engineer actually needs to control.
- *Understanding the representation*: because the specification named the CPT explicitly, I knew to look for `build_counts`/`counts_to_probs` and could check they matched `P(w_j|w_i) = C(w_i,w_j)/sum_k C(w_i,w_k)` from the handout - I would not know what to check against if I hadn't specified the model first.
- *Validating the generated implementation*: Part VII's normalization test is only meaningful because I knew in advance what property a correct implementation *must* have (`sum_v P(v|w)=1`); Approach A gives no such reference point.
- *Testing probabilistic invariants*: the greedy-vs-sampling comparison in Part X surfaced a genuine design consequence (order-1 greedy decoding loops forever) that is only interpretable because I understood the generation algorithm well enough to reason about why it happens (a cycle in the argmax transition graph), not just that it happened.
- *Distinguishing implementation from model*: Approach B keeps "the model" (a first/second-order Markov chain over tokens, estimated by counting) conceptually separate from "the code" (a particular Python class). That separation is what let the exact same code (`NgramModel`, parameterised by `order`) implement two different Bayesian networks, and what makes Part XIII a fair comparison of the models rather than of two different programs.

## Part XX: Final Question - What Did the Bayesian Network Add?

**Question 14 - What did thinking of the language model as a Bayesian network give you?**
- **A representation of dependencies:** the diagrams `X_1 -> X_2 -> ... -> X_T` (order-1) and `X_{t-2} -> X_t <- X_{t-1}` (order-2) make explicit exactly which earlier tokens are allowed to influence a prediction - something that's implicit and easy to lose track of in raw code.
- **A factorisation of the joint distribution:** the chain rule turned an intractable joint over whole sentences into a product of small, estimable conditional factors (Part I) - this is the entire reason the model is implementable at all with simple counting.
- **A way to reason about independence assumptions:** stating the first-order Markov assumption in probability notation (Part II, Question 2) made it precise and checkable, rather than an implicit property of "whatever the code happens to do."
- **A principled method for generation:** the network gives an exact generation procedure (`sample X_1, then X_2 ~ P(·|X_1), then X_3 ~ P(·|X_2), ...`) rather than an ad hoc one - `NgramModel.generate()` is a direct, line-for-line implementation of that procedure.
- **A way to test whether an implementation matches its specification:** the normalization test in Part VII and the comparison in Part XIII are only well-defined because the Bayesian-network view specifies exactly what the implementation is *supposed* to compute, independent of whatever code happens to produce it.

---

## Deliverables Summary

1. Python implementation of the first-order model: `NgramModel(DATASET, order=1)` in `bayesian_lm.py`.
2. Python implementation of the second-order model: `NgramModel(DATASET, order=2)` in `bayesian_lm.py`.
3. CPTs for selected contexts: Part IV table above, full detail in `full_output.txt`.
4. Examples of generated text: Part IX above, `generated_order1.txt`, `generated_order2.txt`.
5. Results of probability-normalization tests: Part VII above (`All contexts normalize to 1.0: True` for both models).
6. Answers to Questions 1-14: throughout this document.
7. Reflection on LLM use: Part XV (Question 13) above, including the normalization-test example and the independently-designed `order`-parameterised counting mechanism that both prompts' generated code was built around.
