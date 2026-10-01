# Lab 6 - Bayesian Networks: Building and Learning (LLM + pgmpy)

**Course:** Undergraduate Artificial Intelligence
**Topic:** Specifying a Bayesian network, generating its implementation with an LLM, and validating the result against an independent oracle
**Author:** Parth Jhalani

> `llm_generated.py` (the network builder and the two fitting functions) was written with the
> assistance of **Claude (Anthropic)** from the prompts in [`prompts.txt`](prompts.txt).
> `trusted_model.py` (including the brute-force oracle) was written independently, first,
> specifically so it could validate everything produced afterwards. No local LLM weights were
> downloaded for this lab - see the note at the end of `prompts.txt` for why that mirrors the
> reference notebook's own setup. All numbers below come from actually running the code
> (`python3 bn_pipeline.py`, captured verbatim in [`full_output.txt`](full_output.txt)).

## Files

| File | Description |
|---|---|
| `trusted_model.py` | Hand-written pgmpy network + independent brute-force posterior oracle (the test ground truth) |
| `llm_generated.py` | LLM-generated network builder, MLE/Bayesian fitting functions, and two deliberately-buggy models |
| `validate.py` | Reusable structural/numerical/semantic tests, usable on any candidate model |
| `bn_pipeline.py` | Runs the entire lab end to end |
| `prompts.txt` | The three prompts used with the LLM |
| `full_output.txt` | Raw captured output of `python3 bn_pipeline.py` |

## How to Run

```bash
python3 -m venv ../.venv && source ../.venv/bin/activate && pip install pgmpy numpy pandas
python3 bn_pipeline.py
```

---

## The network

```
C -> R,   C -> S,   R -> W,   S -> W
P(C,R,S,W) = P(C) P(R|C) P(S|C) P(W|R,S)
```
C=Cloudy, R=Rain, S=Sprinkler, W=WetGrass; 0=False, 1=True. Parameters as specified in the lab (see `trusted_model.py`).

## Steps 1-3: Trusted model, exact inference, independent oracle

`trusted_model.py` builds this network by hand with pgmpy `TabularCPD`s, and separately implements `enumerate_posterior()`, which computes any `P(query=1 | evidence)` by brute-force summation over all 2⁴=16 assignments directly from the factorisation - no pgmpy inference code involved. This is the "we trust ourselves first" step: a second, independent implementation to check the library against.

```
P(R=1|W=1):  pgmpy VariableElimination = 0.7048   independent enumeration = 0.7048
```
They agree to 9 decimal places (`full_output.txt` line 14-15) - pgmpy's exact inference matches a hand-derived calculation on this network.

## Step 4: Validate the trusted model with reusable tests

`validate.py`'s `run_all_checks()` checks structure (nodes/edges/acyclicity), that `check_model()` passes, that every CPD's columns individually sum to 1, and that a sample posterior matches the oracle. On the trusted model, everything passes (lines 21-25).

## Step 5: The "LLM-generated" network (Prompt 1) vs the trusted model

Using Prompt 1 from `prompts.txt`, the generated `build_network_from_prompt1()` in `llm_generated.py` passes every check in `validate.py`, and its four CPDs are numerically identical to the trusted model's (`full_output.txt` lines 36-39: `CPD(C/R/S/W) identical to trusted model: True`). For this prompt, nothing needed correcting - but that was only knowable because of Step 4's checks, not because the code "looked right" or ran without error.

## Step 6-7: Synthetic data and maximum-likelihood estimation (Prompt 2)

5000 rows were sampled from the **trusted** model (`BayesianModelSampling`, fixed seed), to generate data with a known ground truth. `fit_mle()` (Prompt 2) was then used to re-estimate the CPTs from that data alone - no access to the true parameters.

```
Hand count:  P_hat(R=1|C=1) = 1959/2459 = 0.7967
pgmpy MLE :  P_hat(R=1|C=1) = 0.7967
True value:  P(R=1|C=1)     = 0.8000
```
The library's MLE estimate matches a from-scratch count-and-divide calculation exactly (both 0.7967), and both are close to the true 0.8 - as expected with 5000 samples.

## Step 8: Estimation stability vs sample size

| N | P̂(R=1\|C=1) | \|error\| |
|---|---|---|
| 20 | 1.0000 | 0.2000 |
| 50 | 0.7273 | 0.0727 |
| 200 | 0.8544 | 0.0544 |
| 1000 | 0.8004 | 0.0004 |
| 5000 | 0.7980 | 0.0020 |
| 20000 | 0.7973 | 0.0027 |

At N=20, every single sampled instance with C=1 happened to also have R=1, giving the (wildly overconfident) estimate 1.0 - a concrete illustration of why small samples can't be trusted. The error shrinks roughly as N grows (not perfectly monotonically - 5000 happens to have a marginally larger error than 1000 here, which is expected: MLE is unbiased in expectation, not monotonically improving on every individual random draw), consistent with the Law of Large Numbers.

## Step 9: Sparse data and Bayesian (BDeu) estimation

With only 8 sampled rows, the combination (C=1, S=1) never occurred:
```
Observed (C,S) combinations: [(0, 0), (0, 1), (1, 0)]  out of 4 possible
```
MLE CPD(S):       P(S=1|C=1) = **0.0** (exactly, because it was never observed)
Bayesian/BDeu CPD(S): P(S=1|C=1) = **0.357** (regularized toward the prior, equivalent_sample_size=10)

This is precisely the sparse-data failure mode the lab describes: MLE produces a hard, overconfident 0 (implying sprinkler use is *impossible* when cloudy, which is far too strong a claim from one small sample), while the Dirichlet/BDeu prior pulls the estimate back toward a less extreme value by blending in pseudo-counts. Neither is "more correct" in an absolute sense - the Bayesian estimate encodes an explicit modelling assumption (the prior) that MLE does not.

## Step 10: A deliberately broken Bayesian network

`build_broken_cpt_model()` sets one WetGrass column to sum to 1.05 instead of 1.0.
```
REJECTED as expected: ValueError: Sum or integral of conditional probabilities for node W is not equal to 1.
```
pgmpy's own `check_model()` (called inside `add_cpds`/`check_model`) catches this immediately - a structurally invalid CPT is rejected before it can be used.

## Step 11: A subtler failure - valid numbers, wrong evidence order

`build_reversed_evidence_order_model()` swaps the two columns of CPD(R|C) - as if the generated code had assumed the evidence-state order was (C=1, C=0) rather than (C=0, C=1). Every individual column still sums to 1, so:
```
model.check_model() result: True   <- passes!
Semantic test (posterior vs independent oracle): {'pgmpy': 0.8464, 'oracle': 0.7048, 'match': False}
```
`check_model()` has nothing to object to - every number is a valid probability and every column is normalized. But the model now represents `P(R=1|C=1)=0.2` instead of `0.8` (the parent-state order is backwards), and the semantic test - comparing a real query's answer against the independent oracle - catches it immediately (0.8464 vs the true 0.7048). This is exactly the lab's central point: **a program that runs and passes a structural check can still represent the wrong distribution.**

## What exactly did the LLM contribute?

| Task | Responsible |
|---|---|
| Interpret the natural-language specification (Prompts 1-3) | LLM |
| Generate candidate `pgmpy` code | LLM |
| Represent the Bayesian network, perform inference, fit parameters | `pgmpy` |
| Decide the API actually matches the installed pgmpy version (1.1.2's `parameter_estimator.DiscreteMLE`/`DiscreteBayesianEstimator`, not the older `estimators.MaximumLikelihoodEstimator` class-based call) | discovered by running the code and reading the `TypeError`, then fixed - human, with the LLM's docstring-reading help |
| Build the independent brute-force oracle | written first, independently, by hand |
| Decide whether a generated model is semantically correct | `validate.py`'s oracle-comparison test + human review |

## HW exercise: new queries on the unmodified (trusted) network

| Query | Naive prediction | Computed | Prediction correct? |
|---|---|---|---|
| P(S=1\|W=1) | should increase (S directly causes W) vs prior 0.5 | **0.4278** (decreased) | **No** |
| P(C=1\|W=1) | should increase vs prior 0.5 | **0.5746** (increased) | Yes |
| P(R=1\|W=1,S=0) | should be noticeably higher than P(R=1\|W=1)=0.7048 | **0.9922** | Yes |

The R and C predictions were correct, but the S prediction was **wrong**, and the reason is a genuinely useful lesson rather than an error to paper over: `S` and `C` are *negatively* correlated in this network (`P(S=1|C=1)=0.1` vs `P(S=1|C=0)=0.5` - sprinklers are less likely to be needed when it's cloudy), while `R` and `C` are *positively* correlated (`P(R=1|C=1)=0.8`). Observing `W=1` raises the posterior belief that `C=1` (since cloudy weather makes rain, and hence wet grass, much more likely via the strong `R` link). That increased belief in `C=1` then pulls `P(S=1)` **down** through the `C -> S` link, competing against the direct, intuitive `S -> W` effect that would push it up. In this network the indirect "common cause" pull wins, and the naive single-step intuition ("wet grass is evidence for every possible cause") turns out to be incomplete once there's a shared parent `C` tying the two potential causes together - which is exactly why the lab asks you to predict *then compute*, rather than just compute.

## Classroom exercise: explain the WetGrass CPT columns

With `evidence=["R", "S"]` and `evidence_card=[2, 2]`, pgmpy orders the CPD's columns as the Cartesian product of evidence states with the **first-listed evidence varying slowest** - i.e. columns are, in order: `(R=0,S=0)`, `(R=0,S=1)`, `(R=1,S=0)`, `(R=1,S=1)`. This was verified directly, not just asserted: `trusted_model.py`'s CPD values `[[0.99,0.10,0.10,0.01],[0.01,0.90,0.90,0.99]]` reproduced the exact posterior from `enumerate_posterior()`, which independently enumerates `(c,r,s,w)` and indexes `W[(r,s)][w]` using that same `(R,S)` pairing - and Step 11 above shows what happens when a CPD's column order is wrong: it still looks like a valid, normalized table, but represents a different distribution, caught only by comparing against ground truth. Verifying library conventions like this against an independent calculation, rather than trusting documentation prose alone, is the actual check - "explain it" and "verify it reproduces the right numbers" are different tasks, and only the second one is a real test.

## Classroom exercise: estimate-variability across seeds

10 independent draws of N=200 samples each, MLE-fitting P(R=1|C=1) (true value 0.8) each time:
```
Mean = 0.8012, Std = 0.0320, True = 0.8000
```
Individual estimates ranged from 0.7553 to 0.8544 - a spread of about ±0.03-0.05 around the truth purely from sampling variability, even though every dataset came from the exact same model. The correct explanation for this spread is **sampling variability** (finite-sample noise in which particular rows happened to be drawn), not any inconsistency in pgmpy or in the LLM-generated fitting code - the same `fit_mle()` function was used for every seed, and its mean across seeds (0.8012) is very close to the true 0.8, exactly as expected for an unbiased estimator averaged over many draws.

## Where could the system fail? (failure modes actually observed or deliberately induced in this lab)

- **Ambiguous specification -> API mismatch:** Prompt 2 as initially written did not pin down which pgmpy API to call; the generated code used a class-based `estimator=MaximumLikelihoodEstimator` call that is not what this installed pgmpy version (1.1.2) expects, and it raised a `TypeError` rather than silently doing the wrong thing - the safer kind of failure, caught before producing a wrong number.
- **Structurally invalid CPT (Step 10):** caught immediately by `check_model()`.
- **Semantically wrong but structurally valid CPT / reversed parent order (Step 11):** **not** caught by `check_model()`; only caught by comparing a query result against an independent oracle.
- **Sparse data producing overconfident 0/1 estimates (Step 9):** not a code bug at all - a property of MLE under small samples - but easy to mistake for a bug if the 0.0 isn't expected.
- **Sampling noise mistaken for a bug (Step 13):** an estimate of 0.7553 instead of 0.8 is not wrong code - it's exactly what sampling variability looks like, and the fix is more data or averaging, not more debugging.

## A compact evaluation rubric, applied to this lab's generated code

| Check | Result here |
|---|---|
| Correct variables/edges, acyclic | Pass (Step 4/5) |
| Each CPD normalized | Pass for trusted + Prompt-1 model; **fails loudly** for the deliberately broken one (Step 10) |
| Evidence variables and parent-state order correct | Pass for trusted + Prompt-1 model; **passes `check_model()` but fails semantically** for the reversed-order model (Step 11) - this is the one that needs a dedicated oracle test, not just structural checks |
| Posterior agrees with a trusted/hand calculation | Verified directly throughout (Steps 1, 5, 11) |
| Requested estimator actually used, prior explicit when requested | Verified: `fit_bayesian()`'s BDeu prior with `equivalent_sample_size=10` visibly changes the sparse-data estimate relative to MLE's (Step 9) |
| Estimates compared on the same dataset | Step 9 fits MLE and Bayesian estimators on the identical 8-row sample |
| Generated code inspected before execution | `llm_generated.py` was read and its API calls corrected (see "What exactly did the LLM contribute?") before any result in this report was trusted |

## Recap

The Bayesian network `P(C,R,S,W) = P(C)P(R|C)P(S|C)P(W|R,S)` is the scientific object throughout. The LLM translated natural-language specifications of that network, an MLE fitting routine, and a Bayesian fitting routine into `pgmpy` code (Prompts 1-3). None of that code was trusted on the strength of running without error: Step 4's structural tests, the independent brute-force oracle in `trusted_model.py`, and the deliberately broken/subtly-wrong models in Steps 10-11 together establish the actual lesson of this lab - `program runs != probabilistic model is correct != scientific assumptions are appropriate` - and every number in this report is one that was checked against at least one of those tests, not merely produced.
