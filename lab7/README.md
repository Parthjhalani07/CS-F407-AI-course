# Lab 7 - Neural Models: Learning, Depth, Activations, and Output Layers

**Course:** Undergraduate Artificial Intelligence
**Topic:** A minimal 2-2-1 PyTorch MLP for an XOR-like decision problem, built with LLM assistance
**Author:** Parth Jhalani

> `xor_model.py` was written with the assistance of **Claude (Anthropic)** from the prompt in
> [`prompts.txt`](prompts.txt), based on the design specified independently in Task 2 below, and
> then run, inspected, and diagnosed independently (Task 4). All numbers in this report come
> from [`full_output.txt`](full_output.txt) (`python3 run_experiments.py`), including a training
> run that initially **failed** to learn - reported honestly rather than only showing the fix.

## Files

| File | Description |
|---|---|
| `xor_model.py` | `XORNet` (2-2-1, binary), `ThreeClassNet` (2-2-3), training loops, zero-init helper |
| `run_experiments.py` | Runs every experiment: Task 4 Parts A-D and Task 5 |
| `prompts.txt` | The prompt used with the LLM (Task 3) |
| `full_output.txt` | Raw captured output of `python3 run_experiments.py` |

## How to Run

```bash
python3 -m venv ../.venv && source ../.venv/bin/activate && pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 run_experiments.py
```

---

## Task 1: Understand the Problem Before Coding

**Input/output space and the four labelled examples:**
𝒳 = {0,1}², 𝒴 = {0,1}. Examples: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0.

**Sketch (described, since this is a text file):** plotting (x₁,x₂) on a unit square, the two class-0 points (0,0) and (1,1) sit on one diagonal, and the two class-1 points (0,1) and (1,0) sit on the other diagonal.

**Why can't one straight line separate the two classes?** Any single straight line through the plane divides it into two half-planes. For the line to separate {(0,0),(1,1)} from {(0,1),(1,0)}, it would have to place two points that lie on the same diagonal into different half-planes than the two points on the other diagonal - but (0,0) and (1,1) are on opposite corners (as are (0,1) and (1,0)), so no single linear boundary can have both class-0 points strictly on one side and both class-1 points strictly on the other; every line either mixes classes or fails to separate at all. This is the standard statement of XOR's non-linear-separability.

**Prediction for a single affine transform + sigmoid output:** it should fail to reach 100% accuracy in general - a sigmoid output on top of one affine layer (`sigmoid(Wx+b)`) can only represent a single linear decision boundary (thresholding a hyperplane), which was just shown to be insufficient for XOR. At best it would settle for being "least wrong" on average (e.g. predicting around 0.5 everywhere, or getting at most 3 of 4 points right), never a confident, correct 4/4.

**Think About It - what scientific claim does XOR let us test with four points?** That the number of parameters is not what determines whether a function class can represent a target concept - *representational capacity* depends on the functional form (here, whether a nonlinearity sits between two affine maps), not parameter count. A linear model can have arbitrarily many parameters (through the affine map's weights) and still never represent XOR; a 2-2-1 network with a nonlinear hidden layer has only slightly more parameters and can represent it exactly. Four points are enough because XOR is the smallest concept that is provably not linearly separable, making it a clean, minimal test of this claim rather than requiring a large dataset to notice the limitation.

## Task 2: Design the Intelligent Agent

**Design (before using an LLM):** 2 inputs → 2 hidden units (nonlinear activation, sigmoid/tanh/relu selectable) → 1 output logit; `BCEWithLogitsLoss` (sigmoid + binary cross-entropy, computed in a numerically stable fused form rather than as two separate operations); gradient-based optimisation via full-batch gradient descent over the 4 fixed examples.

1. **Why is the hidden nonlinearity scientifically necessary here?** Without it, `output = W2(W1x+b1)+b2 = (W2W1)x + (W2b1+b2)` is still just one affine map in `x` - stacking affine layers without a nonlinearity between them algebraically collapses to a single affine layer, which Task 1 showed cannot represent XOR. The nonlinearity is what lets the hidden layer carve the input space into regions that a single subsequent linear layer can then separate.
2. **Why is sigmoid + binary cross-entropy a sensible engineering pairing for the output?** The target `y` is a single yes/no label, so the model should output a value interpretable as `P(y=1|x) ∈ (0,1)` - exactly what a sigmoid provides. Binary cross-entropy is the negative log-likelihood of a Bernoulli label under that probability, so minimizing it is maximum-likelihood estimation for this task, and its gradient with respect to the logit is the clean `p - y` (no vanishing-gradient term from differentiating sigmoid and log separately, which is exactly why PyTorch fuses the two into `BCEWithLogitsLoss`).
3. **What evidence counts as successful learning? (at least three checks)** (i) final loss much smaller than the initial loss (and specifically much smaller than `ln 2 ≈ 0.693`, the loss of an "always predict 0.5" network); (ii) all four thresholded predictions matching the four targets exactly; (iii) the hidden-layer weight gradient being non-zero and changing across training steps (confirms backprop is actually producing a learning signal, not just running with no effect); (iv) repeated-run behaviour - does a fresh random initialization reliably reach the same correct solution, or does it depend heavily on luck?

**Think About It - what has determined what each hidden unit computes?** Since XOR is symmetric under swapping `x1`/`x2` and under relabelling which hidden unit is "first", there is no single correct assignment of "concept" to each hidden unit - backpropagation, not any explicit target, determines which (arbitrary, up to the problem's symmetries) linear-plus-nonlinearity split of the input space each of the two hidden units settles into, driven purely by which direction reduces the loss fastest from that unit's particular random starting point. This is demonstrated concretely in Task 4 Part C: with identical starting weights, the two hidden units stay identical forever, because nothing in the gradient computation ever distinguishes them.

## Task 3: Use an LLM to Generate a First Implementation

Prompt used: see [`prompts.txt`](prompts.txt). Implemented as `XORNet` / `train_binary` in `xor_model.py`.

**Where things are in the generated code:** forward pass - `XORNet.forward()` (`self.act(self.hidden(x))` then `self.output(...)`); scalar loss - `loss_fn(logits, Y_BINARY)` in `train_binary`, using `BCEWithLogitsLoss`; reverse-mode AD - `loss.backward()`; optimiser step (parameter update) - `optimizer.step()` in `train_binary`.

**Think About It - which parts of this lab could be verified from the code without running it, and which require execution?** Verifiable by reading alone: the architecture shape (2→2→1, consistent with the design), the loss/activation pairing (`BCEWithLogitsLoss` matches the "sigmoid output, BCE loss" choice), and that `zero_init_` really does zero every parameter (`nn.init.zeros_` on every `model.parameters()` entry). **Not** verifiable without execution: whether training actually converges (Task 4 Part A's first attempt looked completely reasonable as code and still got stuck in a local minimum - nothing in the source would reveal that), what the actual gradient magnitudes are, and whether two activations behave differently from identical weights - these are all empirical facts about the trained numbers, not about the code's structure.

## Task 4: Execute, Test, and Diagnose the Generated Code

### Part A - Basic learning check

**First attempt** (plain SGD, seed=0, lr=0.5, 3000 steps): final loss **0.690318**, **not** all 4 correct. The loss sits essentially at `ln 2 = 0.693` - the network learned to output ≈0.5 for every input and never escaped this flat region. This is a known failure mode of sigmoid-hidden XOR networks under plain gradient descent from certain initializations, not a bug in the generated code.

**Fix, following the task's own instructions ("change only justified engineering settings"):** switched the optimiser from SGD to Adam and changed the seed (architecture, loss function, and data all left untouched). **Second attempt** (Adam, seed=1, lr=0.1, 3000 steps): initial loss 0.7028 → final loss **0.000093**.

| input | probability | prediction | target |
|---|---|---|---|
| (0,0) | 0.0001 | 0 | 0 |
| (0,1) | 0.9999 | 1 | 1 |
| (1,0) | 0.9999 | 1 | 1 |
| (1,1) | 0.0001 | 0 | 0 |

All 4 predictions correct.

### Part B - Backpropagation check

Final hidden-layer weight gradient (`model.hidden.weight.grad`, after the last `backward()` call of the successful run):
```
[[-1.37e-07, -1.18e-07],
 [-1.12e-06, -1.28e-06]]
```
`parameter.grad` is PyTorch's reverse-mode-autodiff computation of `∂L/∂W^(1)` - the partial derivative of the scalar loss with respect to every entry of the first-layer weight matrix, obtained by applying the chain rule backward through `output → hidden activation → hidden affine map`, exactly the backpropagation algorithm from the lecture. It is tiny here specifically *because* training has already converged - near a minimum, `∂L/∂W` should approach zero; gradient norms recorded earlier in training were far larger (e.g. 0.0043 at step 0 vs ~0.0004 at step 20, shrinking monotonically as the loss surface flattens near the optimum).

Since `BCEWithLogitsLoss` (like most PyTorch losses by default) returns the **mean** loss over the batch, `∂(mean loss)/∂W = (1/N) Σᵢ ∂Lᵢ/∂W` - the recorded gradient is the average of the four individual examples' per-example gradients, by linearity of differentiation; this is also why full-batch gradient descent here is equivalent to a single "batch gradient descent" step over all four examples at once, not four separate per-example updates.

### Part C - Symmetry experiment

All weights **and** biases set to exactly zero before training (both must be zeroed - zeroing only the weights but leaving PyTorch's default random bias initialization in place would itself break the symmetry; see `zero_init_()` in `xor_model.py`).

```
step    0: row0=[0.0, 0.0]  row1=[0.0, 0.0]  identical=True
...
step    7: row0=[0.0, 0.0]  row1=[0.0, 0.0]  identical=True
After 200 steps, are the two hidden rows still identical? True
Final loss (zero-init run): 0.693147   (= ln 2, exactly)
All 4 predictions correct: False
```

The two hidden-layer weight rows remain **exactly identical through all 200 steps**, and the loss converges to exactly `ln 2`. This is the expected, explainable result: with identical weights and biases, the two hidden units receive identical inputs and compute identical outputs at every step (`h_A = h_B` for every `x`, since they apply the exact same affine map and the same activation function); and since the loss is a differentiable function of `(h_A, h_B)` that is symmetric in the two (swapping which hidden unit is "A" and which is "B" changes nothing about the output), the gradient with respect to each unit's weights must also be identical - `∂L/∂W_A = ∂L/∂W_B` at every step, by symmetry of the computation graph itself. With identical gradients applied by the same optimizer, two initially-identical rows can never diverge: gradient descent provides no mechanism to break a symmetry it did not start with. The network is permanently stuck behaving as if it had only one effective hidden unit, which - as Task 1 established - is architecturally incapable of representing XOR; hence it converges to the best any single-linear-unit-plus-output model can do (constant 0.5 output, loss = `ln 2`), not to a correct solution.

### Part D - Activation experiment

Same initial weights for all three runs (`seed=100`, fixed before each model is constructed - only the hidden activation function differs), plain SGD, lr=0.5, 3000 steps:

| Hidden activation | Final loss | 4/4 correct? | Early ‖∇W⁽¹⁾L‖₂ (step 0) |
|---|---|---|---|
| Sigmoid | 0.017066 | **True** | 0.026860 |
| Tanh | 0.347547 | False | 0.051409 |
| ReLU | 0.346703 | False | 0.075095 |

From this **one specific shared initialization**, sigmoid converges to a correct solution while tanh and ReLU both plateau near loss ≈0.3466 (verified stable even at 10,000 steps, not merely slow - a genuine local minimum for this starting point, not a convergence-speed issue). This is reported as an observation about *this experiment*, not a general ranking of activations (as the lab explicitly warns against) - a different seed earlier in this very report (Task 4 Part A) showed sigmoid itself getting stuck for *other* initializations, and it is well documented in the literature that ReLU and tanh often train faster than sigmoid in deeper networks precisely because they don't saturate as readily. What the early gradient norms do support, independent of which run ultimately converges: ReLU's is the largest (0.075) and sigmoid's the smallest (0.027) at step 0, consistent with the background material's point that an active ReLU unit contributes a local derivative of exactly 1, while sigmoid's derivative is at most 0.25 (at `a=0`) and shrinks toward 0 as its input saturates - sigmoid structurally contributes smaller per-layer derivative factors than ReLU does, even though that alone does not determine which run happens to converge from a given starting point.

**Think About It - distinguishing saturation from a dead ReLU:** both can produce a near-zero gradient, but they are distinguishable by inspecting the *pre-activation* alongside the *post-activation* value, not just the gradient. A saturated sigmoid unit has a pre-activation `a` far from 0 in either direction (very positive or very negative) and a post-activation `h=σ(a)` close to 0 or 1 - the unit is "on" or "off" but its derivative `σ(a)(1-σ(a))` is small regardless. A dead ReLU unit has a pre-activation `a ≤ 0` and a post-activation `h=0` exactly - checking the sign of the pre-activation directly distinguishes "ReLU is off" (`a≤0`, derivative exactly 0) from "sigmoid is saturated" (`|a|` large, derivative small but never exactly 0, and the post-activation is near an extreme of `(0,1)` rather than exactly 0).

## Task 5: Extend the Task and Reflect on the Outcome

2→2→3 network (`ThreeClassNet`, same hidden layer as before, three output logits), `CrossEntropyLoss`. Classes: 0=both off (0,0), 1=disagree ((0,1) or (1,0)), 2=both on (1,1).

**Predictions made before running the modified code:**
1. Shape of the final weight matrix: `(3, 2)` - 3 output units, each a linear combination of the 2 hidden activations.
2. Logits per example: 3 (one unnormalized score per class).
3. Why softmax probabilities sum to 1: softmax is defined as `exp(z_i) / Σ_j exp(z_j)` - each output is that class's exponentiated logit divided by the sum of *all* classes' exponentiated logits, so by construction the outputs sum to `Σ_i exp(z_i) / Σ_j exp(z_j) = 1`.
4. Why the logit gradient has the form `p - y`: for softmax + cross-entropy, `L = -log p_c` where `c` is the true class and `y` is its one-hot encoding; differentiating through the softmax gives `∂L/∂z_i = p_i - y_i` for every class `i` simultaneously - a clean, bounded gradient (unlike differentiating log and softmax as separate unfused operations), which is exactly why PyTorch's `CrossEntropyLoss` takes raw logits directly rather than requiring a separate softmax layer.

**Results after running:**
```
Final loss: 0.000957
(0,0) -> probs [0.9989, 0.0011, 0.0000]   pred 0  target 0
(0,1) -> probs [0.0003, 0.9992, 0.0005]   pred 1  target 1
(1,0) -> probs [0.0003, 0.9992, 0.0005]   pred 1  target 1
(1,1) -> probs [0.0000, 0.0011, 0.9989]   pred 2  target 2
All 4 predictions correct: True
```
All 4 predictions match; predictions 1-4 above were all confirmed correct.

Softmax vector for input (0,0): `[0.998925507068634, 0.0010744699975475669, 3.858508179632736e-09]`, sum = **1.0000000000** (verified numerically, not just asserted).

**Optional diagnostic - adding a constant to all logits before softmax:**
```
logits             : [8.544, 1.709, -10.828]
logits + 100        : [108.544, 101.709, 89.172]
softmax(logits)     : [0.998925507068634, 0.0010744699975475669, 3.858508179632736e-09]
softmax(logits+100) : [0.998925507068634, 0.001074476633220911, 3.858508179632736e-09]
max abs difference  : 6.636e-09
```
The probability vector is unchanged up to ordinary floating-point rounding (~1e-9, not a real difference) - confirming softmax is invariant to adding the same constant to every logit, since `exp(z_i+k)/Σ_j exp(z_j+k) = e^k·exp(z_i) / (e^k·Σ_j exp(z_j)) = exp(z_i)/Σ_j exp(z_j)`, the `e^k` factor cancelling top and bottom.

**Why stable implementations subtract the max logit before exponentiating:** computing the *naive* (unshifted-internally) version directly on `logits+100` demonstrates the actual failure this guards against:
```
exp(logits+100)           : [inf, inf, inf]      <- float32 overflows (max representable ~3.4e38 < e^108)
naive exp(logits+100)/sum : [nan, nan, nan]       <- inf/inf, the formula breaks completely
torch.softmax(logits+100) : [0.9989..., 0.00107..., 3.86e-09]   <- correct, unaffected
```
`torch.softmax` internally subtracts `max(logits)` before exponentiating, which (by the identity above, with `k = -max(logits)`) leaves the mathematical result exactly unchanged while making the largest shifted logit exactly 0 (`exp(0)=1`) and every other shifted logit ≤ 0, so no intermediate `exp(...)` value can ever overflow - this is precisely why a logit shift that would otherwise crash a naive implementation passes through an industrial one unaffected.

**Think About It - what stays the same / changes when scaling to a large-vocabulary LM?** What stays mathematically identical: the softmax-plus-cross-entropy machinery itself (`p = softmax(logits)`, `L=-log p_c`, gradient `p-y`) and the numerical-stability argument for subtracting the max logit before exponentiating - both apply unchanged whether there are 3 classes or 50,000 subword tokens. What changes dramatically: the *cost* of computing that softmax (a sum over 50,000 terms instead of 3, often now a meaningful fraction of total compute, motivating tricks like sampled softmax or hierarchical/adaptive softmax in practice), the fact that most of the probability mass vanishingly concentrates on a tiny fraction of the vocabulary for any given context (unlike this toy problem's fairly spread label distribution), and the surrounding architecture producing the logits (a small 2-2 hidden layer here vs. a full transformer stack feeding a final linear "unembedding" layer in a real LM).

## Suggested Result Table (from Part D, repeated for convenience)

| Hidden activation | Final loss | 4/4 correct? | Early ‖∇_{W⁽¹⁾} L‖₂ |
|---|---|---|---|
| Sigmoid | 0.017066 | True | 0.026860 |
| Tanh | 0.347547 | False | 0.051409 |
| ReLU | 0.346703 | False | 0.075095 |

---

## Reflection Questions

1. **What did the XOR experiment demonstrate about the difference between depth and nonlinearity?** Depth (more affine layers) alone contributes nothing representationally without a nonlinearity between the layers - Task 1 showed algebraically that stacked affine maps collapse to one affine map, and Part C's zero-initialization experiment demonstrated the same point empirically from a different angle: even with a nonlinear activation *present*, if the network never breaks the symmetry between hidden units it behaves as though it had a single effective unit, converging to exactly the single-linear-unit optimum (`ln 2`) rather than the true solution. Nonlinearity is necessary but the network also has to actually *use* it differently across units; depth without that differentiation buys nothing.
2. **What evidence showed backpropagation supplied a useful learning signal rather than merely a nonzero gradient?** The gradient norm at step 0 (0.0043) was nonzero, which alone only shows *some* signal exists; what actually demonstrates a *useful* signal is that following it drove the loss from 0.70 down to 0.00009 and flipped all four predictions to correct over the course of training (Task 4 Part A/B) - a nonzero-but-useless gradient (e.g. pointing in a direction orthogonal to anything that reduces classification error) would not have produced that monotonic loss decrease and correct final predictions.
3. **Why did identical/zero weight initialisation prevent the two hidden units from learning distinct features?** Because the hidden units' outputs and the gradients flowing back to them are *functions of the current weights*, and when two units start with identical weights, the entire forward and backward computation graph is symmetric in those two units at every step - there is nothing in the gradient computation that can ever treat them differently, so an update that is identical for both can never make them diverge (Part C, argued in detail above).
4. **How did changing the hidden activation affect the observed gradient? Distinguish the scientific explanation from the engineering observation.** *Scientific explanation:* the chain rule multiplies the local derivative of each activation into the backward pass; ReLU's derivative is exactly 1 when active and 0 when inactive (a hard, binary gate), while sigmoid's derivative is at most 0.25 and shrinks continuously as the unit saturates - so in principle ReLU should preserve gradient magnitude better and sigmoid should attenuate it. *Engineering observation, from this experiment specifically:* the early gradient norms did follow that ranking (ReLU 0.075 > tanh 0.051 > sigmoid 0.027 at step 0), but the activation with the *smallest* early gradient (sigmoid) was also the only one that reached a correct solution from this particular shared initialization - a reminder that "larger gradient early on" is not the same claim as "converges to a better solution," and that the two should not be conflated.
5. **Why must the output layer and loss be selected together according to the task?** The loss function needs the output's range/interpretation to match what it is designed to penalize: binary cross-entropy expects a probability in `(0,1)` to compare against a `{0,1}` label (hence pairing with sigmoid), while categorical cross-entropy expects a probability *distribution* over K mutually exclusive classes summing to 1 (hence pairing with softmax). Mismatching them - e.g., feeding raw unbounded logits into plain BCE, or using a sigmoid output with categorical cross-entropy - would make the loss's mathematical derivation (as a negative log-likelihood under some assumed output distribution) simply wrong, even if the code runs without error.
6. **One example where the LLM improved engineering productivity, and one where human verification was essential.** *Productivity:* translating the already-specified design (2-2-1 architecture, chosen loss, zero-init flag, gradient-recording hooks) into working PyTorch - `nn.Linear`, `BCEWithLogitsLoss`, the `optimizer.zero_grad()/backward()/step()` loop - took one prompt rather than looking up each API individually. *Verification essential:* the LLM-generated training loop looked entirely reasonable and ran without error, yet the very first attempt (plain SGD, seed=0) silently converged to the wrong answer (stuck at `ln 2`, 0/4 correct) - nothing about the code's structure signalled this; only running it and checking the actual predictions against the known-correct targets revealed the failure, exactly the "program runs ≠ model is correct" lesson from the Bayesian-network lab (Lab 6).
7. **Which tests would you keep at scale, and which become too expensive?** Keep: checking final-loss sanity against a trivial baseline (here, `ln 2`; at scale, the loss of a frequency-only/majority-class baseline), checking a held-out set of predictions against known labels, and inspecting gradient norms over training for signs of vanishing/exploding behaviour (cheap summary statistics, not expensive to compute even for a huge model). Become too expensive: exhaustive finite-difference gradient checking (comparing every analytic gradient entry against a numerically perturbed version) - affordable here with 2x2+2x1 parameters, but with millions/billions of parameters each finite-difference check requires a full extra forward pass per parameter, making exhaustive checking computationally infeasible; at scale this is instead done only spot-wise, on a small random subset of parameters or on a drastically smaller proxy model.

## Responsible Use of LLMs - reflection

This lab's own warning proved concretely true during Task 4 Part A: the LLM-generated training code was syntactically correct, ran without any error, and *looked* like a complete, reasonable implementation of the specified design - and it still silently produced the wrong result (stuck at `ln 2`, 0/4 correct) for the first initialization tried. Nothing in the code itself signalled this; it was only caught by actually running it and comparing the reported predictions against the four known-correct labels, exactly the kind of independent verification this course's "AI Science vs. AI Engineering" framing asks for - the LLM accelerated getting from a specification to running code, but did not and could not substitute for checking whether that code's *behaviour* matched the intended model.
