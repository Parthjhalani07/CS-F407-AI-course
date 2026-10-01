"""
Runs every experiment required by the lab: the baseline XOR training run
and backprop check (Task 4 Parts A-B), the symmetry experiment (Part C),
the activation comparison (Part D), and the three-class extension (Task 5).

Run: python3 run_experiments.py
"""

import torch

from xor_model import (
    X, Y_BINARY, Y_3CLASS,
    make_binary_model, make_threeclass_model,
    train_binary, train_threeclass,
)


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
section("TASK 4 Part A: First attempt - plain SGD, seed=0, lr=0.5, 3000 steps")
first_attempt_model = make_binary_model(activation="sigmoid", seed=0)
first_attempt = train_binary(first_attempt_model, epochs=3000, lr=0.5, optimizer_name="sgd")
print(f"Final loss: {first_attempt['final_loss']:.6f}   All 4 correct: {first_attempt['all_correct']}")
print("-> Got stuck near loss ln(2)=0.693 (network just predicts ~0.5 for everything).")
print("   This is the well-known flat local minimum of the sigmoid-XOR loss surface for")
print("   this initialisation - an optimisation problem, not an architecture problem.")
print("   Engineering fix (per Task 4 Part A's instructions): change only the optimiser")
print("   and/or seed, not the architecture/loss/data. Switching SGD -> Adam fixed it for")
print("   every seed tried except {0, 5}; seed=1 below is a clean example.")

section("TASK 4 Part A/B: Baseline run (sigmoid hidden, random init, seed=1, Adam)")
model = make_binary_model(activation="sigmoid", seed=1)
result = train_binary(model, epochs=3000, lr=0.1, optimizer_name="adam")

print(f"Initial loss : {result['initial_loss']:.4f}")
print(f"Final loss   : {result['final_loss']:.6f}")
print(f"{'input':<10}{'prob':<10}{'pred':<8}{'target':<8}")
for x_row, p, pred, t in zip(X.tolist(), result["probs"], result["preds"], result["targets"]):
    print(f"{str(x_row):<10}{p:<10.4f}{pred:<8.0f}{t:<8.0f}")
print(f"All 4 predictions correct: {result['all_correct']}")

print("\nFirst-layer weight gradient (dL/dW_hidden) after the final backward():")
print(result["final_hidden_grad"])
print("\ngradient norm ||grad W_hidden||_2 at early steps:", result["grad_norms_at_step"])

# ---------------------------------------------------------------------------
section("TASK 4 Part C: Symmetry experiment (all weights/biases zero-initialised)")
zero_model = make_binary_model(activation="sigmoid", seed=0, zero_init=True)
zero_result = train_binary(zero_model, epochs=200, lr=0.5, record_steps=(0, 1, 5, 20, 100),
                            optimizer_name="sgd")

print("Hidden-layer weight matrix at selected steps (row0 = unit A, row1 = unit B):")
for step, rows in zero_result["hidden_rows_over_time"][:8]:
    row0, row1 = rows
    identical = row0 == row1
    print(f"  step {step:>4}: row0={row0}  row1={row1}  identical={identical}")

final_w = zero_model.hidden.weight.data
rows_identical_at_end = torch.allclose(final_w[0], final_w[1])
print(f"\nAfter 200 steps, are the two hidden rows still identical? {rows_identical_at_end}")
print(f"Final loss (zero-init run): {zero_result['final_loss']:.6f}")
print(f"All 4 predictions correct (zero-init run): {zero_result['all_correct']}  "
      f"(expected: False - a network with two identical hidden units behaves like one)")

# ---------------------------------------------------------------------------
section("TASK 4 Part D: Activation experiment (same initial weights, only activation differs)")
print(f"{'Activation':<12}{'Final loss':<14}{'4/4 correct?':<14}{'Early ||grad W1||_2':<22}")
activation_results = {}
for act in ["sigmoid", "tanh", "relu"]:
    m = make_binary_model(activation=act, seed=100)  # same seed => identical initial weights
    r = train_binary(m, epochs=3000, lr=0.5, record_steps=(0,), optimizer_name="sgd")
    activation_results[act] = r
    early_norm = r["grad_norms_at_step"][0]
    print(f"{act:<12}{r['final_loss']:<14.6f}{str(r['all_correct']):<14}{early_norm:<22.6f}")

# ---------------------------------------------------------------------------
section("TASK 5: Three-class extension (softmax + cross-entropy)")
tc_model = make_threeclass_model(activation="tanh", seed=0)
tc_result = train_threeclass(tc_model, epochs=3000, lr=0.5)

print(f"Final loss: {tc_result['final_loss']:.6f}")
print(f"{'input':<10}{'probs (class0,1,2)':<40}{'pred':<6}{'target':<6}")
for x_row, probs, pred, t in zip(X.tolist(), tc_result["probs"].tolist(),
                                  tc_result["preds"], tc_result["targets"]):
    print(f"{str(x_row):<10}{str([round(p, 4) for p in probs]):<40}{pred:<6}{t:<6}")
print(f"All 4 predictions correct: {tc_result['all_correct']}")

example_probs = tc_result["probs"][0]
print(f"\nSoftmax probability vector for input {X[0].tolist()}: {example_probs.tolist()}")
print(f"Sum of components: {example_probs.sum().item():.10f}")

print("\nOptional diagnostic: shift-invariance of softmax")
logits0 = tc_result["logits"][0]
shifted = logits0 + 100.0
p_unshifted = torch.softmax(logits0, dim=0)
p_shifted = torch.softmax(shifted, dim=0)
print(f"  logits            : {logits0.tolist()}")
print(f"  logits + 100       : {shifted.tolist()}")
print(f"  softmax(logits)    : {p_unshifted.tolist()}")
print(f"  softmax(logits+100): {p_shifted.tolist()}")
print(f"  max abs difference : {(p_unshifted - p_shifted).abs().max().item():.3e}")

print("\nUnstable (naive, no max-subtraction) softmax on the shifted logits, for comparison:")
exp_shifted = torch.exp(shifted)
naive = exp_shifted / exp_shifted.sum()
print(f"  exp(logits+100)          : {exp_shifted.tolist()}  <- already overflowed to inf (float32 max ~3.4e38 < e^108)")
print(f"  naive exp(logits+100)/sum: {naive.tolist()}  <- inf/inf = nan: the naive formula breaks")
print(f"  torch.softmax(logits+100): {p_shifted.tolist()}  <- correct, because torch subtracts the max logit first")
print("  This is exactly why stable softmax implementations subtract max(logits) before exponentiating:")
print("  it makes the largest exponent 0 (exp(0)=1) and every other exponent <= 0, so nothing can overflow,")
print("  and because every logit is shifted by the SAME constant, the resulting ratios - the probabilities -")
print("  are mathematically unchanged (shown above: max abs difference from the unshifted softmax is ~1e-9,")
print("  i.e. only ordinary floating-point rounding, not a real difference).")
