"""
Runs every experiment required by the lab: CPT construction (Part IV),
normalization testing (Part VII), next-word prediction (Part VIII),
text generation (Part IX), greedy vs sampling (Part X), and the
first-order vs second-order comparison (Part XIII).

Run: python3 run_experiments.py
All randomness uses a fixed seed so the output is reproducible.
"""

import random

from bayesian_lm import NgramModel, DATASET, START, END

SEED = 42


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def part_iv_cpt(order1):
    section("PART IV: Conditional Probability Table (order-1)")
    for word in ["the", "cat", "dog", "sat", "ran"]:
        dist = order1.distribution([word])
        print(f"P(next | {word!r}) = {dist}")
    print("\nZero-probability transitions for these words (never observed in data):")
    for word in ["the", "cat", "dog", "sat", "ran"]:
        zeros = order1.zero_probability_continuations([word])
        print(f"  {word!r} -> never followed by: {zeros}")


def part_vii_normalization(model, name):
    section(f"PART VII: Normalization test ({name})")
    all_ok = True
    for context, total in model.check_normalization():
        ok = abs(total - 1.0) < 1e-9
        all_ok = all_ok and ok
        print(f"  context={context!s:<30} sum(P)={total:.6f}  {'OK' if ok else 'FAIL'}")
    print(f"All contexts normalize to 1.0: {all_ok}")
    return all_ok


def part_viii_predict(order1):
    section("PART VIII: Next-word prediction (order-1)")
    for word in ["the", "cat", "dog", "sat", "ran"]:
        dist = order1.distribution([word])
        best = order1.most_probable_next([word])
        print(f"P(X_t+1 | X_t={word!r}) = {dist}")
        print(f"  argmax_w P(w | {word!r}) = {best!r}\n")


def part_ix_generate(model, name, n=20, seed=SEED):
    section(f"PART IX: Generate {n} sentences by sampling ({name})")
    rng = random.Random(seed)
    sentences = [model.generate(mode="sample", rng=rng) for _ in range(n)]
    for i, s in enumerate(sentences, 1):
        print(f"  {i:2d}. {s}")
    out_path = f"generated_{name}.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(sentences) + "\n")
    print(f"Saved to {out_path}")
    return sentences


def part_x_greedy_vs_sampling(model, name, seed=SEED):
    section(f"PART X: Deterministic (greedy) vs probabilistic (sampling) - {name}")
    rng = random.Random(seed)
    greedy = [model.generate(mode="greedy") for _ in range(5)]
    sampled = [model.generate(mode="sample", rng=rng) for _ in range(5)]
    print("Mode A - greedy (argmax every step):")
    for s in greedy:
        print("   ", s)
    print("Mode B - sampling:")
    for s in sampled:
        print("   ", s)
    print(f"Distinct greedy sentences : {len(set(greedy))} / {len(greedy)}")
    print(f"Distinct sampled sentences: {len(set(sampled))} / {len(sampled)}")
    return greedy, sampled


def part_xiii_compare(order1, order2):
    section("PART XIII: Comparing the first-order and second-order models")
    print(f"Distinct contexts   : order1={len(order1.probs)}   order2={len(order2.probs)}")
    print(f"Distinct parameters : order1={order1.num_parameters()}   "
          f"order2={order2.num_parameters()}")

    single_choice_o1 = sum(1 for d in order1.probs.values() if len(d) == 1)
    single_choice_o2 = sum(1 for d in order2.probs.values() if len(d) == 1)
    print(f"Contexts with only ONE possible continuation (zero uncertainty): "
          f"order1={single_choice_o1}/{len(order1.probs)}   "
          f"order2={single_choice_o2}/{len(order2.probs)}")

    rng1 = random.Random(SEED)
    rng2 = random.Random(SEED)
    gen1 = [order1.generate(mode="sample", rng=rng1) for _ in range(30)]
    gen2 = [order2.generate(mode="sample", rng=rng2) for _ in range(30)]
    print(f"\nDiversity over 30 sampled sentences: "
          f"order1 unique={len(set(gen1))}/30   order2 unique={len(set(gen2))}/30")

    print("\nSample order-1 generations:")
    for s in gen1[:6]:
        print("   ", s)
    print("Sample order-2 generations:")
    for s in gen2[:6]:
        print("   ", s)


if __name__ == "__main__":
    order1 = NgramModel(DATASET, order=1)
    order2 = NgramModel(DATASET, order=2)

    part_iv_cpt(order1)
    part_vii_normalization(order1, "order-1")
    part_vii_normalization(order2, "order-2")
    part_viii_predict(order1)
    part_ix_generate(order1, "order1", n=20)
    part_ix_generate(order2, "order2", n=20)
    part_x_greedy_vs_sampling(order1, "order-1")
    part_x_greedy_vs_sampling(order2, "order-2")
    part_xiii_compare(order1, order2)
