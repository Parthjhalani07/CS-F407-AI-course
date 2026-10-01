"""
Full engineering pipeline for the Sprinkler/Rain/WetGrass Bayesian network:

  specify -> generate -> inspect -> approve -> execute -> validate -> test

Run: python3 bn_pipeline.py
All synthetic-data experiments use fixed seeds for reproducibility.
"""

import warnings
import numpy as np
import pandas as pd
from pgmpy.sampling import BayesianModelSampling
from pgmpy.inference import VariableElimination

from trusted_model import build_trusted_model, enumerate_posterior, TRUE_PARAMS
from llm_generated import (
    build_network_from_prompt1, fit_mle, fit_bayesian,
    build_broken_cpt_model, build_reversed_evidence_order_model,
)
from validate import run_all_checks, check_posterior_matches_oracle

warnings.filterwarnings("ignore", category=FutureWarning)

EDGES = [("C", "R"), ("C", "S"), ("R", "W"), ("S", "W")]


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
section("STEP 1-3: Trusted model, exact inference, independent oracle")
trusted = build_trusted_model()
infer = VariableElimination(trusted)
res = infer.query(variables=["R"], evidence={"W": 1}, show_progress=False)
print(res)
oracle = enumerate_posterior("R", {"W": 1})
print(f"Independent brute-force enumeration: P(R=1|W=1) = {oracle:.4f}")
print(f"pgmpy VariableElimination:           P(R=1|W=1) = {float(res.values[1]):.4f}")
assert abs(float(res.values[1]) - oracle) < 1e-9

# ---------------------------------------------------------------------------
section("STEP 4: Validate the trusted model with reusable tests")
run_all_checks(trusted, "trusted (hand-written) model")

# ---------------------------------------------------------------------------
section("STEP 5: 'LLM-generated' network (Prompt 1) vs trusted model")
llm_model = build_network_from_prompt1()
run_all_checks(llm_model, "LLM-generated model (Prompt 1)")
for var in "CRSW":
    same = np.allclose(trusted.get_cpds(var).get_values(), llm_model.get_cpds(var).get_values())
    print(f"  CPD({var}) identical to trusted model: {same}")

# ---------------------------------------------------------------------------
section("STEP 6: Generate synthetic data from the trusted model")
sampler = BayesianModelSampling(trusted)
rng_seed = 0
big_data = sampler.forward_sample(size=5000, seed=rng_seed, show_progress=False)
print(big_data.head())
print(f"N = {len(big_data)} rows generated.")

# ---------------------------------------------------------------------------
section("STEP 7: MLE parameter estimation (Prompt 2) vs the true CPTs")
mle_model = fit_mle(big_data, EDGES)
cpd_r_hat = mle_model.get_cpds("R")
print(cpd_r_hat)
# Hand-computed check of one entry: P(R=1|C=1) from raw counts.
n_c1 = (big_data["C"] == 1).sum()
n_r1_c1 = ((big_data["C"] == 1) & (big_data["R"] == 1)).sum()
hand_estimate = n_r1_c1 / n_c1
lib_estimate = cpd_r_hat.get_values()[1, 1]  # row R=1, column C=1
print(f"Hand count:  P_hat(R=1|C=1) = {n_r1_c1}/{n_c1} = {hand_estimate:.4f}")
print(f"pgmpy MLE :  P_hat(R=1|C=1) = {lib_estimate:.4f}")
print(f"True value:  P(R=1|C=1)     = {TRUE_PARAMS['R'][1][1]:.4f}")
assert abs(hand_estimate - lib_estimate) < 1e-9

# ---------------------------------------------------------------------------
section("STEP 8: How does estimation stability change with sample size?")
true_r1_c1 = TRUE_PARAMS["R"][1][1]
print(f"{'N':>6}  {'P_hat(R=1|C=1)':>16}  {'|error|':>10}")
for n in [20, 50, 200, 1000, 5000, 20000]:
    data_n = sampler.forward_sample(size=n, seed=1, show_progress=False)
    m_n = fit_mle(data_n, EDGES)
    est = m_n.get_cpds("R").get_values()[1, 1]
    print(f"{n:>6}  {est:>16.4f}  {abs(est - true_r1_c1):>10.4f}")

# ---------------------------------------------------------------------------
section("STEP 9: Sparse data - MLE vs Bayesian (BDeu) estimation")
sparse_data = sampler.forward_sample(size=8, seed=2, show_progress=False)
print(sparse_data)
print("\nObserved (C,S) combinations in this sample:",
      sorted(set(zip(sparse_data["C"], sparse_data["S"]))), "out of 4 possible")
mle_sparse = fit_mle(sparse_data, EDGES)
bayes_sparse = fit_bayesian(sparse_data, EDGES, prior_type="BDeu", equivalent_sample_size=10)
print("\nMLE CPD(S):")
print(mle_sparse.get_cpds("S"))
print("\nBayesian/BDeu CPD(S):")
print(bayes_sparse.get_cpds("S"))

# ---------------------------------------------------------------------------
section("STEP 10: A deliberately broken Bayesian network (bad normalization)")
try:
    broken = build_broken_cpt_model()
    broken.check_model()
    print("UNEXPECTED: broken model was accepted")
except Exception as e:
    print(f"REJECTED as expected: {type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
section("STEP 11: A subtler failure - valid numbers, wrong evidence order")
reversed_model = build_reversed_evidence_order_model()
structure_ok = reversed_model.check_model()
print(f"model.check_model() result: {structure_ok}  <- passes! each column still sums to 1")
semantic = check_posterior_matches_oracle(reversed_model, "R", {"W": 1})
print(f"Semantic test (posterior vs independent oracle): {semantic}")
print("-> check_model() alone does NOT catch this bug; the semantic/oracle test does.")

# ---------------------------------------------------------------------------
section("STEP 12: HW exercise - new queries on the unmodified network")
hw_queries = [
    ("S", {"W": 1}, "P(S=1|W=1): naive prediction - wet grass is direct evidence FOR sprinkler use, so this should increase vs prior P(S=1)=0.5"),
    ("C", {"W": 1}, "P(C=1|W=1): naive prediction - wet grass is explained by rain or sprinkler, both of which are more likely when cloudy, so this should increase vs prior P(C=1)=0.5"),
    ("R", {"W": 1, "S": 0}, "P(R=1|W=1,S=0): naive prediction - with the sprinkler ruled out, wet grass becomes strong evidence for rain, so this should be noticeably higher than the unconditioned P(R=1|W=1)=0.7048"),
]
prior = {"C": 0.5, "S": 0.5, "R": 0.5}
for var, evidence, prediction in hw_queries:
    q = infer.query(variables=[var], evidence=evidence, show_progress=False)
    p1 = float(q.values[1])
    print(f"\n{prediction}")
    print(f"  Computed: P({var}=1 | {evidence}) = {p1:.4f}  (prior P({var}=1)={prior[var]})")

# ---------------------------------------------------------------------------
section("STEP 13: Classroom exercise - variability of the MLE estimate across seeds")
print(f"{'seed':>6}  {'N':>6}  {'P_hat(R=1|C=1)':>16}")
estimates = []
for seed in range(10):
    data_s = sampler.forward_sample(size=200, seed=seed, show_progress=False)
    m_s = fit_mle(data_s, EDGES)
    est = m_s.get_cpds("R").get_values()[1, 1]
    estimates.append(est)
    print(f"{seed:>6}  {200:>6}  {est:>16.4f}")
print(f"\nMean = {np.mean(estimates):.4f}, Std = {np.std(estimates):.4f}, True = {true_r1_c1}")
