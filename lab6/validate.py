"""
Reusable validation tests for LLM-generated Bayesian-network code (Part:
"Turn validation into reusable tests"). These check structure, parameter
validity, and semantics - not just "did the code run without raising".
"""

from trusted_model import enumerate_posterior
from pgmpy.inference import VariableElimination

EXPECTED_EDGES = {("C", "R"), ("C", "S"), ("R", "W"), ("S", "W")}
EXPECTED_NODES = {"C", "R", "S", "W"}


def check_structure(model):
    nodes_ok = set(model.nodes()) == EXPECTED_NODES
    edges_ok = set(model.edges()) == EXPECTED_EDGES
    acyclic_ok = model.check_model()  # pgmpy raises/returns False on cycles too
    return {"nodes_ok": nodes_ok, "edges_ok": edges_ok, "acyclic_ok": acyclic_ok}


def check_cpd_shapes(model):
    """Each CPD's variable/evidence cardinalities match the 2x2x2x2 binary network."""
    results = {}
    for var in EXPECTED_NODES:
        cpd = model.get_cpds(var)
        results[var] = {
            "variable_card": cpd.variable_card == 2,
            "evidence": list(cpd.get_evidence()),
        }
    return results

def check_normalization(model, tol=1e-8):
    """Every CPD's columns must each sum to 1 - pgmpy's check_model() already
    enforces this, but we also check it directly so the test is readable on
    its own and does not depend on that library call succeeding silently."""
    results = {}
    for var in EXPECTED_NODES:
        cpd = model.get_cpds(var)
        vals = cpd.get_values()
        col_sums = vals.sum(axis=0)
        results[var] = bool((abs(col_sums - 1.0) < tol).all())
    return results


def check_posterior_matches_oracle(model, query_var, evidence, tol=1e-6):
    """Compare VariableElimination's answer against the brute-force oracle."""
    infer = VariableElimination(model)
    pgmpy_result = infer.query(variables=[query_var], evidence=evidence, show_progress=False)
    pgmpy_p1 = float(pgmpy_result.values[1])
    oracle_p1 = enumerate_posterior(query_var, evidence)
    return {
        "pgmpy": pgmpy_p1,
        "oracle": oracle_p1,
        "match": abs(pgmpy_p1 - oracle_p1) < tol,
    }


def run_all_checks(model, label):
    print(f"\n--- Validating: {label} ---")
    structure = check_structure(model)
    print("Structure:", structure)
    try:
        cpds_valid = model.check_model()
    except Exception as e:
        cpds_valid = f"REJECTED: {e}"
    print("model.check_model():", cpds_valid)
    if cpds_valid is True:
        print("Normalization per-CPD:", check_normalization(model))
        posterior = check_posterior_matches_oracle(model, "R", {"W": 1})
        print("P(R=1|W=1) vs oracle:", posterior)
        return structure, cpds_valid, posterior
    return structure, cpds_valid, None


if __name__ == "__main__":
    from trusted_model import build_trusted_model
    run_all_checks(build_trusted_model(), "trusted model")
