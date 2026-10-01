"""
The trusted, hand-written Sprinkler/Rain/WetGrass Bayesian network.

Graph:   C -> R,  C -> S,  R -> W,  S -> W
Factorisation: P(C,R,S,W) = P(C) P(R|C) P(S|C) P(W|R,S)
States: 0 = False, 1 = True

This module is deliberately written and verified BEFORE any LLM-generated
code is produced (see bn_pipeline.py), so it serves as the test oracle that
LLM-generated implementations are checked against.
"""

from itertools import product

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


def build_trusted_model():
    model = DiscreteBayesianNetwork([("C", "R"), ("C", "S"), ("R", "W"), ("S", "W")])

    # P(C=1) = 0.5
    cpd_c = TabularCPD(variable="C", variable_card=2, values=[[0.5], [0.5]])

    # P(R=1|C=0)=0.2, P(R=1|C=1)=0.8
    cpd_r = TabularCPD(
        variable="R", variable_card=2,
        values=[[0.8, 0.2], [0.2, 0.8]],
        evidence=["C"], evidence_card=[2],
    )

    # P(S=1|C=0)=0.5, P(S=1|C=1)=0.1
    cpd_s = TabularCPD(
        variable="S", variable_card=2,
        values=[[0.5, 0.9], [0.5, 0.1]],
        evidence=["C"], evidence_card=[2],
    )

    # P(W=1|R,S): columns ordered (R=0,S=0),(R=0,S=1),(R=1,S=0),(R=1,S=1)
    cpd_w = TabularCPD(
        variable="W", variable_card=2,
        values=[
            [0.99, 0.10, 0.10, 0.01],  # P(W=0 | R,S)
            [0.01, 0.90, 0.90, 0.99],  # P(W=1 | R,S)
        ],
        evidence=["R", "S"], evidence_card=[2, 2],
    )

    model.add_cpds(cpd_c, cpd_r, cpd_s, cpd_w)
    assert model.check_model()
    return model


def enumerate_posterior(query_var, evidence, true_params=None):
    """
    Independently compute P(query_var=1 | evidence) by brute-force
    enumeration over all 2**4 = 16 assignments of (C,R,S,W), using the
    factorisation directly - no pgmpy involved. Serves as a trusted,
    from-scratch oracle independent of the library.
    """
    p = true_params or TRUE_PARAMS
    joint = 0.0
    numer = 0.0
    for c, r, s, w in product([0, 1], repeat=4):
        assignment = {"C": c, "R": r, "S": s, "W": w}
        if any(assignment[k] != v for k, v in evidence.items()):
            continue
        pc = p["C"][c]
        pr = p["R"][c][r]
        ps = p["S"][c][s]
        pw = p["W"][(r, s)][w]
        prob = pc * pr * ps * pw
        joint += prob
        if assignment[query_var] == 1:
            numer += prob
    return numer / joint


TRUE_PARAMS = {
    "C": {0: 0.5, 1: 0.5},
    "R": {0: {0: 0.8, 1: 0.2}, 1: {0: 0.2, 1: 0.8}},  # R[c][r]
    "S": {0: {0: 0.5, 1: 0.5}, 1: {0: 0.9, 1: 0.1}},  # S[c][s]
    "W": {
        (0, 0): {0: 0.99, 1: 0.01},
        (0, 1): {0: 0.10, 1: 0.90},
        (1, 0): {0: 0.10, 1: 0.90},
        (1, 1): {0: 0.01, 1: 0.99},
    },  # W[(r,s)][w]
}


if __name__ == "__main__":
    model = build_trusted_model()
    infer = VariableElimination(model)
    result = infer.query(variables=["R"], evidence={"W": 1}, show_progress=False)
    print(result)
    print("Independent enumeration P(R=1|W=1) =",
          enumerate_posterior("R", {"W": 1}))
