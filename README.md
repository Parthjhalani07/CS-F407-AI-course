# CSF407 - Artificial Intelligence Labs

**Author:** Parth Jhalani

Each folder below is self-contained: code, the prompt(s) used with an LLM, captured output from
actually running the code, and a `README.md` with the full write-up (specification, design,
results, and reflection questions) for that lab. Folder names match the lab topics as used in the
course (weeks 1-8 - this course has no material beyond Week 8).

| Folder | Course week | Topic |
|---|---|---|
| [`neur_models_lab/`](neur_models_lab/README.md) | Week 1 - Neural Models: Hands-on | XOR with PyTorch: depth, activations, output layers |
| [`agents_lab/`](agents_lab/README.md) | Week 2 - Agents: Hands-on | A goal-based warehouse-navigation agent |
| [`search_lab/`](search_lab/README.md) | Week 3 - Search: Hands-on | Search and A* (warehouse robot navigation) |
| [`logic_lab/`](logic_lab/REPORT.md) | Week 4 - Logic: Hands-on | Logical reasoning for planning (BFS planning agent + Prolog verifier) |
| [`bn_lab/`](bn_lab/README.md) | Week 7 - AR Models: Hands-on / Week 8 - BN: Lab | Bayesian networks and autoregressive language models (n-gram LM) |
| [`llm_bn_lab/`](llm_bn_lab/README.md) | Week 6 - BN: Hands-on (1) | Bayesian networks with `pgmpy`, LLM-assisted, validated against an independent oracle |
| [`transformers_lab/`](transformers_lab/README.md) | (not on this course's own week 1-8 schedule; included to match [souhhmm/csf407_labs](https://github.com/souhhmm/csf407_labs)) | Transformers: encoder/decoder/encoder-only/decoder-only demos + a local LLM |

## Setup

`logic_lab/`, `search_lab/`, and `agents_lab/` need only the Python standard library.
`bn_lab/`, `llm_bn_lab/`, `neur_models_lab/`, and `transformers_lab/` use a shared virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pgmpy numpy pandas torch transformers sentencepiece sacremoses \
    --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

Then, from inside any lab folder: `python3 run_experiments.py` (or the lab's equivalently named
entry point - see that lab's own `README.md`).
