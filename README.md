# CSF407 - Artificial Intelligence Labs

**Author:** Parth Jhalani

Each `labN/` folder is self-contained: code, the prompt(s) used with an LLM, captured output
from actually running the code, and a `README.md` with the full write-up (specification, design,
results, and reflection questions) for that lab.

| Folder | Topic |
|---|---|
| [`lab2/`](lab2/README.md) | Logical reasoning for planning (BFS planning agent + Prolog verifier) |
| [`lab3/`](lab3/REPORT.md) | Logical reasoning for planning, continued |
| [`lab4/`](lab4/README.md) | Search and A* (warehouse robot navigation) |
| [`lab5/`](lab5/README.md) | Bayesian networks and autoregressive language models (n-gram LM) |
| [`lab6/`](lab6/README.md) | Bayesian networks with `pgmpy`, LLM-assisted, validated against an independent oracle |
| [`lab7/`](lab7/README.md) | Neural models: depth, activations, and output layers (XOR with PyTorch) |
| [`lab8/`](lab8/README.md) | Agents: a goal-based warehouse-navigation agent |
| [`lab9/`](lab9/README.md) | Transformers: encoder/decoder/encoder-only/decoder-only demos + a local LLM |

## Setup

Labs 2-4 need only the Python standard library. Labs 5-9 use a shared virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pgmpy numpy pandas torch transformers sentencepiece sacremoses \
    --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

Then, from inside any `labN/` folder: `python3 run_experiments.py` (or the lab's equivalently
named entry point - see that lab's own `README.md`).
