# Agents: Constructing a Goal-Based Agent using an LLM

**Course:** Undergraduate Artificial Intelligence
**Topic:** Warehouse navigation as a goal-based agent, built with LLM assistance
**Author:** Parth Jhalani

> `warehouse_agent.py` and `test_agent.py` were written with the assistance of **Claude
> (Anthropic)** from the two prompts in [`prompts.txt`](prompts.txt) (an initial implementation,
> then an iterative-improvement prompt adding independent path validation), based on the design
> specified independently in Task 2 below, and then run and verified. All output in this report
> is from [`full_output.txt`](full_output.txt) (`python3 test_agent.py`).

## Files

| File | Description |
|---|---|
| `warehouse_agent.py` | `Environment` (grid model) + `GoalBasedAgent` (state/goal/BFS planner) |
| `test_agent.py` | Independent path validation + trivial/no-solution test cases |
| `prompts.txt` | The two prompts used with the LLM |
| `full_output.txt` | Raw captured output of `python3 test_agent.py` |

## How to Run

```bash
python3 warehouse_agent.py   # runs the agent once on the lab's warehouse map
python3 test_agent.py        # full test suite with independent path validation
```

---

## The Warehouse Navigation Problem

```
#####################
#S....#............G#
#.##....##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################
```
S = start (loading bay), G = goal (dispatch area), `#` = obstacle (shelving), `.` = free space. Moves: Up, Down, Left, Right, each changing position by one grid square.

## Task 1: Understanding the Problem

1. **What is the environment?** The warehouse floor: a fixed, fully-observable, static, discrete 2D grid of free cells and shelving obstacles. It is deterministic (every move has exactly one outcome) and the agent operates alone in it (no other moving vehicles are modelled in this lab).
2. **What is the goal of the agent?** Reach the dispatch-area cell `G`, starting from the loading-bay cell `S`, via a sequence of moves that never crosses an obstacle - a collision-free path.
3. **What actions are available?** `Up`, `Down`, `Left`, `Right` - each attempts to move the vehicle one grid cell in that direction.
4. **What information must the agent maintain to choose its next action?** Its current position (state), the goal position, and a model of the environment (which cells are free) - and, because choosing a *good* next action requires more than reacting to just the current cell, it also needs a plan: the result of having searched ahead for a sequence of actions that reaches the goal, not just a rule for the immediate percept.
5. **Why is this a goal-based agent rather than a simple reflex agent?** A simple reflex agent maps the current percept directly to an action via fixed condition-action rules (e.g. "if the cell ahead is free, move forward; else turn") - it has no explicit representation of *where it is trying to get to*, so it cannot evaluate whether a candidate action actually makes progress toward a destination, and can easily loop forever or wander into a dead end on a map like this one (note the number of dead-end shelving pockets in the map above). This agent instead explicitly represents a goal (`self.goal`) and a separate decision-making/planning component (`formulate_plan`, which searches the whole state space before committing to any action) - it decides what to do by considering the *consequences* of action sequences with respect to the goal, which is the defining property of a goal-based agent.

**Think About It - if the warehouse became twice as large, would the same search strategy still work? What additional difficulties might arise?** BFS would still work correctly (it does not depend on map size for correctness, only for the *number of states*), but its cost would grow: with twice the width and twice the height, the number of free cells - and so the number of states BFS might expand - grows roughly 4x, since it scales with grid area, not a single linear dimension. The frontier (the queue of states not yet expanded) and the visited set would also both grow correspondingly, costing more memory. At some size, an *uninformed* strategy like BFS would become noticeably slower than an informed one (A* with a Manhattan-distance heuristic, as used in `search_lab/`) that can direct its search toward the goal rather than expanding outward in every direction uniformly - exactly the BFS-vs-A* distinction investigated in the search lab.

## Task 2: Designing the Agent

| Component | This design |
|---|---|
| Environment | `Environment` class: parses the ASCII map into a grid, knows which cells are free (`is_free`), and generates valid successor moves from any cell (`successors`) |
| Current state | `GoalBasedAgent.state`: the vehicle's `(row, col)` position, initialised from the environment's `S` |
| Goal | `GoalBasedAgent.goal`: the `(row, col)` position of `G` |
| Available actions | `Up`, `Down`, `Left`, `Right` (the `MOVES` dict) |
| Decision-making component | `GoalBasedAgent.formulate_plan()`: breadth-first search over `Environment.successors`, returning a full action sequence (as a path of states) from the current state to the goal |

**Block diagram (described in words, since this is a plain-text report):**

```
   Environment (grid, obstacles)
          |
          |  perceive (locate S, G; query free cells)
          v
   GoalBasedAgent.state  <----  current position
          |
          |  together with: GoalBasedAgent.goal
          v
   Decision-making component (formulate_plan: BFS over Environment.successors)
          |
          |  produces: a sequence of actions / path of states
          v
   act() --> print the path (or report failure) to the dispatch area
```

This matches the goal-based architecture from the lecture: percepts come from the environment, the agent maintains an internal state and an explicit goal, and a dedicated decision-making component searches over possible action sequences and their *predicted effect on the environment* (via `Environment.successors`, the transition model) before any action is reported/taken - rather than reacting to the immediate percept alone.

## Task 3: Prompt Engineering

Prompt used: see `prompts.txt`, Prompt 1 (closely follows the lab's own suggested prompt, with the explicit addition of structuring the code around the goal-based components named above).

1. **Did the LLM generate a working program on the first attempt?** Yes - `warehouse_agent.py` ran immediately and printed a valid-looking path from `S` to `G` (captured in `full_output.txt`, first section). No syntax or runtime errors needed fixing.
2. **If not, how can you improve your prompt?** (N/A for correctness here, but see the second, deliberate improvement below.) The first prompt asked for a path to be *printed*, which is observable, but did not ask for the path to be *independently verifiable* - a subtly incorrect planner could still print something path-shaped. Prompt 2 specifically closes that gap by requesting an independent check, re-derived from the environment rather than trusting the planner's own bookkeeping (`validate_path()` in `test_agent.py`) - this is the genuine "iterative prompting" improvement for this lab, not a bug fix.
3. **What search algorithm did the LLM choose?** Breadth-first search (`formulate_plan`, using a FIFO `deque` frontier and a `visited` set).
4. **Why do you think the LLM selected this algorithm?** Every move in this problem has equal cost (one grid square), so BFS is the simplest algorithm that is *guaranteed* to find a shortest collision-free path - the first time BFS reaches any cell is provably via a shortest route to it, with no need for a heuristic (unlike A*) or for backtracking/path-cost bookkeeping (unlike Dijkstra's algorithm, which BFS reduces to exactly when all edge costs are equal). Given the lab's own framing ("the emphasis of this laboratory is not Python programming" and the goal is simply "a collision-free path", not explicitly the *shortest* one), BFS is also the lowest-complexity choice that still happens to over-deliver a shortest path for free, which is a reasonable, defensible default for an LLM asked for "a collision-free path" without further qualification.

## Results

```
Path found from (1, 1) to (1, 19):
(1, 1) -> (1, 2) -> (1, 3) -> (1, 4) -> (2, 4) -> (2, 5) -> (2, 6) -> (2, 7) -> (1, 7) ->
(1, 8) -> (1, 9) -> (1, 10) -> (1, 11) -> (1, 12) -> (1, 13) -> (1, 14) -> (1, 15) ->
(1, 16) -> (1, 17) -> (1, 18) -> (1, 19)
Path length: 20 moves
States expanded: 59
```

## Testing and Validation (Task 3's "test and validate", extended per Prompt 2)

Three cases, each independently checked (full output in `full_output.txt`):

1. **Original warehouse map:** path found (above), and independently validated - every step is onto a free cell, every consecutive pair differs by exactly one valid `Up`/`Down`/`Left`/`Right` move, the path starts at `S` and ends at `G`. This re-derives validity from the `Environment` directly (`validate_path()` never consults `formulate_plan`'s internal `came_from`/`visited` state) - it is a genuine check of the output, not a restatement of the algorithm's own assumptions.
2. **Trivial case (`G` adjacent to `S`):** one-step path found and validated.
3. **No-path case (`G` walled off):** the agent correctly reports `"No collision-free path exists..."` after expanding all 9 reachable cells, rather than hanging or crashing - confirmed by the `visited` set bounding the search, exactly as in the search labs.

---

## Reflection: Strengths and Limitations of LLM-Assisted Software Engineering (Learning Objective 5)

**Strengths observed in this lab:** given the goal-based-agent architecture specified independently in Task 2 (environment / state / goal / decision-making component, as distinct, named pieces), translating that into working Python with a sensible algorithm choice (BFS, correctly justified for uniform-cost grid search) took one prompt and needed no debugging - a genuine productivity gain over writing the grid-parsing and BFS bookkeeping from scratch.

**Limitations observed:** the first prompt, exactly as suggested by the lab handout, asked for a path to be found and printed, but not for that path to be *independently checked* - a planner with a subtle bug (e.g., one that occasionally returns a path that clips a corner, or that doesn't actually verify every intermediate cell is free) could satisfy "prints a path" while still being wrong, and nothing in the first prompt's requirements would catch that. This mirrors the exact lesson from `search_lab/` and `llm_bn_lab/` (Bayesian networks) in this submission: a generated program's *output looking plausible* is not evidence of correctness on its own. The fix here was not a bug fix but a second, deliberate prompt asking for an independent validator - a concrete example of "improving software through iterative prompting" that improves the *engineering rigor* of the result, not just its functionality.
