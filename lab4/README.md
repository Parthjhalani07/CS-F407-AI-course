# Lab 4 - Search and A*

**Course:** Undergraduate Artificial Intelligence
**Topic:** Using an LLM to Construct and Test a Simple Planning/Search Agent
**Author:** Parth Jhalani

> `planner.py` and `test_planner.py` were written with the assistance of **Claude (Anthropic)**,
> based on a design specified independently beforehand (Task 1), and subsequently run, tested,
> and modified independently (Tasks 3-6). All test results and written answers below are from
> actually running the code, not invented.

---

## Files

| File | Description |
|------|-------------|
| `planner.py` | Warehouse map parser, BFS agent, A* agent (parameterised heuristic) |
| `test_planner.py` | All systematic tests: Task 3, Task 5 (BFS/A* comparison), Task 6 (heuristic investigation) |
| `prompt.txt` | The prompt used with the LLM (Task 2) |
| `full_test_output.txt` | Raw captured output of `python3 test_planner.py` |
| `README.md` | This file - full submission |

## How to Run

```bash
python3 planner.py          # runs BFS and A* once on the warehouse map
python3 test_planner.py     # runs the full test/comparison/heuristic suite
```

---

## Task 0: Understand the Search Problem

| Component | Specification |
|---|---|
| State *S* | A grid position `(row, col)` of the robot. |
| Actions *A* | `Up`, `Down`, `Left`, `Right` - each attempts to move the robot one cell in that direction. |
| Transition *T* | `T((r,c), action) = (r', c')` if `(r', c')` is in bounds and is not `#`; undefined (action invalid) otherwise. |
| Initial state *s₀* | The cell marked `S` in the map. |
| Goal *G* | `{(r, c) : map[r][c] == 'G'}` - a single target cell. |
| Cost *c* | 1 per move, uniform. |

(a) **What information is necessary to specify a state?** Just the robot's `(row, col)` position - the warehouse layout itself is static background knowledge, not part of the state, since it never changes between states.

(b) **What makes an action invalid?** Either the destination cell is outside the grid bounds, or the destination cell is an obstacle (`#`).

(c) **Is this a deterministic search problem?** Yes - every action, from every state, has exactly one outcome. There is no uncertainty in the transition function.

(d) **What would constitute a solution?** A sequence of actions `a1, ..., an` such that applying them in order from `s0` is always valid (every intermediate state satisfies the corresponding action's preconditions) and the final state is a member of `G`.

---

## Task 1: Plan the Agent (design, before using an LLM)

1. **State representation in Python:** a `(row, col)` tuple of ints. Tuples are hashable and immutable, so they work directly as dictionary keys and set members - needed for the visited-set and the `came_from` map.
2. **Warehouse representation:** the ASCII map is parsed once into a `list[str]` (one string per row). `grid[r][c]` gives the symbol at that cell in O(1). `S` and `G` are located during parsing and then treated like free cells (`.`) during search.
3. **Valid actions:** for a given state, try all four `(dr, dc)` offsets; an action is valid if the resulting cell is in bounds and not `#`.
4. **Goal recognition:** compare the current state to the single goal position found during parsing (`current == goal`).
5. **Frontier contents:** for BFS, a FIFO queue of states is enough (cost is uniform, so insertion order is expansion order). For A*, a min-heap of `(f, tie_breaker, state)`, plus a `g_score` dict mapping state to best known cost-so-far, since A* needs to update that information every time a cheaper path to a state is found.
6. **Path reconstruction:** a `came_from: dict[state, parent_state]` populated whenever a state is first reached (BFS) or whenever its `g_score` improves (A*). The path is rebuilt by walking `came_from` backwards from the goal to the start, then reversing.

Minimum information reported at termination: whether a solution was found, the solution path, the path length, and the number of states expanded - all implemented in `planner.py`'s `print_result`.

---

## Task 2: Ask an LLM to Generate A*

The prompt used is in [`prompt.txt`](prompt.txt); it is based on the design above (state = grid cell, heuristic = Manhattan distance, report `g`/`h`/`f`, path, and expansion count), plus a request to keep the heuristic swappable and to also produce a BFS variant sharing the same state/transition model for later comparison.

### Code explanation (Task 2 / Task 4 combined)

| Concept | Location in `planner.py` |
|---|---|
| State | `(row, col)` tuples throughout; produced by `parse_map` |
| Action | the `MOVES` dict (`"Up": (-1,0)`, etc.) |
| Transition | `neighbors()` - applies `MOVES` and filters by `in_bounds`/`is_free` |
| Goal test | `current == goal` inside `bfs` / `astar` |
| g(n) | `g_score` dict in `astar`; implicitly "steps taken" via BFS layer order in `bfs` |
| h(n) | the `heuristic(state, goal)` argument - `manhattan`, `euclidean`, or `zero_heuristic` |
| f(n) | computed inline as `tentative_g + weight * heuristic(nxt, goal)` before pushing to the heap |
| Frontier | `deque` for BFS; `heapq` min-heap of `(f, counter, state)` for A* |
| Visited states | `visited` set (BFS); `closed` set + `g_score` dict (A*, so a state can be re-opened if a cheaper path is later found) |
| Path reconstruction | shared `reconstruct_path()` helper, walking `came_from` backwards from goal to start |

(a) **What data structure is used for the A* frontier?** A binary min-heap (`heapq`), keyed on `(f, counter, state)`. The `counter` is a strictly increasing tie-breaker so that two states with equal `f` never get compared against each other directly (tuples would otherwise fall through to comparing the state tuples themselves, which is unnecessary work and, for non-comparable state types, would error).

(b) **How does the program select the next state to expand?** `heapq.heappop` always returns the entry with the smallest `f`; ties are broken by insertion order via `counter`.

(c) **Where is the heuristic calculated?** Inside the neighbor-relaxation loop in `astar`, once per candidate successor, as `heuristic(nxt, goal)`.

(d) **Does the program explicitly calculate f(n) = g(n) + h(n)?** Yes, literally: `f = tentative_g + weight * heuristic(nxt, goal)`.

(e) **How does the program prevent unnecessary repeated exploration?** A state is only pushed back onto the frontier if its `g_score` strictly improves (`tentative_g < g_score.get(nxt, inf)`). A popped entry is skipped if the state is already in `closed` (handles stale heap entries left over from an earlier, worse `g`).

---

## Task 3: Test the Generated Program

Full raw output: [`full_test_output.txt`](full_test_output.txt). Run with `python3 test_planner.py`.

### Test 1 - Original warehouse (from the lab handout)
```
Solution found : True
Path length    : 40 moves
States expanded: 64
```
The warehouse map turns out to contain exactly 64 free cells and **no branching at all** - it is a single winding corridor from `S` to `G`. (Verified: `sum(row.count('.')+row.count('S')+row.count('G') for row in grid) == 64`.) That is why, as Task 5 below shows, BFS and A* behave identically on this particular map - there is no choice of direction for a heuristic to usefully prune.

### Test 2 - Trivial case (`#SG##` in a 3-row box)
```
Solution found : True
Path length    : 1 moves
Path           : [(1, 1), (1, 2)]
States expanded: 2
```
Correct: the one-step solution is found immediately.

### Test 3 - No solution (goal walled off)
```
Solution found : False
States expanded: 9
```
The program explores every reachable cell (9 of them), fails to find `G` among them, exhausts the frontier, and reports failure - it does not loop forever, because the `closed`/`visited` set guarantees every state is expanded at most once.

### Test 4 - Alternative paths (open room with a central block)
```
A*:  found=True  path_len=11  expanded=30
BFS: found=True  path_len=11  expanded=30
```
Both algorithms return a path of the same (optimal) length 11, confirming A* with an admissible heuristic still finds a shortest path even when more than one route exists.

---

## Task 5: Compare A* with Blind Search

| Map | Measure | BFS | A* (Manhattan) |
|---|---|---|---|
| Original warehouse (single corridor, 64 free cells) | Solution found | True | True |
| | Path length | 40 | 40 |
| | States expanded | 64 | 64 |
| Open room w/ central block | Solution found | True | True |
| | Path length | 11 | 11 |
| | States expanded | 30 | 30 |
| Large open room, S and G on the same row, far apart (`_build_corridor_room_map`) | Solution found | True | True |
| | Path length | 20 | 20 |
| | States expanded | **309** | **21** |

(a) **Did both algorithms find a solution?** Yes, on every map tested.

(b) **Did they find paths of the same length?** Yes, always - both are guaranteed optimal here (BFS because costs are uniform, A* because Manhattan distance is admissible and consistent on a 4-connected uniform-cost grid).

(c) **Which algorithm expanded fewer states?** It depends entirely on the map. On the original warehouse and the small open room, they tied. On the large open room, A* expanded **21 states vs BFS's 309** - roughly 15x fewer.

(d) **Why might A* expand fewer states?** BFS expands outward from `S` as a uniform "diamond" in every direction, with no notion of which direction the goal lies in; it only stops once it happens to dequeue the goal. A* uses `h(n)` to bias the frontier toward states that are plausibly closer to the goal, so states that move *away* from the goal (and would never lie on an optimal path) get a higher `f` and are expanded later or not at all. This only translates into a measurable advantage when the map actually gives BFS somewhere "wasteful" to expand into - a long corridor or small room with no real alternative directions gives the heuristic nothing to prune, which is exactly why the first two maps tied.

---

## Task 6: Investigate the Heuristic

**Why is Manhattan distance appropriate here?** The robot can only move horizontally or vertically (no diagonals), one cell at a time, cost 1 per move. In an *obstacle-free* grid the Manhattan distance `|dx|+|dy|` is exactly the number of moves required - it is the true cost-to-go whenever nothing blocks the direct staircase path. With obstacles it can only ever *underestimate* the true cost (a detour can only make the real path longer, never shorter than the straight-line grid distance), so it remains admissible and keeps A* optimal.

### Experimental results

| Heuristic | Original warehouse | Open room (block) | Large open room |
|---|---|---|---|
| Manhattan | len 40, expanded 64 | len 11, expanded 30 | len 20, expanded **21** |
| `h(n) = 0` | len 40, expanded 64 | len 11, expanded 30 | len 20, expanded **309** |
| Euclidean | len 40, expanded 64 | len 11, expanded **28** | len 20, expanded 21 |
| Manhattan × 2 (inadmissible) | len 40, expanded 64 | len 11, expanded **12** | len 20, expanded 21 |

1. **`h(n) = 0`:** A* degenerates exactly into uniform-cost search, which - since every move costs 1 - behaves identically to BFS. This is confirmed numerically: 309 expansions on the large open room, matching BFS exactly from the Task 5 table. This is the expected theoretical result: A* with a zero heuristic is Dijkstra/UCS, and UCS with uniform costs explores states in the same order as BFS.
2. **Euclidean distance:** Still admissible (straight-line distance ≤ any 4-connected grid distance) but a slightly different ranking than Manhattan among diagonal neighbors; on the open room with a block it pruned marginally better (28 vs 30), on the large open room about the same as Manhattan (21). It remains a valid, if less natural, admissible heuristic for a grid that only allows orthogonal moves.
3. **Manhattan × 2 (inadmissible, `h` overestimates):** Expands far fewer states (12 vs 30 on the open room with a block) because an overestimated `h` makes the search much greedier - it commits hard to whatever looks closest to the goal and rarely reconsiders. It still happened to return the optimal-length path on every map tested here, but that is not guaranteed in general: an inadmissible heuristic can cause A* to settle for the first path found even when it is **not** the shortest one, because it can stop expanding a cheaper alternative before exploring it (its inflated `f` makes it look worse than it really is).

**What happens as the heuristic becomes too optimistic or too aggressive?**
- Too optimistic (`h` too small, e.g. `h=0`): A* is still optimal, but loses its informedness and degrades toward blind search - expanding needlessly many states, as seen above (309 vs 21).
- Too aggressive (`h` too large, overestimating, e.g. ×2 Manhattan): A* expands very few states (fast) but **loses the optimality guarantee** - it is willing to accept a worse path because the inflated heuristic makes the true shortest path look artificially expensive relative to a path that happens to be explored first. The experiments above did not happen to produce a suboptimal path with ×2 Manhattan, but that is a property of these particular maps (few genuinely competing routes of different length), not a property of the algorithm.

---

## Task 7: Evaluate the LLM-Generated Agent

1. **What parts of the generated code were correct immediately?** The overall structure (state = tuple, `heapq`-based frontier, `g_score`/`came_from` dicts, `reconstruct_path`) worked on the first run without changes.
2. **Did you find any bugs or design problems?** One real issue during development: an initial version of the A* loop used `heapq` entries of the form `(f, state)` without a tie-breaker. When two states had equal `f`, Python tried to compare the `state` tuples directly as a fallback, which worked by luck here (tuples of ints compare fine) but is fragile and would error for non-orderable state types; this was fixed by adding the `counter` tie-breaker, which is also faster since it avoids ever comparing state tuples.
3. **How did you discover those problems?** By reading the comparison logic carefully while writing Task 4's "where does each concept appear" table, not from a crash - the bug was latent, not loud, which is itself the point of Task 3's "working output ≠ validated algorithm" warning.
4. **Did the LLM use terminology or data structures you did not understand?** No - `heapq`, `deque`, and dict-based `came_from`/`g_score` maps are standard Python/CS vocabulary covered in the course's search module.
5. **Did you modify the LLM-generated code?** Yes: added the heap tie-breaker described above, added the `weight` parameter to `astar` so Task 6 could scale the heuristic without duplicating the search loop, and added `zero_heuristic`/`euclidean` as drop-in alternatives to `manhattan`.
6. **Which tests were most useful?** Test 3 (no-solution map) was the most valuable, because it is the one test where a subtly wrong implementation (e.g., missing visited/closed tracking) would hang forever instead of just giving a wrong-but-plausible answer - it directly tests whether termination is actually guaranteed, not just typical-case behavior.
7. **Could you have trusted the program without testing it?** No. The BFS-vs-A* tie on the first two maps is a good example: without constructing the large, deliberately "wasteful" open-room map, it would have looked like A* provides **no** benefit over BFS at all, which is a wrong conclusion about the algorithm, not just about these two maps.
8. **What did you understand about A* that you did not understand before implementing it?** That A*'s advantage over blind search is not intrinsic to the algorithm - it is entirely a function of how much the heuristic and the map's structure let it prune directions that don't lead to the goal. On a single corridor or a small room, "informed" search has nothing to be informed *about*.

---

## Final Reflection

1. **Why is it important to formulate the search problem before writing the search algorithm?** Writing down `S`, `A`, `T`, `s0`, `G`, `c` first forces precise decisions - e.g., exactly what counts as a valid action - that the code must implement correctly. Without this, it's easy to write code that looks reasonable (it runs, it returns *a* path) while silently encoding the wrong problem, such as allowing diagonal moves or forgetting to treat the boundary as an obstacle.
2. **In what sense is A* an "informed" search algorithm?** It uses problem-specific knowledge beyond the raw graph structure - the heuristic `h(n)`, an estimate of remaining cost - to decide which frontier node to expand next, rather than expanding purely in discovery order (BFS) or depth order (DFS). This lets it direct its effort toward the goal instead of exploring uniformly in every direction.
3. **Why does the choice of heuristic matter?** As Task 6 shows concretely, the same algorithm with `h=0` degrades to blind search (309 expansions), with Manhattan distance it is dramatically more efficient (21 expansions) while remaining optimal, and with an inflated, inadmissible heuristic it becomes even faster but can lose the optimality guarantee. The heuristic is the only thing that distinguishes "informed" from "blind" search - a bad or absent one throws away exactly the advantage A* is supposed to provide.
4. **What did the LLM contribute to the engineering process?** It translated an already-specified design (state representation, frontier contents, termination conditions, from Task 1) into working Python quickly - the `heapq`-based priority queue, the `g_score`/`came_from` bookkeeping, and a BFS sibling sharing the same state/transition code. It did not decide what the search problem *was*; that specification work (Tasks 0-1) was done first, independently.
5. **What could go wrong if an engineer simply accepted LLM-generated code without testing it?** The heap tie-breaker bug (point 2 of Task 7) is a concrete example: it would not crash and would often produce correct-looking results, but could fail unpredictably depending on state representation or Python version, and the BFS/A* tie on the original warehouse map could easily be mistaken for "A* doesn't help here" when it is actually a property of that specific map, not of the algorithm - a wrong general conclusion drawn from an untested special case.

**AI Science → AI Engineering.** The scientific idea is informed search; A* is one algorithm implementing it. The LLM was a useful tool for implementing that algorithm quickly, but every claim in this report - that the algorithms are correct, that A* can expand far fewer states than BFS, that an inadmissible heuristic trades optimality for speed - rests on the test results actually produced by running the code above, not on the LLM's explanation of its own output.
