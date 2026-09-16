# Lab 2 - Logical Reasoning for Planning

**Course:** Undergraduate Artificial Intelligence  
**Topic:** Using an LLM to Construct and Test a Simple Planning Agent  
**Author:** Parth Jhalani

> Parts of `planner.py` were generated with the assistance of **Claude (Anthropic)** and subsequently verified, tested, and modified independently. `planner.pl` was written with LLM assistance. All test results and written answers are independently produced.

---

## Files

| File | Description |
|------|-------------|
| `planner.py` | BFS planning agent (Tasks 0-5) |
| `planner.pl` | Prolog verifier (Optional Tasks 6-8) |
| `README.md` | Full submission: specification, plan, prompt, results, answers |

---

## How to Run

```bash
python3 planner.py
```

For the optional Prolog extension (requires SWI-Prolog):
```bash
sudo apt install swi-prolog
swipl planner.pl
# then type queries like: ?- can_move(a,b).
```

---

## 1. Specification of the Planning Problem (Task 0)

### Initial State
```
I = { At(Robot,A),  At(Package,A) }
```

### Goal
```
G = { At(Package,C) }
```

### Warehouse Layout
Locations: **A**, **B**, **C**  
Connections: A ↔ B ↔ C (no direct A-C link)

### Actions, Preconditions, and Effects

| Action | Positive Preconditions | Negative Preconditions | Positive Effects | Negative Effects |
|--------|----------------------|----------------------|-----------------|-----------------|
| Move(A,B) | At(Robot,A) | - | At(Robot,B) | At(Robot,A) |
| Move(B,A) | At(Robot,B) | - | At(Robot,A) | At(Robot,B) |
| Move(B,C) | At(Robot,B) | - | At(Robot,C) | At(Robot,B) |
| Move(C,B) | At(Robot,C) | - | At(Robot,B) | At(Robot,C) |
| PickUp(Package,L) | At(Robot,L), At(Package,L) | Holding(Package) | Holding(Package) | At(Package,L) |
| Drop(Package,L) | At(Robot,L), Holding(Package) | - | At(Package,L) | Holding(Package) |

### Initially Applicable Actions (from I)

- **Move(A,B):** `At(Robot,A)` ✓ → **applicable**
- **PickUp(Package,A):** `At(Robot,A)` ✓ and `At(Package,A)` ✓ → **applicable**
- All others require propositions not in I → **not applicable**

**Is PickUp(Package,A) applicable?**  
Yes - both preconditions `At(Robot,A)` and `At(Package,A)` are in I.

**Is Drop(Package,C) applicable?**  
No - `At(Robot,C)` and `Holding(Package)` are both absent from I.

---

## 2. Manually Constructed Plan (Task 1)

| State | Facts |
|-------|-------|
| S₀ | At(Robot,A), At(Package,A) |
| S₁ | At(Robot,A), Holding(Package) |
| S₂ | At(Robot,B), Holding(Package) |
| S₃ | At(Robot,C), Holding(Package) |
| S₄ | At(Robot,C), At(Package,C) |

**Action sequence:**

```
a1: PickUp(Package,A)   [preconds: At(Robot,A)✓, At(Package,A)✓]
a2: Move(A,B)           [preconds: At(Robot,A)✓]
a3: Move(B,C)           [preconds: At(Robot,B)✓]
a4: Drop(Package,C)     [preconds: At(Robot,C)✓, Holding(Package)✓]
```

S₄ ⊨ G = { At(Package,C) } ✓

---

## 3. Prompt Used with the LLM (Task 2)

The following prompt was given to Claude (Anthropic):

> I want to implement a simple planning agent in Python.
> Represent a state as a set of logical propositions.
> Each action should contain:
> - a name
> - positive preconditions
> - negative preconditions
> - positive effects
> - negative effects
>
> An action is applicable if all of its preconditions are satisfied by the current state.
> When an action is applied:
> 1. remove its negative effects from the state;
> 2. add its positive effects to the state.
>
> Use breadth-first search to find a sequence of actions that achieves a specified goal.
> The program should also:
> - detect when no plan exists;
> - print the resulting sequence of actions;
> - print the states reached after each action.
>
> Explain the implementation and identify any assumptions you make.

The domain (warehouse locations, connections, and all action schemas) was then provided as part of the specification.

---

## 4. Generated Python Program (Task 2)

See [`planner.py`](planner.py).

**LLM-assisted parts:** The `Action` class structure (using `frozenset` for hashability), the `is_applicable` and `apply` methods, the BFS loop skeleton with `deque`, and the visited-state tracking.

**Independently added / verified:** The warehouse domain encoding (all actions with correct preconditions/effects), the three test cases, the goal-validity print at the end, and negative preconditions support.

**Key design decisions:**
- State = Python `frozenset` of strings (hashable, immutable, supports set operations)
- `is_applicable`: `pos_preconds ⊆ state` AND `neg_preconds ∩ state = ∅`
- `apply`: `new_state = (state − neg_effects) ∪ pos_effects`
- BFS guarantees the shortest plan (fewest actions)

**Where specification concepts appear in the code:**

| Concept | Location |
|---------|----------|
| Preconditions | `Action.is_applicable()` - checks S ⊨ Preconditions(a) |
| Effects | `Action.apply()` - computes S' = Apply(S, a) |
| Goal | `goal <= new_state` in `bfs_plan()` |
| BFS | `deque`-based frontier in `bfs_plan()` |

---

## 5. Test Results (Task 3)

### Test A - Solvable Problem

```
Initial state : ['At(Package,A)', 'At(Robot,A)']
Goal          : ['At(Package,C)']

Result        : Plan found (4 step(s))

  S0  : ['At(Package,A)', 'At(Robot,A)']
  a1  : PickUp(Package,A)
  S1  : ['At(Robot,A)', 'Holding(Package)']
  a2  : Move(A,B)
  S2  : ['At(Robot,B)', 'Holding(Package)']
  a3  : Move(B,C)
  S3  : ['At(Robot,C)', 'Holding(Package)']
  a4  : Drop(Package,C)
  S4  : ['At(Package,C)', 'At(Robot,C)']

  Goal satisfied in final state: True
```

**Plan verification:**
- a1 PickUp(Package,A): At(Robot,A)✓ At(Package,A)✓ → valid
- a2 Move(A,B): At(Robot,A)✓ → valid
- a3 Move(B,C): At(Robot,B)✓ → valid
- a4 Drop(Package,C): At(Robot,C)✓ Holding(Package)✓ → valid

### Test B - Impossible Problem (PickUp removed)

```
Initial state : ['At(Package,A)', 'At(Robot,A)']
Goal          : ['At(Package,C)']
Actions       : Move only + Drop (no PickUp)

Result        : No plan found.
```

The robot can reach C but can never carry the package - correctly reports no plan rather than inventing an action.

### Test C - Irrelevant Actions

```
Initial state : ['At(Package,A)', 'At(Robot,A)']
Goal          : ['At(Package,C)']

Result        : Plan found (4 step(s))   [same correct plan as Test A]

  Goal satisfied in final state: True

Sub-check (Move actions only - robot reaches C, package stays at A):
Result        : No plan found.
```

The BFS explores paths where the robot moves to C without the package, but correctly rejects them because `At(Package,C)` is not satisfied. This confirms the planner does **not** treat `At(Robot,C)` as equivalent to `At(Package,C)`.

---

## 6. Answers to "Think About It" Questions (Task 4)

### Task 0 - Think About It
> An action should not be considered applicable merely because it appears in the list of available actions. Ask: are all of its preconditions satisfied in the current state?

This is where logical reasoning enters planning: we check **S ⊨ Preconditions(a)** before allowing any action. In code, `is_applicable` enforces this - an action that fails the check is simply skipped, not applied.

### Task 2 - Think About It (where spec concepts appear)

| Concept | Code location |
|---------|--------------|
| Preconditions | `is_applicable` - guards whether an action can fire |
| Effects | `apply` - defines how the state changes |
| Goal | `goal <= new_state` - termination condition |
| BFS | `deque` frontier - how alternative plans are explored |

### Task 4 - Logic and Search

The missing step in the planning loop diagram is **"Apply action"**:

```
Current state
      ↓
Check action preconditions   ← LOGIC (S ⊨ Preconditions(a)?)
      ↓
Apply action                 ← LOGIC (S' = Apply(S, a))
      ↓
Generate successor state
      ↓
Search over alternatives     ← SEARCH (BFS frontier)
      ↓
Goal?
```

**How they work together:**  
Logic determines *whether* each action can be taken and *what* state results. Search determines *which* sequence of applicable actions to try. In short:

> **Logic determines what is possible; search determines what to try.**

### Task 5 - Which to trust: LLM explanation or independent state transitions?

**Trust the independently executed state transitions.**  
The Python program mechanically applies each action's effects step by step and produces a verifiable trace. An LLM explanation is a generated narrative that can sound plausible even when incorrect. The program cannot lie about what frozenset results from each step - the goal check either passes or it doesn't.

> Key principle: **A generated explanation is not the same as an independent verification.**

---

## 7. Reflection on the Use of the LLM (Section 5)

### Reflection Questions

**1. Why is it useful to specify action preconditions and effects before asking an LLM to write the planner?**  
Specifying the formal action schema upfront gives the LLM an unambiguous contract to implement. Without it, the LLM might guess what "action" means and produce code that does not properly gate on preconditions or apply effects correctly.

**2. Give an example of an error that could occur if the planner failed to check preconditions.**  
The planner could apply `Drop(Package,C)` in the initial state - the robot is at A and not holding the package - producing a sequence that appears to deliver the package without ever picking it up.

**3. Why is a plan that "looks reasonable" not necessarily a valid plan?**  
A plan is valid only if every action's preconditions are satisfied in the state *at the moment that action executes*. A plan can look sensible on paper (e.g., "move to C, drop package") but fail because an intermediate step (picking up the package) was omitted.

**4. What did the LLM contribute to the implementation?**  
The BFS structure using `deque`, the `Action` class with `frozenset` fields, the `is_applicable` and `apply` methods, and the overall planner loop skeleton.

**5. What did you have to verify independently?**  
That the goal check uses `<=` (subset), not `==` (equality); that visited-state tracking prevents infinite loops; that negative preconditions are correctly handled; and that all three tests produce the expected outputs.

**6. In this laboratory, where is logical reasoning being used?**  
In `Action.is_applicable()`: checking `S ⊨ Preconditions(a)` is a direct logical entailment check - every required proposition must be present and every forbidden proposition must be absent.

**7. How is planning related to the search algorithms studied in the previous module?**  
The BFS planner is structurally identical to standard BFS - a frontier queue, a visited set, and a goal test. The novelty is that transitions are *generated* by actions with precondition guards, rather than read from a fixed graph.

---

## Optional Extension: Prolog (Tasks 6-8)

See [`planner.pl`](planner.pl).

### Task 6 - Queries and Results

```prolog
?- can_move(a,b).   % true  - connected(a,b) fact exists
?- can_move(a,c).   % false - no connected(a,c) fact
```

**(a)** `can_move(a,b)` is true because `connected(a,b)` is a base fact and the rule `can_move(X,Y) :- connected(X,Y)` fires directly.  
**(b)** `can_move(a,c)` fails because there is no `connected(a,c)` fact - A and C have no direct link.  
**(c)** The Prolog rule `can_move(X,Y) :- connected(X,Y)` directly encodes the logical implication Connected(X,Y) → CanMove(X,Y).

### Task 7 - Plan Verification

```prolog
?- valid_move(a,b).   % true
?- valid_move(b,c).   % true
?- valid_move(a,c).   % false  ← Move(a,c) is NOT supported
```

The Python planner's generated moves (a→b, b→c) are both verified. The challenge move `Move(a,c)` is correctly rejected.

### Task 8 - Wet-road logical reasoning chain

```
wet_road  ⟹  slippery  ⟹  reduce_speed
```

`?- reduce_speed.` succeeds via backward chaining. This illustrates **Facts + Rules → Inference → Query Answer**.

### Section 7.2 Reflection

**1. Difference between a Prolog fact and a rule?**  
A fact is an unconditional assertion (e.g., `wet_road.`). A rule derives new conclusions from existing ones (e.g., `slippery :- wet_road.`).

**2. How does a Prolog query correspond to logical entailment?**  
`?- X.` asks "does X follow from the knowledge base?" - equivalent to checking KB ⊨ X.

**3. Why use Prolog to verify a Python-generated plan?**  
Prolog provides an independent logical system. If the Python planner generates a candidate action, Prolog checks it against a separately specified knowledge base - catching errors the planner itself cannot detect.

**4. What advantage does an independent verifier provide when the plan was LLM-generated?**  
An LLM may generate a plausible-sounding explanation that is actually wrong. A formal logic system either derives the conclusion or it doesn't - it cannot hallucinate. This separates generation from verification:

> **Generate → Independent verification**
