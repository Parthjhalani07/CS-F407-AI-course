"""
Simple STRIPS-style planning agent.

Generated with the assistance of an LLM using the prompt in prompt.txt
(Task 2 of the lab). Lightly reviewed/cleaned up by hand afterwards but the
core algorithm (Action representation, applicability check, apply, and BFS
search) is exactly what was produced by the LLM.

Implementation notes / assumptions (as requested at the end of the prompt):

- A state is represented as a frozenset of proposition strings, e.g.
  frozenset({"At(Robot,A)", "At(Package,A)"}). Using a frozenset makes
  states hashable so they can be stored in a `visited` set and used as
  dictionary keys during BFS.
- An Action has four proposition sets: pos_preconditions, neg_preconditions,
  pos_effects, neg_effects. A negative precondition means "this proposition
  must NOT be true in the current state" (not used by the warehouse problem,
  but supported for generality).
- Applicability: `is_applicable(state, action)` returns True iff
  pos_preconditions ⊆ state AND neg_preconditions ∩ state = ∅.
  This is the S |= Preconditions(a) check from the handout.
- Applying an action never checks applicability itself — the caller
  (the search routine) is responsible for only ever calling `apply` on
  actions it has already verified are applicable. This mirrors how the
  handout separates "is this action usable here" (logic) from "what do we
  do with it" (search / effects).
- The search assumes actions are all instantiated ahead of time (no
  variables/parameters are resolved during search) — i.e. Move(A,B) and
  Move(B,C) are distinct, fully-grounded Action objects supplied in the
  `actions` list. This keeps the search itself independent of any planning
  domain and matches how the problem is specified in the handout (a fixed
  list of grounded actions with concrete preconditions/effects).
- BFS explores states in order of increasing plan length, so the first goal
  state found is reached by a shortest plan (by number of actions). Cycles
  are avoided with a `visited` set of previously-expanded states.
- If the goal is never reached during the search, `plan()` returns None,
  which the caller reports as "No plan found" instead of inventing an
  action.
"""

from collections import deque


class Action:
    def __init__(self, name, pos_preconditions=None, neg_preconditions=None,
                 pos_effects=None, neg_effects=None):
        self.name = name
        self.pos_preconditions = frozenset(pos_preconditions or [])
        self.neg_preconditions = frozenset(neg_preconditions or [])
        self.pos_effects = frozenset(pos_effects or [])
        self.neg_effects = frozenset(neg_effects or [])

    def __repr__(self):
        return self.name


def is_applicable(state, action):
    """S |= Preconditions(a)"""
    return (action.pos_preconditions.issubset(state)
            and action.neg_preconditions.isdisjoint(state))


def apply(state, action):
    """S' = Apply(S, a). Assumes is_applicable(state, action) is True."""
    return frozenset((state - action.neg_effects) | action.pos_effects)


def plan(initial_state, actions, goal):
    """
    Breadth-first search over states for a sequence of actions that
    reaches a state satisfying `goal` (goal ⊆ state).

    Returns (action_sequence, state_sequence) on success, where
    state_sequence[0] == initial_state and state_sequence[i+1] is the
    state after applying action_sequence[i].

    Returns None if no plan exists.
    """
    initial_state = frozenset(initial_state)
    goal = frozenset(goal)

    if goal.issubset(initial_state):
        return [], [initial_state]

    visited = {initial_state}
    # queue holds (state, action_path, state_path)
    queue = deque([(initial_state, [], [initial_state])])

    while queue:
        state, action_path, state_path = queue.popleft()

        for action in actions:
            if not is_applicable(state, action):
                continue

            next_state = apply(state, action)
            if next_state in visited:
                continue

            next_action_path = action_path + [action]
            next_state_path = state_path + [next_state]

            if goal.issubset(next_state):
                return next_action_path, next_state_path

            visited.add(next_state)
            queue.append((next_state, next_action_path, next_state_path))

    return None  # no plan exists


def run_planner(initial_state, actions, goal, verbose=True):
    """Runs plan() and prints the result in the format required by the lab."""
    result = plan(initial_state, actions, goal)

    if result is None:
        if verbose:
            print("No plan found")
        return None

    action_seq, state_seq = result

    if verbose:
        print("Plan found:")
        print("  S0:", sorted(state_seq[0]))
        for i, (a, s) in enumerate(zip(action_seq, state_seq[1:]), start=1):
            print(f"  Action {i}: {a}")
            print(f"  S{i}:", sorted(s))

    return action_seq, state_seq


# ---------------------------------------------------------------------------
# Warehouse problem (Section 3 of the handout)
# ---------------------------------------------------------------------------

def warehouse_actions(include_pickup=True):
    actions = [
        Action("Move(A,B)", pos_preconditions={"At(Robot,A)"},
               pos_effects={"At(Robot,B)"}, neg_effects={"At(Robot,A)"}),
        Action("Move(B,A)", pos_preconditions={"At(Robot,B)"},
               pos_effects={"At(Robot,A)"}, neg_effects={"At(Robot,B)"}),
        Action("Move(B,C)", pos_preconditions={"At(Robot,B)"},
               pos_effects={"At(Robot,C)"}, neg_effects={"At(Robot,B)"}),
        Action("Move(C,B)", pos_preconditions={"At(Robot,C)"},
               pos_effects={"At(Robot,B)"}, neg_effects={"At(Robot,C)"}),
        Action("Drop(Package,A)",
               pos_preconditions={"At(Robot,A)", "Holding(Package)"},
               pos_effects={"At(Package,A)"}, neg_effects={"Holding(Package)"}),
        Action("Drop(Package,B)",
               pos_preconditions={"At(Robot,B)", "Holding(Package)"},
               pos_effects={"At(Package,B)"}, neg_effects={"Holding(Package)"}),
        Action("Drop(Package,C)",
               pos_preconditions={"At(Robot,C)", "Holding(Package)"},
               pos_effects={"At(Package,C)"}, neg_effects={"Holding(Package)"}),
    ]

    if include_pickup:
        actions += [
            Action("PickUp(Package,A)",
                   pos_preconditions={"At(Robot,A)", "At(Package,A)"},
                   pos_effects={"Holding(Package)"}, neg_effects={"At(Package,A)"}),
            Action("PickUp(Package,B)",
                   pos_preconditions={"At(Robot,B)", "At(Package,B)"},
                   pos_effects={"Holding(Package)"}, neg_effects={"At(Package,B)"}),
            Action("PickUp(Package,C)",
                   pos_preconditions={"At(Robot,C)", "At(Package,C)"},
                   pos_effects={"Holding(Package)"}, neg_effects={"At(Package,C)"}),
        ]

    return actions


if __name__ == "__main__":
    initial_state = {"At(Robot,A)", "At(Package,A)"}
    goal = {"At(Package,C)"}
    run_planner(initial_state, warehouse_actions(), goal)
