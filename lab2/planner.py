"""
Laboratory - Logical Reasoning for Planning
Warehouse Robot Planning Agent (BFS)

Planning problem (I, A, G):
  I = {At(Robot,A), At(Package,A)}
  G = {At(Package,C)}
  A = Move(X,Y), PickUp(Package,L), Drop(Package,L)

Generated with assistance of Claude (Anthropic) and independently tested/verified.
"""

from collections import deque


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

class Action:
    """STRIPS-style action: name, positive/negative preconditions and effects."""

    def __init__(self, name, pos_preconds, neg_preconds, pos_effects, neg_effects):
        self.name = name
        self.pos_preconds = frozenset(pos_preconds)  # must hold in state
        self.neg_preconds = frozenset(neg_preconds)  # must NOT hold in state
        self.pos_effects   = frozenset(pos_effects)  # become true
        self.neg_effects   = frozenset(neg_effects)  # become false

    def is_applicable(self, state: frozenset) -> bool:
        """S |= Preconditions(a)  iff  all positive preconds in S and all negative preconds absent."""
        return self.pos_preconds <= state and self.neg_preconds.isdisjoint(state)

    def apply(self, state: frozenset) -> frozenset:
        """Return S' = Apply(S, a): remove neg-effects, add pos-effects."""
        return (state - self.neg_effects) | self.pos_effects

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Action({self.name!r})"


# ---------------------------------------------------------------------------
# BFS planner
# ---------------------------------------------------------------------------

def bfs_plan(initial_state, goal, actions):
    """
    Breadth-first search for a valid plan.

    Returns a list of (Action, resulting_state) pairs,
    an empty list if the goal already holds,
    or None if no plan exists.

    Logic + Search = Planning:
      - Logic:  is_applicable checks S |= Preconditions(a)
      - Search: BFS explores the space of reachable states level by level
    """
    initial_state = frozenset(initial_state)
    goal          = frozenset(goal)

    if goal <= initial_state:
        return []                          # already satisfied

    frontier = deque([(initial_state, [])])
    visited  = {initial_state}

    while frontier:
        state, plan = frontier.popleft()

        for action in actions:
            if not action.is_applicable(state):   # logical check
                continue

            new_state = action.apply(state)       # state transition

            if new_state in visited:
                continue

            new_plan = plan + [(action, new_state)]

            if goal <= new_state:                 # goal test
                return new_plan

            visited.add(new_state)
            frontier.append((new_state, new_plan))

    return None                                   # exhausted all states


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def run_planner(initial_state, goal, actions, test_name=""):
    """Run BFS planner and print a formatted trace."""
    print(f"\n{'='*65}")
    if test_name:
        print(f"TEST: {test_name}")
    print(f"Initial state : {sorted(initial_state)}")
    print(f"Goal          : {sorted(goal)}")
    print(f"Actions       : {[str(a) for a in actions]}")
    print(f"{'-'*65}")

    result = bfs_plan(initial_state, goal, actions)

    if result is None:
        print("Result        : No plan found.")
    else:
        print(f"Result        : Plan found ({len(result)} step(s))\n")
        state = frozenset(initial_state)
        print(f"  S0  : {sorted(state)}")
        for i, (action, new_state) in enumerate(result):
            print(f"  a{i+1}  : {action}")
            state = new_state
            print(f"  S{i+1}  : {sorted(state)}")

        goal_satisfied = frozenset(goal) <= state
        print(f"\n  Goal satisfied in final state: {goal_satisfied}")

    print(f"{'='*65}")
    return result


# ---------------------------------------------------------------------------
# Warehouse domain
# ---------------------------------------------------------------------------

def make_warehouse_actions(include_pickup=True, include_drop=True):
    """
    Build the action set for the warehouse robot problem.
    Connections: A <-> B <-> C  (no direct A-C link)
    """
    actions = [
        Action("Move(A,B)",
               pos_preconds=["At(Robot,A)"], neg_preconds=[],
               pos_effects=["At(Robot,B)"],  neg_effects=["At(Robot,A)"]),
        Action("Move(B,A)",
               pos_preconds=["At(Robot,B)"], neg_preconds=[],
               pos_effects=["At(Robot,A)"],  neg_effects=["At(Robot,B)"]),
        Action("Move(B,C)",
               pos_preconds=["At(Robot,B)"], neg_preconds=[],
               pos_effects=["At(Robot,C)"],  neg_effects=["At(Robot,B)"]),
        Action("Move(C,B)",
               pos_preconds=["At(Robot,C)"], neg_preconds=[],
               pos_effects=["At(Robot,B)"],  neg_effects=["At(Robot,C)"]),
    ]

    if include_pickup:
        for loc in ["A", "B", "C"]:
            actions.append(Action(
                f"PickUp(Package,{loc})",
                pos_preconds=[f"At(Robot,{loc})", f"At(Package,{loc})"],
                neg_preconds=["Holding(Package)"],
                pos_effects=["Holding(Package)"],
                neg_effects=[f"At(Package,{loc})"],
            ))

    if include_drop:
        for loc in ["A", "B", "C"]:
            actions.append(Action(
                f"Drop(Package,{loc})",
                pos_preconds=[f"At(Robot,{loc})", "Holding(Package)"],
                neg_preconds=[],
                pos_effects=[f"At(Package,{loc})"],
                neg_effects=["Holding(Package)"],
            ))

    return actions


# ---------------------------------------------------------------------------
# Main: run all three required tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    INITIAL = ["At(Robot,A)", "At(Package,A)"]
    GOAL    = ["At(Package,C)"]

    # ------------------------------------------------------------------
    # Test A - Solvable problem (original warehouse scenario)
    # Expected: a valid plan that achieves At(Package,C)
    # ------------------------------------------------------------------
    actions_full = make_warehouse_actions(include_pickup=True, include_drop=True)
    run_planner(INITIAL, GOAL, actions_full,
                test_name="A - Solvable: original warehouse problem")

    # ------------------------------------------------------------------
    # Test B - Impossible problem (PickUp action removed)
    # Expected: "No plan found" - the robot can reach C but cannot carry
    # the package there because it cannot pick it up.
    # ------------------------------------------------------------------
    actions_no_pickup = make_warehouse_actions(include_pickup=False, include_drop=True)
    run_planner(INITIAL, GOAL, actions_no_pickup,
                test_name="B - Impossible: PickUp action removed")

    # ------------------------------------------------------------------
    # Test C - Irrelevant actions (robot can reach C without package)
    # The Move actions let the robot reach C while the package stays at A.
    # The planner must NOT confuse At(Robot,C) with At(Package,C).
    # Expected: the BFS still finds the correct plan including PickUp/Drop,
    # and the final state satisfies At(Package,C) - not just At(Robot,C).
    # ------------------------------------------------------------------
    # We run the full action set; the irrelevant trajectories
    # (move robot to C without the package) are explored but rejected
    # because they do not satisfy the goal At(Package,C).
    run_planner(INITIAL, GOAL, actions_full,
                test_name="C - Irrelevant actions: robot reaching C ≠ package at C")

    # Explicit sub-check: with ONLY move actions the problem is unsolvable
    actions_move_only = make_warehouse_actions(include_pickup=False, include_drop=False)
    run_planner(INITIAL, GOAL, actions_move_only,
                test_name="C (sub-check) - Move-only: robot at C but package never moves")
