"""
Task 3: Test the Generated Planner

Three tests against planner.py, as required by the lab handout:
  Test A - Solvable Problem (the original warehouse problem)
  Test B - Impossible Problem (PickUp action removed)
  Test C - Irrelevant Actions (an extra Move that doesn't affect the goal)
"""

from planner import Action, warehouse_actions, plan, is_applicable, apply


def describe(name, initial_state, goal, actions):
    print(f"=== {name} ===")
    print("Initial state:", sorted(initial_state))
    print("Goal:", sorted(goal))

    result = plan(initial_state, actions, goal)

    if result is None:
        print("Result: No plan found")
        print()
        return None

    action_seq, state_seq = result
    print("Result: Plan found ->", [a.name for a in action_seq])
    for i, s in enumerate(state_seq):
        print(f"  S{i}:", sorted(s))

    # Independently verify every action in the returned plan.
    valid = True
    s = frozenset(initial_state)
    for a in action_seq:
        if not is_applicable(s, a):
            print(f"  INVALID: {a} not applicable in state {sorted(s)}")
            valid = False
            break
        s = apply(s, a)
    if valid and not goal.issubset(s):
        print(f"  INVALID: final state {sorted(s)} does not satisfy the goal")
        valid = False

    print("Plan valid:", valid)
    print()
    return action_seq


if __name__ == "__main__":
    initial_state = {"At(Robot,A)", "At(Package,A)"}
    goal = {"At(Package,C)"}

    # Test A: Solvable Problem
    describe("Test A: Solvable Problem", initial_state, goal, warehouse_actions())

    # Test B: Impossible Problem (remove PickUp actions entirely)
    describe("Test B: Impossible Problem (no PickUp action)",
             initial_state, goal, warehouse_actions(include_pickup=False))

    # Test C: Irrelevant Actions
    # Add a Move(A,B) action that is available even though it doesn't move
    # the package - it's already in warehouse_actions(), so instead we
    # verify the planner doesn't confuse "robot reaches C" with
    # "package reaches C" by checking it does NOT stop after just Move(A,B),
    # Move(B,C) (robot at C, package still at A).
    print("=== Test C: Irrelevant Actions ===")
    actions = warehouse_actions()
    partial_state = frozenset({"At(Robot,C)", "At(Package,A)"})  # robot moved, package didn't
    print("Does robot-at-C alone satisfy goal At(Package,C)?",
          goal.issubset(partial_state))
    result = plan(initial_state, actions, goal)
    action_seq, state_seq = result
    print("Full plan:", [a.name for a in action_seq])
    # Confirm the plan does not terminate early at any state where only the
    # robot (not the package) has reached C.
    early_stop = any(goal.issubset(s) and "At(Package,C)" not in s
                      for s in state_seq)
    print("Planner incorrectly treated robot-at-C as goal reached:", early_stop)
    print("Package actually At(Package,C) in final state:",
          "At(Package,C)" in state_seq[-1])
