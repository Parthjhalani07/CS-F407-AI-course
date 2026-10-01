"""
Tests and validation for the generated agent (Task 3's "test to see if the
generated program runs", plus the independent path-validity check added in
the second prompting iteration - see prompts.txt).

Run: python3 test_agent.py
"""

from warehouse_agent import Environment, GoalBasedAgent, WAREHOUSE_MAP, MOVES

NO_PATH_MAP = """\
#######
#S....#
###.###
#...#G#
#######
"""

TRIVIAL_MAP = """\
#####
#SG##
#####
"""


def validate_path(env, path):
    """
    Independently check a returned path: every cell must be free, every
    consecutive pair must differ by exactly one valid move, the path must
    start at S and end at G. This does not trust the planner's own
    bookkeeping - it re-derives validity from the environment directly.
    """
    assert path[0] == env.start, "path does not start at S"
    assert path[-1] == env.goal, "path does not end at G"
    for pos in path:
        assert env.is_free(pos), f"path steps on a non-free cell: {pos}"
    for a, b in zip(path, path[1:]):
        delta = (b[0] - a[0], b[1] - a[1])
        assert delta in MOVES.values(), f"not a single valid move: {a} -> {b}"
    return True


def run_case(name, ascii_map, expect_path):
    print(f"\n--- {name} ---")
    env = Environment(ascii_map)
    agent = GoalBasedAgent(env)
    path = agent.act()
    if expect_path:
        assert path is not None, "expected a path but none was found"
        validate_path(env, path)
        print("Independent path validation: PASSED "
              "(every step is a valid move onto a free cell, S->...->G)")
    else:
        assert path is None, "expected no path, but one was found"
        print("Correctly reported no path (did not loop or crash).")


if __name__ == "__main__":
    run_case("Original warehouse map", WAREHOUSE_MAP, expect_path=True)
    run_case("Trivial case (goal adjacent to start)", TRIVIAL_MAP, expect_path=True)
    run_case("No-path case (goal walled off)", NO_PATH_MAP, expect_path=False)
    print("\nAll tests passed.")
