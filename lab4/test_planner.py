"""
Systematic tests for the warehouse planner (Task 3), the BFS/A* comparison
(Task 5), and the heuristic investigation (Task 6).

Run: python3 test_planner.py
"""

from planner import (
    parse_map, bfs, astar, print_result,
    manhattan, euclidean, zero_heuristic, WAREHOUSE_MAP,
)

TRIVIAL_MAP = """
#####
#SG##
#####
"""

UNREACHABLE_MAP = """
#######
#S....#
###.###
#...#G#
#######
"""

# An open room gives the robot genuine alternative routes, unlike the
# original warehouse map, which turns out to be a single corridor
# (64 free cells, all of which both BFS and A* must expand).
OPEN_ROOM_MAP = """
#########
#S......#
#..###..#
#..#.#..#
#..#.#..#
#..###..#
#......G#
#########
"""

def _build_corridor_room_map(width=23, height=21, dist=20):
    """
    A wide-open room with S and G on the same row, far apart horizontally,
    and plenty of free space above and below.

    Because S and G share a row, Manhattan-guided A* can stay almost
    entirely on that row (f stays at the optimum, D, only for cells with
    row-offset 0; any detour up/down costs +2 to f). Blind BFS has no such
    preference: it expands a growing diamond around S in every direction
    and must cover most of the room before it happens to reach G. This is
    where an informed heuristic is expected to clearly reduce the number
    of expanded states, unlike the two maps above.
    """
    mid = height // 2
    rows = ["#" * width]
    for r in range(1, height - 1):
        row = ["#"] + ["."] * (width - 2) + ["#"]
        if r == mid:
            row[1] = "S"
            row[dist + 1] = "G"
        rows.append("".join(row))
    rows.append("#" * width)
    return "\n" + "\n".join(rows) + "\n"


BIG_OPEN_MAP = _build_corridor_room_map()


def task3_tests():
    print("=" * 60)
    print("TASK 3: Test the Generated Program")
    print("=" * 60)

    print("\nTest 1: Original warehouse")
    grid, start, goal = parse_map(WAREHOUSE_MAP)
    print_result("A* on original warehouse", astar(grid, start, goal, manhattan))

    print("Test 2: Trivial case (goal adjacent to start)")
    grid, start, goal = parse_map(TRIVIAL_MAP)
    print_result("A* on trivial map", astar(grid, start, goal, manhattan))

    print("Test 3: No solution (goal unreachable)")
    grid, start, goal = parse_map(UNREACHABLE_MAP)
    print_result("A* on unreachable map", astar(grid, start, goal, manhattan))

    print("Test 4: Alternative paths (open room)")
    grid, start, goal = parse_map(OPEN_ROOM_MAP)
    print_result("A* on open-room map", astar(grid, start, goal, manhattan))
    print_result("BFS on open-room map", bfs(grid, start, goal))


def task5_compare():
    print("=" * 60)
    print("TASK 5: Compare A* with Blind Search")
    print("=" * 60)

    for name, warehouse_map in [("original warehouse", WAREHOUSE_MAP),
                                 ("open room", OPEN_ROOM_MAP),
                                 ("big open room", BIG_OPEN_MAP)]:
        grid, start, goal = parse_map(warehouse_map)
        bfs_res = bfs(grid, start, goal)
        astar_res = astar(grid, start, goal, manhattan)
        print(f"\nMap: {name}")
        print(f"{'Measure':<16}{'BFS':>10}{'A*':>10}")
        print(f"{'Solution found':<16}{str(bfs_res['found']):>10}{str(astar_res['found']):>10}")
        print(f"{'Path length':<16}{len(bfs_res['path']) - 1:>10}{len(astar_res['path']) - 1:>10}")
        print(f"{'States expanded':<16}{bfs_res['expanded']:>10}{astar_res['expanded']:>10}")


def task6_heuristics():
    print("\n" + "=" * 60)
    print("TASK 6: Investigate the Heuristic")
    print("=" * 60)

    variants = [
        ("Manhattan (h = |dx|+|dy|)", manhattan, 1.0),
        ("Zero (h = 0, i.e. plain Dijkstra/uniform-cost)", zero_heuristic, 1.0),
        ("Euclidean distance", euclidean, 1.0),
        ("Manhattan x 2 (inadmissible, overestimates)", manhattan, 2.0),
    ]

    for label, maze_name, maze in [("original warehouse (single corridor)", "orig", WAREHOUSE_MAP),
                                    ("open room (branching)", "open", OPEN_ROOM_MAP),
                                    ("big open room (no obstacles)", "big", BIG_OPEN_MAP)]:
        print(f"\n--- Map: {label} ---")
        grid, start, goal = parse_map(maze)
        for name, h, w in variants:
            res = astar(grid, start, goal, h, weight=w)
            plen = (len(res["path"]) - 1) if res["found"] else None
            print(f"{name:<45} found={res['found']!s:<6} "
                  f"path_len={plen!s:<6} expanded={res['expanded']}")


if __name__ == "__main__":
    task3_tests()
    task5_compare()
    task6_heuristics()
