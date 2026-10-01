"""
Warehouse robot navigation - BFS and A* search agents.

State        : (row, col) grid position of the robot
Actions      : Up, Down, Left, Right - each moves one cell, cost 1
Transition   : moving onto a free cell ('.' or 'S' or 'G'); '#' cells are blocked
Goal test    : current position == goal position
Heuristics   : manhattan (default), zero, euclidean, or any heuristic scaled by a factor
"""

import math
import heapq
from collections import deque

MOVES = {
    "Up": (-1, 0),
    "Down": (1, 0),
    "Left": (0, -1),
    "Right": (0, 1),
}


def parse_map(ascii_map):
    """Turn an ASCII warehouse map into a grid of rows plus start/goal positions."""
    grid = [line for line in ascii_map.strip("\n").split("\n")]
    start = goal = None
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == "S":
                start = (r, c)
            elif ch == "G":
                goal = (r, c)
    if start is None or goal is None:
        raise ValueError("Map must contain exactly one 'S' and one 'G'")
    return grid, start, goal


def in_bounds(grid, pos):
    r, c = pos
    return 0 <= r < len(grid) and 0 <= c < len(grid[r])


def is_free(grid, pos):
    r, c = pos
    return grid[r][c] != "#"


def neighbors(grid, pos):
    """Yield (action_name, next_state) pairs that are valid from pos."""
    for action, (dr, dc) in MOVES.items():
        nxt = (pos[0] + dr, pos[1] + dc)
        if in_bounds(grid, nxt) and is_free(grid, nxt):
            yield action, nxt


def reconstruct_path(came_from, start, goal):
    if goal not in came_from and goal != start:
        return None
    path = [goal]
    cur = goal
    while cur != start:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path


def bfs(grid, start, goal):
    """Blind breadth-first search. Returns a result dict."""
    frontier = deque([start])
    came_from = {}
    visited = {start}
    expanded = 0

    while frontier:
        current = frontier.popleft()
        expanded += 1
        if current == goal:
            return {
                "found": True,
                "path": reconstruct_path(came_from, start, goal),
                "expanded": expanded,
            }
        for _action, nxt in neighbors(grid, current):
            if nxt not in visited:
                visited.add(nxt)
                came_from[nxt] = current
                frontier.append(nxt)

    return {"found": False, "path": None, "expanded": expanded}


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def zero_heuristic(a, b):
    return 0


def astar(grid, start, goal, heuristic=manhattan, weight=1.0):
    """
    A* search. f(n) = g(n) + weight * heuristic(n, goal).
    Returns a result dict with found/path/expanded/path_length.
    """
    counter = 0  # tie-breaker so the heap never compares states directly
    frontier = [(weight * heuristic(start, goal), counter, start)]
    came_from = {}
    g_score = {start: 0}
    closed = set()
    expanded = 0

    while frontier:
        _f, _, current = heapq.heappop(frontier)
        if current in closed:
            continue
        closed.add(current)
        expanded += 1

        if current == goal:
            return {
                "found": True,
                "path": reconstruct_path(came_from, start, goal),
                "expanded": expanded,
            }

        for _action, nxt in neighbors(grid, current):
            tentative_g = g_score[current] + 1
            if nxt in closed and tentative_g >= g_score.get(nxt, math.inf):
                continue
            if tentative_g < g_score.get(nxt, math.inf):
                g_score[nxt] = tentative_g
                came_from[nxt] = current
                f = tentative_g + weight * heuristic(nxt, goal)
                counter += 1
                heapq.heappush(frontier, (f, counter, nxt))

    return {"found": False, "path": None, "expanded": expanded}


def print_result(name, result):
    print(f"--- {name} ---")
    print("Solution found :", result["found"])
    if result["found"]:
        print("Path length    :", len(result["path"]) - 1, "moves")
        print("Path           :", result["path"])
    print("States expanded:", result["expanded"])
    print()


WAREHOUSE_MAP = """
#################
#S....#.........#
#.###.#.#######.#
#...#.#.......#.#
###.#.#######.#.#
#...#.........#.#
#.###########.#.#
#.............#G#
#################
"""


if __name__ == "__main__":
    grid, start, goal = parse_map(WAREHOUSE_MAP)

    print_result("BFS", bfs(grid, start, goal))
    print_result("A* (Manhattan)", astar(grid, start, goal, manhattan))
