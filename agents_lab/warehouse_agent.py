"""
A goal-based agent for the warehouse navigation problem.

Goal-based agent architecture (vs. a simple reflex agent):
  Environment  -> what the agent perceives (the grid, obstacles)
  State        -> the agent's current position
  Goal         -> the destination cell
  Actions      -> Up / Down / Left / Right
  Decision-making component -> a planner that SEARCHES for a sequence of
                                 actions reaching the goal, rather than
                                 reacting to only the immediate percept.

A simple reflex agent would pick an action from a fixed condition-action
rule based only on what's immediately around it (e.g. "if a free cell is
ahead, move forward"), with no notion of a destination and no guarantee
of ever reaching one, and no way to recognise or avoid dead ends. This
agent instead explicitly represents a goal and plans a full path to it
before acting, which is the defining property of a goal-based agent.
"""

from collections import deque

WAREHOUSE_MAP = """\
#####################
#S....#............G#
#.##....##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################
"""

MOVES = {
    "Up": (-1, 0),
    "Down": (1, 0),
    "Left": (0, -1),
    "Right": (0, 1),
}


class Environment:
    """The agent's model of the warehouse: a 2D grid of obstacle/free cells."""

    def __init__(self, ascii_map):
        self.grid = ascii_map.strip("\n").split("\n")
        self.start = self.goal = None
        for r, row in enumerate(self.grid):
            for c, ch in enumerate(row):
                if ch == "S":
                    self.start = (r, c)
                elif ch == "G":
                    self.goal = (r, c)
        if self.start is None or self.goal is None:
            raise ValueError("Map must contain exactly one 'S' and one 'G'")

    def is_free(self, pos):
        r, c = pos
        if not (0 <= r < len(self.grid) and 0 <= c < len(self.grid[r])):
            return False
        return self.grid[r][c] != "#"

    def successors(self, pos):
        """(action, next_state) pairs reachable from pos with one valid move."""
        for action, (dr, dc) in MOVES.items():
            nxt = (pos[0] + dr, pos[1] + dc)
            if self.is_free(nxt):
                yield action, nxt


class GoalBasedAgent:
    """
    Maintains its current state and goal, and uses a decision-making
    (planning) component to decide on a full action sequence before acting.
    """

    def __init__(self, environment):
        self.environment = environment
        self.state = environment.start   # current state, perceived from the environment
        self.goal = environment.goal     # explicit objective

    def formulate_plan(self):
        """
        Decision-making component: breadth-first search over the
        environment's successor relation.

        BFS is the natural, simplest choice here: every move has the same
        cost (1), so the first time BFS reaches a cell is guaranteed to be
        via a shortest path to it - no heuristic is needed (unlike A*),
        and it is simpler to implement correctly than depth-first search,
        which would find *a* path but not necessarily a collision-free
        *shortest* one, and could explore much longer before succeeding
        on a map with dead ends.
        """
        frontier = deque([self.state])
        came_from = {}
        visited = {self.state}
        expanded = 0

        while frontier:
            current = frontier.popleft()
            expanded += 1
            if current == self.goal:
                return self._reconstruct_path(came_from), expanded
            for _action, nxt in self.environment.successors(current):
                if nxt not in visited:
                    visited.add(nxt)
                    came_from[nxt] = current
                    frontier.append(nxt)

        return None, expanded  # no path exists

    def _reconstruct_path(self, came_from):
        path = [self.goal]
        cur = self.goal
        while cur != self.state:
            cur = came_from[cur]
            path.append(cur)
        path.reverse()
        return path

    def act(self):
        """Plan a path to the goal and report it (or report failure)."""
        path, expanded = self.formulate_plan()
        if path is None:
            print(f"No collision-free path exists from {self.state} to {self.goal}.")
            print(f"(States expanded before giving up: {expanded})")
            return None
        print(f"Path found from {self.state} to {self.goal}:")
        print(" -> ".join(str(p) for p in path))
        print(f"Path length: {len(path) - 1} moves")
        print(f"States expanded: {expanded}")
        return path


if __name__ == "__main__":
    env = Environment(WAREHOUSE_MAP)
    agent = GoalBasedAgent(env)
    agent.act()
