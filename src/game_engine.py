import sys
from collections import deque
from enum import Enum
from itertools import product
from typing import Optional

import numpy as np


class MovementAction(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    DO_NOTHING = 4

    def __str__(self):
        return self.name

    @staticmethod
    def delta(movement, success_rate: Optional[float], rng: np.random.Generator):
        # Normalize movement
        if isinstance(movement, np.int64):
            movement = MovementAction(movement.item())
        if isinstance(movement, int):
            movement = MovementAction(movement)

        deltas = {
            MovementAction.LEFT: (-1, 0),
            MovementAction.DOWN: (0, 1),
            MovementAction.RIGHT: (1, 0),
            MovementAction.UP: (0, -1),
            MovementAction.DO_NOTHING: (0, 0),
        }

        if success_rate is None or movement == MovementAction.DO_NOTHING:
            return deltas[movement]

        if rng.random() < success_rate:
            return deltas[movement]

        # Orthogonal slips
        ortho = {
            MovementAction.LEFT: [MovementAction.UP, MovementAction.DOWN],
            MovementAction.RIGHT: [MovementAction.UP, MovementAction.DOWN],
            MovementAction.UP: [MovementAction.LEFT, MovementAction.RIGHT],
            MovementAction.DOWN: [MovementAction.LEFT, MovementAction.RIGHT],
        }

        slipped = rng.choice(ortho[movement])
        return deltas[slipped]


class Tile(Enum):
    OUT_OF_MAP = 0
    EMPTY = 1
    HOLE = 2
    GOAL = 3
    START = 4


# class Position():
#     def __init__(self, x, y):
#         self._pos = (x, y)


class MAFrozenLakeEngine:
    def __init__(
        self,
        seed: int,
        agent_list: list[str],
        grid_size: int,
        vision_range: int,
        success_rate: Optional[float],
    ):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.possible_agents = agent_list
        self.agent_list = agent_list
        self.GRID_SIZE = grid_size
        self.default_positions = {}
        self.VISION_RANGE = vision_range
        self._grid = self.gen_grid(
            grid_size=self.GRID_SIZE, agent_list=self.possible_agents
        )
        self.success_rate = success_rate

    def reset(self):
        self.agent_list = self.possible_agents.copy()
        self.agent_positions = self.default_positions.copy()
        self._grid = self.gen_grid(
            grid_size=self.GRID_SIZE, agent_list=self.possible_agents
        )

    def grid_size(self) -> int:
        return self.GRID_SIZE

    def set_agent_pos(self, agent_id: str, x: int, y: int):
        self.agent_positions[agent_id] = (x, y)

    def gen_grid(
        self,
        grid_size: int,
        agent_list: list[str],
    ):
        agents = agent_list.copy()
        grid = np.zeros((grid_size, grid_size), dtype=np.int8)
        m = self.generate_random_map(size=grid_size, how_many_agents=len(agents))
        for x, y in product(range(grid_size), range(grid_size)):
            row = y
            col = x
            tile = m[row][col]
            if tile == "H":
                grid[x, y] = Tile.HOLE.value
            elif tile == "G":
                grid[x, y] = Tile.GOAL.value
            elif tile == "F":
                grid[x, y] = Tile.EMPTY.value
            elif tile == "S":
                grid[x, y] = Tile.START.value
                if len(agents) == 0:
                    print(
                        "🚨 BUG: There are more starting positions than agents. "
                        "This should not happen. "
                        "Please, open a PR if that occurs"
                    )
                    sys.exit(1)

                agent = agents.pop()
                self.default_positions[agent] = (x, y)

        self.agent_positions = self.default_positions.copy()
        return grid

    @staticmethod
    def is_valid(
        board: list[list[str]], size: int, require_all_starts_reach_goal: bool = True
    ) -> bool:
        """
        Checks whether the generated board is valid.

        A board is valid if:
        - It has at least one 'G' tile.
        - It has one or more 'S' tiles.
        - At least one S can reach G (default), OR all S can reach G (optional).

        Args:
            board: a 2D list of strings (e.g. [['F','H',...], ...])
            size: width/height of the grid
            require_all_starts_reach_goal: if True, every S must reach G

        Returns:
            True if the map is valid.
        """

        # -----------------------------
        # 1. Gather all S and G coordinates
        # -----------------------------
        start_positions = []
        goal_positions = []

        for r in range(size):
            for c in range(size):
                if board[r][c] == "S":
                    start_positions.append((r, c))
                elif board[r][c] == "G":
                    goal_positions.append((r, c))

        if not start_positions:
            return False
        if not goal_positions:
            return False

        # NOTE: You are generating exactly 1 G. If multiple G allowed, this still works.
        goal = goal_positions[0]

        # -----------------------------
        # 2. BFS helper
        # -----------------------------
        def bfs(start_r: int, start_c: int) -> bool:
            """Return True if (start_r, start_c) can reach the goal via non-hole tiles."""
            queue = deque([(start_r, start_c)])
            visited = set([(start_r, start_c)])

            while queue:
                r, c = queue.popleft()

                # Reached goal?
                if (r, c) == goal:
                    return True

                # Explore neighbors (no diagonals)
                for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < size
                        and 0 <= nc < size
                        and (nr, nc) not in visited
                        and board[nr][nc] != "H"  # cannot pass through holes
                    ):
                        visited.add((nr, nc))
                        queue.append((nr, nc))

            return False

        # -----------------------------
        # 3. Check depending on requirement
        # -----------------------------
        if require_all_starts_reach_goal:
            # Every agent must be able to reach the goal
            return all(bfs(r, c) for (r, c) in start_positions)
        else:
            # Only one S is required to have a valid path
            return any(bfs(r, c) for (r, c) in start_positions)

    def generate_random_map(self, how_many_agents: int, size: int = 8, p: float = 0.8):
        """Generates a random valid map with multiple agent start tiles.

        Args:
            how_many_agents: number of starting tiles "S"
            size: map size (size x size)
            p: probability that a tile is frozen ("F")

        Returns:
            A random valid map as list[str]
        """

        assert how_many_agents >= 1, "Need at least one agent"
        assert how_many_agents < size * size, "Too many agents for the map size"

        valid = False
        board = []

        while not valid:
            # Generate base board of F/H
            board = self.rng.choice(["F", "H"], (size, size), p=[p, 1 - p]).astype(
                "<U1"
            )

            # ----------------------
            # 1. Pick starting positions
            # ----------------------
            # Pick how_many_agents distinct positions from the entire grid
            flat_indices = self.rng.choice(size * size, how_many_agents, replace=False)
            start_positions = [(idx // size, idx % size) for idx in flat_indices]

            # ----------------------
            # 2. Pick a goal position
            # ----------------------
            # Must NOT be one of the start positions
            all_positions = set((r, c) for r in range(size) for c in range(size))
            remaining_positions = list(all_positions - set(start_positions))

            goal_idx = self.rng.choice(len(remaining_positions))
            goal_pos = remaining_positions[goal_idx]

            # ----------------------
            # 3. Stamp S/G tiles (force them to be F)
            # ----------------------
            for r, c in start_positions:
                board[r][c] = "S"

            gr, gc = goal_pos
            board[gr][gc] = "G"

            # ----------------------
            # 4. Validate path existence
            # ----------------------
            valid = MAFrozenLakeEngine.is_valid(
                board, size, require_all_starts_reach_goal=True
            )

        return ["".join(row) for row in board]

    def agent_movement(self, agent_id: str, movement: int) -> tuple[int, int]:
        """Moves the agent according to the action taken, if valid.
        Returns the agent new position.
        """
        x, y = self.agent_positions[agent_id]
        dx, dy = MovementAction.delta(movement, self.success_rate, self.rng)
        target_x, target_y = x + dx, y + dy

        # Check boundaries
        if 0 <= target_x < self.GRID_SIZE and 0 <= target_y < self.GRID_SIZE:
            self.set_agent_pos(agent_id, target_x, target_y)
        else:
            target_x, target_y = x, y

        return (target_x, target_y)

    def get_local_view(self, agent_id):
        x, y = self.agent_positions[agent_id]
        # pad the grid to handle edge cases
        padded = np.pad(
            self._grid,
            pad_width=self.VISION_RANGE,
            mode="constant",
            constant_values=Tile.OUT_OF_MAP.value,
        )

        # adjust for padding
        x_p, y_p = x + self.VISION_RANGE, y + self.VISION_RANGE

        # slice local area
        local = padded[
            x_p - self.VISION_RANGE : x_p + self.VISION_RANGE + 1,
            y_p - self.VISION_RANGE : y_p + self.VISION_RANGE + 1,
        ]
        # one-hot encode
        local_one_hot = np.eye(len(Tile), dtype=np.float32)[local]
        return local_one_hot

    def get_relative_positions(self, agent_id):
        x_self, y_self = self.agent_positions[agent_id]
        other_agents = [a for a in self.possible_agents if a != agent_id]

        rel_positions = []
        for other in other_agents:
            x_o, y_o = self.agent_positions[other]
            dx = x_o - x_self
            dy = y_o - y_self

            dist = max(abs(dx), abs(dy))  # Chebyshev distance

            if dist <= self.VISION_RANGE:
                dx_norm = np.clip(dx / (self.GRID_SIZE - 1), -1.0, 1.0)
                dy_norm = np.clip(dy / (self.GRID_SIZE - 1), -1.0, 1.0)
                rel_positions.append([dx_norm, dy_norm, 1.0])
            else:
                # Mask unseen agents
                rel_positions.append([0.0, 0.0, 0.0])

        return np.array(rel_positions, dtype=np.float32)

    def tile_at(self, x, y):
        return Tile(self._grid[x, y])
