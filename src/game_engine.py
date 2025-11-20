from enum import Enum
from typing import Optional
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import random

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

class MAFrozenLakeEngine():
    def __init__(
        self,
        seed: int,
        agent_list: list[str],
        grid_size: int,
        agent_positions: dict[str, tuple[int, int]],
        vision_range: int,
        success_rate: Optional[float],
    ):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.possible_agents = agent_list
        self.agent_list = agent_list
        self.GRID_SIZE = grid_size
        self.default_positions = agent_positions
        self.agent_positions = agent_positions
        self.VISION_RANGE = vision_range
        self._grid = MAFrozenLakeEngine.gen_grid(
            seed = seed, 
            grid_size = self.GRID_SIZE,
        )
        self.success_rate = success_rate

    def reset(self):
        self.agent_list = self.possible_agents
        self.agent_positions = self.default_positions

    def grid_size(self) -> int:
        return self.GRID_SIZE

    def set_agent_pos(self, agent_id: str, x: int, y: int):
        self.agent_positions[agent_id] = (x, y)

    @staticmethod
    def gen_grid(seed: int, grid_size: int):
        grid = np.zeros((grid_size, grid_size), dtype=np.int8)
        m = generate_random_map(size=grid_size, seed=seed)
        for y in range(grid_size):
            for x in range(grid_size):
                tile = m[y][x]
                if tile == "H":
                    grid[x, y] = Tile.HOLE.value
                elif tile == "G":
                    grid[x, y] = Tile.GOAL.value
                elif tile == "F":
                    grid[x, y] = Tile.EMPTY.value
                elif tile == "S":
                    grid[x, y] = Tile.START.value

        return grid

    def agent_movement(self, agent_id: str, movement: int) -> tuple[int, int]:
        """ Moves the agent according to the action taken, if valid.
            Returns the agent new position.
        """
        x, y = self.agent_positions[agent_id]
        dx, dy = MovementAction.delta(movement, self.success_rate, self.rng)
        target_x, target_y = x + dx, y + dy

        # Check boundaries
        if (
            0 <= target_x < self.GRID_SIZE and
            0 <= target_y < self.GRID_SIZE
        ):
            self.set_agent_pos(agent_id, target_x, target_y)
        else:
            target_x, target_y = x, y

        return (target_x, target_y)

    def get_local_view(self, agent_id):
        x, y = self.agent_positions[agent_id]
        # pad the grid to handle edge cases
        padded = np.pad(self._grid, pad_width=self.VISION_RANGE, mode='constant', constant_values=Tile.OUT_OF_MAP.value)

        # adjust for padding
        x_p, y_p = x + self.VISION_RANGE, y + self.VISION_RANGE

        # slice local area
        local = padded[x_p - self.VISION_RANGE : x_p + self.VISION_RANGE + 1,
                    y_p - self.VISION_RANGE : y_p + self.VISION_RANGE + 1]
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

            dist = max(abs(dx), abs(dy)) # Chebyshev distance

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