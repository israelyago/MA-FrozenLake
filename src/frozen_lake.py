import functools
from os import path
import numpy as np

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from gymnasium.utils import seeding
from gymnasium.spaces import Dict, Box, Tuple, Discrete, MultiDiscrete
import pygame
import colorsys

from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector, wrappers

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
DO_NOTHING = 4

OUT_OF_MAP, EMPTY, HOLE, GOAL, START = 0, 1, 2, 3, 4

MOVE_DELTA = {
    LEFT: (-1, 0),
    DOWN: (0, 1),
    RIGHT: (1, 0),
    UP: (0, -1),
    DO_NOTHING: (0, 0),
}

ACTION_TO_TEXT = {
    LEFT: "LEFT",
    DOWN: "DOWN",
    RIGHT: "RIGHT",
    UP: "UP",
    DO_NOTHING: "DO_NOTHING",
}

MAPS = {
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}

def env(seed=None, render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the PettingZoo developer documentation.
    """
    env = raw_env(seed=seed, render_mode=render_mode)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class raw_env(AECEnv):
    """
    The metadata holds environment constants. From gymnasium, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        "render_modes": ["human"], 
        "name": "MA_FrozenLake_v0",
        "render_fps": 4,
    }

    def __init__(self,
                seed=None,
                render_mode=None,
                desc: list[str] = None,
                map_name: str = "4x4",
                is_slippery: bool = True,
                success_rate: float = 1.0 / 3.0,
                reward_schedule: tuple[int, int, int] = (1, 0, 0),
        ):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        N_AGENTS = 2
        self.N_MESSAGES = 5
        self.N_ACTIONS = 5
        self.N_OF_TILES_TYPE = 5
        self.GRID_SIZE = 4
        self.VISION_RANGE = 1 # tiles around agent
        self.OBS_SIZE = 2 * self.VISION_RANGE + 1
        self.possible_agents = ["player_" + str(r) for r in range(N_AGENTS)]
        self.reward_schedule = reward_schedule

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self._action_spaces = {
            agent: Tuple((Discrete(self.N_ACTIONS), Discrete(self.N_MESSAGES)), seed=seed + i)
            for i, agent in enumerate(self.possible_agents)
        }
        self._observation_spaces = {
            agent: Dict({
                "grid": Box(low=0, high=1, shape=(self.OBS_SIZE, self.OBS_SIZE, self.N_OF_TILES_TYPE), dtype=np.float32),
                "messages": MultiDiscrete([self.N_MESSAGES] * (len(self.possible_agents) - 1)),
                "relative_positions": Box(
                    low=-1.0, high=1.0,
                    shape=(len(self.possible_agents) - 1, 3), # (dx, dy, visible)
                    dtype=np.float32
                ),
            })
            for agent in self.possible_agents
        }
        self.agent_colored_elfs = {}
        self.render_mode = render_mode

        # Generate game state
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
        m = generate_random_map(size=self.GRID_SIZE, seed=seed)
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                tile = m[y][x]
                if tile == "H":
                    self.grid[x, y] = HOLE
                elif tile == "G":
                    self.grid[x, y] = GOAL
                elif tile == "F":
                    self.grid[x, y] = EMPTY
                elif tile == "S":
                    self.grid[x, y] = START
        
        self.agent_positions = {
            agent: (0, 0)
            for agent in self.possible_agents
        }
        self.agent_messages = {agent: 0 for agent in self.possible_agents}

        desc = np.asarray(m, dtype="c")

        nrow, ncol = nrow, ncol = desc.shape
        self.reward_range = (min(reward_schedule), max(reward_schedule))

        # pygame utils
        # Main grid
        main_width = min(64 * ncol, 512)
        main_height = min(64 * nrow, 512)
        self.cell_size = (main_width // ncol, main_height // nrow)

        # --- Compute mini-grid sizes (half cell size) ---
        mini_tile_w = self.cell_size[0] // 2
        mini_tile_h = self.cell_size[1] // 2
        mini_grid_w = self.OBS_SIZE * mini_tile_w
        mini_grid_h = self.OBS_SIZE * mini_tile_h

        # Mini-grids for agent vision
        spacing = 10
        extra_width = len(self.possible_agents) * (mini_grid_w + spacing)
        total_width = main_width + extra_width
        total_height = max(main_height, mini_grid_h)

        # --- Final window size ---
        self.window_size = (total_width, total_height)

        self.window_surface = None
        self.cell_size = (
            self.window_size[0] // ncol,
            self.window_size[1] // nrow,
        )
        self.window_surface = None
        self.clock = None
        self.hole_img = None
        self.cracked_hole_img = None
        self.ice_img = None
        self.ice_img_dark = None
        self.elf_images = None
        self.goal_img = None
        self.start_img = None

    def get_local_view(self, agent_id):
        x, y = self.agent_positions[agent_id]
        # pad the grid to handle edge cases
        padded = np.pad(self.grid, pad_width=self.VISION_RANGE, mode='constant', constant_values=OUT_OF_MAP)

        # adjust for padding
        x_p, y_p = x + self.VISION_RANGE, y + self.VISION_RANGE

        # slice local area
        local = padded[x_p - self.VISION_RANGE : x_p + self.VISION_RANGE + 1,
                    y_p - self.VISION_RANGE : y_p + self.VISION_RANGE + 1]
        # one-hot encode
        local_one_hot = np.eye(self.N_OF_TILES_TYPE, dtype=np.float32)[local]
        return local_one_hot

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    def render(self):
        """
        Renders the environment.
        """
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="human")'
            )
            return

        if self.render_mode == "ansi":
            return self._render_text()
        else:  # self.render_mode in {"human", "rgb_array"}:
            return self._render_gui(self.render_mode)

    def _render_gui(self, mode):
        ncol, nrow = self.grid.shape

        # --- Compute main grid size ---
        main_width = min(64 * ncol, 512)
        main_height = min(64 * nrow, 512)
        self.cell_size = (main_width // ncol, main_height // nrow)

        # --- Compute mini-grid sizes (half cell size) ---
        mini_tile_w = self.cell_size[0] // 2
        mini_tile_h = self.cell_size[1] // 2
        mini_grid_w = self.OBS_SIZE * mini_tile_w
        mini_grid_h = self.OBS_SIZE * mini_tile_h

        # --- Add space for mini-grids on the right ---
        spacing = 10
        extra_width = len(self.agents) * (mini_grid_w + spacing)
        total_width = main_width + extra_width
        total_height = max(main_height, mini_grid_h)

        # --- Final window size ---
        self.window_size = (total_width, total_height)

        # --- Initialize window ---
        if self.window_surface is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Multi Agent Frozen Lake")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.hole_img is None:
            file_name = path.join(path.dirname(__file__), "img/hole.png")
            self.hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.cracked_hole_img is None:
            file_name = path.join(path.dirname(__file__), "img/cracked_hole.png")
            self.cracked_hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.ice_img is None:
            file_name = path.join(path.dirname(__file__), "img/ice.png")
            self.ice_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.ice_img_dark is None:
            file_name = path.join(path.dirname(__file__), "img/ice.png")
            self.ice_img_dark = self._darken_surface(self.ice_img, factor=0.7)

        if self.goal_img is None:
            file_name = path.join(path.dirname(__file__), "img/goal.png")
            self.goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.start_img is None:
            file_name = path.join(path.dirname(__file__), "img/stool.png")
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.elf_images is None:
            elfs = {
                LEFT: path.join(path.dirname(__file__), "img/elf_left.png"),
                DOWN: path.join(path.dirname(__file__), "img/elf_down.png"),
                RIGHT: path.join(path.dirname(__file__), "img/elf_right.png"),
                UP: path.join(path.dirname(__file__), "img/elf_up.png"),
            }
            self.elf_images = {
                LEFT: pygame.transform.scale(pygame.image.load(elfs[LEFT]), self.cell_size),
                DOWN: pygame.transform.scale(pygame.image.load(elfs[DOWN]), self.cell_size),
                RIGHT: pygame.transform.scale(pygame.image.load(elfs[RIGHT]), self.cell_size),
                UP: pygame.transform.scale(pygame.image.load(elfs[UP]), self.cell_size),
            }

            n_agents = len(self.possible_agents)
            for i, agent_id in enumerate(self.possible_agents):
                hue_offset = i / n_agents # evenly spaced hues
                self.agent_colored_elfs[agent_id] = {
                    k: self._hue_shift(v, hue_offset)
                    for k, v in self.elf_images.items()
                }

        # --- Draw main grid ---
        for agent_y in range(nrow):
            for agent_x in range(ncol):
                pos = (agent_x * self.cell_size[0], agent_y * self.cell_size[1])
                rect = (*pos, *self.cell_size)

                self.window_surface.blit(self.ice_img, pos)
                if self.grid[agent_x, agent_y] == HOLE:
                    self.window_surface.blit(self.hole_img, pos)
                elif self.grid[agent_x, agent_y] == GOAL:
                    self.window_surface.blit(self.goal_img, pos)
                elif self.grid[agent_x, agent_y] == START:
                    self.window_surface.blit(self.start_img, pos)

                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)

        # --- Draw agents on main grid ---
        for agent_id in self.agents:
            bot_x, bot_y = self.agent_positions[agent_id]
            cell_rect = (bot_x * self.cell_size[0], bot_y * self.cell_size[1])

            image_from_last_action = DOWN # Default image
            if self.agent_last_action[agent_id] not in (None, DO_NOTHING):
                image_from_last_action = self.agent_last_action[agent_id]

            elf_img = self.agent_colored_elfs[agent_id][image_from_last_action]

            if self.grid[bot_x, bot_y] == HOLE:
                self.window_surface.blit(self.cracked_hole_img, cell_rect)
            else:
                self.window_surface.blit(elf_img, cell_rect)

        # --- Draw each agent's mini-grid on the right ---
        center_x, center_y = self.OBS_SIZE // 2, self.OBS_SIZE // 2
        for i, agent_id in enumerate(self.agents):
            local_patch = self.get_local_view(agent_id)
            local_int = np.argmax(local_patch, axis=-1)

            # compute top-left of this agent's window
            top_left_x = main_width + spacing + i * (mini_grid_w + spacing)
            top_left_y = 10

            agent_x, agent_y = self.agent_positions[agent_id]

            for row in range(self.OBS_SIZE):
                for col in range(self.OBS_SIZE):
                    tile = local_int[col, row]
                    pos = (top_left_x + col * mini_tile_w, top_left_y + row * mini_tile_h)

                    # --- Always draw ice as background first ---
                    base_img = pygame.transform.scale(self.ice_img, (mini_tile_w, mini_tile_h))
                    self.window_surface.blit(base_img, pos)

                    # --- Then draw tile overlay if any ---
                    if tile == HOLE and (col, row) == (center_x, center_y):
                        img = self.cracked_hole_img
                    elif tile == HOLE:
                        img = self.hole_img
                    elif tile == GOAL:
                        img = self.goal_img
                    elif tile == START:
                        img = self.start_img
                    elif tile == OUT_OF_MAP:
                        img = self.ice_img_dark
                    else:
                        img = None

                    if img is not None:
                        img_scaled = pygame.transform.scale(img, (mini_tile_w, mini_tile_h))
                        self.window_surface.blit(img_scaled, pos)

                    rect = pygame.Rect(pos[0], pos[1], mini_tile_w, mini_tile_h)
                    pygame.draw.rect(self.window_surface, (100, 100, 100), rect, 1)

            # Draw the agent icon in the center of mini-grid
            center_pos = (
                top_left_x + (self.OBS_SIZE // 2) * mini_tile_w,
                top_left_y + (self.OBS_SIZE // 2) * mini_tile_h,
            )
            if self.grid[agent_x, agent_y] != HOLE:
                image_from_last_action = DOWN # Default image
                if self.agent_last_action[agent_id] not in (None, DO_NOTHING):
                    image_from_last_action = self.agent_last_action[agent_id]

                elf_img = self.agent_colored_elfs[agent_id][image_from_last_action]

                elf_img_scaled = pygame.transform.scale(elf_img, (mini_tile_w, mini_tile_h))
                self.window_surface.blit(elf_img_scaled, center_pos)

                # Label each mini-grid
                font = pygame.font.SysFont(None, 20)
                label = font.render(agent_id, True, (255, 255, 255))
                self.window_surface.blit(label, (top_left_x, top_left_y - 15))

        # --- Update display ---
        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )

    def _render_text(self):
        raise NotImplementedError("Text rendering not implemented yet.")

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        # Unlike gymnasium's Env, the environment is responsible for setting the random seed explicitly.
        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        """
        Our AgentSelector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.agent_positions = {
            agent: (0, 0)
            for agent in self.possible_agents
        }
        self.agent_messages = {agent: 0 for agent in self.possible_agents}

        self.agent_last_action = {agent: None for agent in self.agents}

        if self.render_mode == "human":
            self.render()
        return self.observations, self.infos

    def _agent_movement(self, agent_id, action):
        """ Moves the agent according to the action taken, if valid.
            Returns the agent new position.
        """
        x, y = self.agent_positions[agent_id]
        dx, dy = MOVE_DELTA[action]
        target_x, target_y = x + dx, y + dy

        # Check boundaries
        if (
            0 <= target_x < self.GRID_SIZE and
            0 <= target_y < self.GRID_SIZE
        ):
            self.agent_positions[agent_id] = (target_x, target_y)
        else:
            target_x, target_y = x, y

        return (target_x, target_y)

    def _was_dead_step(self, action):
        """
        Handles the case where the agent is already done (truncated or terminated).
        Returns dummy observations/rewards/truncations so the step() call does not break.
        """
        obs = np.zeros_like(self.get_local_view(self.agent_selection))
        reward = 0
        truncation = True
        termination = True
        return obs, reward, termination, truncation

    def get_relative_positions(self, agent_id):
        x_self, y_self = self.agent_positions[agent_id]
        other_agents = [a for a in self.agents if a != agent_id]

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

    def step(self, action_bundle):
        """
        step(action_bundle) takes in an (action, message) for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        agent_id = self.agent_selection
        action, message = action_bundle

        if (
            self.terminations[agent_id]
            or self.truncations[agent_id]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self.observations[agent_id], self.rewards[agent_id], \
            self.terminations[agent_id], self.truncations[agent_id] = \
                self._was_dead_step(action)

            self.agent_messages[agent_id] = 0
            self.agent_selection = self._agent_selector.next()
            return

        self.agent_messages[agent_id] = message

        # Move agent according to action
        new_x, new_y = self._agent_movement(agent_id, action)

        self.rewards = {agent: 0 for agent in self.agents}

        # Handle new tile effects
        tile = self.grid[new_x, new_y]
        if tile == EMPTY or tile == START:
            self.rewards[agent_id] = self.reward_schedule[2]
            self.observations[agent_id] = {
                "grid": self.get_local_view(agent_id),
                "relative_positions": self.get_relative_positions(agent_id),
                "messages": [self.agent_messages[a] for a in self.agents if a != agent_id],
            }
        elif tile == GOAL:
            self.terminations[agent_id] = True
            self.rewards[agent_id] = self.reward_schedule[0]
            self.observations[agent_id] = {
                "grid": self.get_local_view(agent_id),
                "relative_positions": self.get_relative_positions(agent_id),
                "messages": [self.agent_messages[a] for a in self.agents if a != agent_id],
            }
        elif tile == HOLE:
            self.truncations[agent_id] = True
            self.rewards[agent_id] = self.reward_schedule[1]
            self.observations[agent_id] = {
                "grid": np.zeros_like(self.get_local_view(agent_id)),
                "relative_positions": np.zeros_like(self.get_relative_positions(agent_id)),
                "messages": [self.agent_messages[a] for a in self.agents if a != agent_id],
            }

        self.agent_last_action[agent_id] = action

        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def _darken_surface(self, surface, factor=0.5):
        """
        Return a new Surface that's a darker copy of `surface`.
        factor in (0..1] where 1.0 = original brightness, 0.0 = black.
        Preserves the alpha channel.
        """
        surf = surface.convert_alpha()

        # RGB array (shape (w,h,3))
        rgb = pygame.surfarray.array3d(surf).astype(np.float32)
        rgb *= factor
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        # Create a surface from RGB, then reapply alpha
        dark = pygame.surfarray.make_surface(rgb)
        dark = dark.convert_alpha()

        alpha = pygame.surfarray.array_alpha(surf)
        pygame.surfarray.pixels_alpha(dark)[:] = alpha

        return dark

    def _hue_shift(self, surface, shift):
        """Shift hue of a pygame surface (0–1 scale)."""
        surface = surface.convert_alpha()
        arr = pygame.surfarray.pixels3d(surface).copy()
        alpha = pygame.surfarray.pixels_alpha(surface).copy()

        arr = arr.astype(np.float32) / 255.0
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

        h, s, v = np.vectorize(colorsys.rgb_to_hsv)(r, g, b)
        h = (h + shift) % 1.0
        r2, g2, b2 = np.vectorize(colorsys.hsv_to_rgb)(h, s, v)

        arr[..., 0] = r2 * 255
        arr[..., 1] = g2 * 255
        arr[..., 2] = b2 * 255

        shifted = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        pygame.surfarray.blit_array(shifted, arr.astype(np.uint8))
        pygame.surfarray.pixels_alpha(shifted)[:] = alpha

        return shifted

# Elf and stool from https://franuka.itch.io/rpg-snow-tileset
# All other assets by Mel Tillery http://www.cyaneus.com/