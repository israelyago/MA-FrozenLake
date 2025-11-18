import colorsys
import functools
from os import path

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv
from pettingzoo.utils import AgentSelector

from game_engine import MAFrozenLakeEngine, MovementAction, Tile

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

def env(seed=None, render_mode=None, flatten_observations=False):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the PettingZoo developer documentation.
    """
    env = raw_env(seed=seed, render_mode=render_mode, flatten_observations=flatten_observations)
    # this wrapper helps error handling for discrete action spaces
    # env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    # env = wrappers.OrderEnforcingWrapper(env)
    return env

class raw_env(ParallelEnv):
    """
    The metadata holds environment constants. From gymnasium, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        "render_modes": ["human"], 
        "name": "MA_FrozenLake_v0",
        "render_fps": 6,
        "is_parallelizable": True,
    }

    def __init__(self,
                seed=42,
                render_mode=None,
                desc: list[str] = None,
                map_name: str = "4x4",
                is_slippery: bool = True,
                success_rate: float = 1.0 / 3.0,
                reward_schedule: tuple[int, int, int, int] = (1, 0, -1, -0.01),
                flatten_observations=False,
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
        self.VISION_RANGE = 2 # tiles around agent
        self.OBS_SIZE = 2 * self.VISION_RANGE + 1
        self.possible_agents = ["player_" + str(r) for r in range(N_AGENTS)]
        self.reward_schedule = reward_schedule
        self.flatten_observations = flatten_observations
        self.max_steps = 50
        self.current_step = 0

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
        if self.flatten_observations:
            for agent in self.possible_agents:
                flat_size = (
                    np.prod(self._observation_spaces[agent]["grid"].shape)
                    + np.prod(self._observation_spaces[agent]["relative_positions"].shape)
                    + len(self._observation_spaces[agent]["messages"].nvec)
                )
                self._observation_spaces[agent] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(int(flat_size),), dtype=np.float32)
        
        self.agent_colored_elfs = {}
        self.render_mode = render_mode

        # Generate game state
        # self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
        # m = generate_random_map(size=self.GRID_SIZE, seed=seed)
        # for y in range(self.GRID_SIZE):
        #     for x in range(self.GRID_SIZE):
        #         tile = m[y][x]
        #         if tile == "H":
        #             self.grid[x, y] = HOLE
        #         elif tile == "G":
        #             self.grid[x, y] = GOAL
        #         elif tile == "F":
        #             self.grid[x, y] = EMPTY
        #         elif tile == "S":
        #             self.grid[x, y] = START
        
        self.agent_messages = {agent: 0 for agent in self.possible_agents}

        self.reward_range = (min(reward_schedule), max(reward_schedule))

        agent_positions = {
            agent: (0, 0)
            for agent in self.possible_agents
        }
        self.game_engine = MAFrozenLakeEngine(
            seed = seed,
            agent_list = self.possible_agents,
            grid_size = self.GRID_SIZE,
            agent_positions = agent_positions,
            vision_range = self.VISION_RANGE,
        )

        nrow, ncol = self.GRID_SIZE, self.GRID_SIZE

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
        ncol = self.game_engine.grid_size()
        nrow = ncol

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
                MovementAction.LEFT: path.join(path.dirname(__file__), "img/elf_left.png"),
                MovementAction.DOWN: path.join(path.dirname(__file__), "img/elf_down.png"),
                MovementAction.RIGHT: path.join(path.dirname(__file__), "img/elf_right.png"),
                MovementAction.UP: path.join(path.dirname(__file__), "img/elf_up.png"),
            }
            self.elf_images = {
                MovementAction.LEFT: pygame.transform.scale(pygame.image.load(elfs[MovementAction.LEFT]), self.cell_size),
                MovementAction.DOWN: pygame.transform.scale(pygame.image.load(elfs[MovementAction.DOWN]), self.cell_size),
                MovementAction.RIGHT: pygame.transform.scale(pygame.image.load(elfs[MovementAction.RIGHT]), self.cell_size),
                MovementAction.UP: pygame.transform.scale(pygame.image.load(elfs[MovementAction.UP]), self.cell_size),
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
                if self.game_engine.tile_at(agent_x, agent_y) == Tile.HOLE:
                    self.window_surface.blit(self.hole_img, pos)
                elif self.game_engine.tile_at(agent_x, agent_y) == Tile.GOAL:
                    self.window_surface.blit(self.goal_img, pos)
                elif self.game_engine.tile_at(agent_x, agent_y) == Tile.START:
                    self.window_surface.blit(self.start_img, pos)

                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)

        # --- Draw agents on main grid ---
        for agent_id in self.agents:
            bot_x, bot_y = self.game_engine.agent_positions[agent_id]
            cell_rect = (bot_x * self.cell_size[0], bot_y * self.cell_size[1])

            elf_img = self._get_elf_image(agent_id)

            if self.game_engine.tile_at(bot_x, bot_y) == Tile.HOLE:
                self.window_surface.blit(self.cracked_hole_img, cell_rect)
            else:
                self.window_surface.blit(elf_img, cell_rect)

        # --- Draw each agent's mini-grid on the right ---
        center_x, center_y = self.OBS_SIZE // 2, self.OBS_SIZE // 2
        for i, agent_id in enumerate(self.possible_agents):
            local_patch = self.game_engine.get_local_view(agent_id)
            local_int = np.argmax(local_patch, axis=-1)

            # compute top-left of this agent's window
            top_left_x = main_width + spacing + i * (mini_grid_w + spacing)
            top_left_y = 10

            agent_x, agent_y = self.game_engine.agent_positions[agent_id]

            for row in range(self.OBS_SIZE):
                for col in range(self.OBS_SIZE):
                    tile = Tile(local_int[col, row])
                    pos = (top_left_x + col * mini_tile_w, top_left_y + row * mini_tile_h)

                    # --- Always draw ice as background first ---
                    base_img = pygame.transform.scale(self.ice_img, (mini_tile_w, mini_tile_h))
                    self.window_surface.blit(base_img, pos)

                    # --- Then draw tile overlay if any ---
                    if tile == Tile.HOLE and (col, row) == (center_x, center_y):
                        img = self.cracked_hole_img
                    elif tile == Tile.HOLE:
                        img = self.hole_img
                    elif tile == Tile.GOAL:
                        img = self.goal_img
                    elif tile == Tile.START:
                        img = self.start_img
                    elif tile == Tile.OUT_OF_MAP:
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
            if self.game_engine.tile_at(agent_x, agent_y) != Tile.HOLE:
                elf_img = self._get_elf_image(agent_id)

                elf_img_scaled = pygame.transform.scale(elf_img, (mini_tile_w, mini_tile_h))
                self.window_surface.blit(elf_img_scaled, center_pos)

                # Label each mini-grid
                font = pygame.font.SysFont(None, 20)
                label = font.render(agent_id, True, (255, 255, 255))
                self.window_surface.blit(label, (top_left_x, top_left_y - 15))

                # Draw subtle border around self agent
                border_rect = pygame.Rect(center_pos[0], center_pos[1], mini_tile_w, mini_tile_h)
                pygame.draw.rect(self.window_surface, (255, 0, 0), border_rect, 1)  # red border, width=1

            # --- Draw other agents visible in this mini-grid ---
            for other_id in self.possible_agents:
                if other_id == agent_id:
                    continue  # skip self

                other_x, other_y = self.game_engine.agent_positions[other_id]
                # compute relative coords in local patch
                rel_x = other_x - agent_x + center_x
                rel_y = other_y - agent_y + center_y

                if 0 <= rel_x < self.OBS_SIZE and 0 <= rel_y < self.OBS_SIZE:
                    pos = (
                        top_left_x + rel_x * mini_tile_w,
                        top_left_y + rel_y * mini_tile_h,
                    )
                    # Use other agent's last action for sprite direction
                    elf_img = self._get_elf_image(other_id)

                    elf_img_scaled = pygame.transform.scale(elf_img, (mini_tile_w, mini_tile_h))
                    self.window_surface.blit(elf_img_scaled, pos)

        # --- Update display ---
        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )

    def _get_elf_image(self, agent_id):
        elf_direction = MovementAction.DOWN \
            if self.agent_last_action[agent_id] in (None, MovementAction.DO_NOTHING) \
            else self.agent_last_action[agent_id]
        return self.agent_colored_elfs[agent_id][elf_direction]

    def _render_text(self):
        raise NotImplementedError("Text rendering not implemented yet.")

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        grid = self.game_engine.get_local_view(agent)
        relative_positions = self.game_engine.get_relative_positions(agent)
        messages = [self.agent_messages[a] for a in self.possible_agents if a != agent]
        obs = {
            "grid": grid,
            "relative_positions": relative_positions,
            "messages": messages,
        }
        if self.flatten_observations:
            grid_flat = obs["grid"].flatten()
            rel_flat = obs["relative_positions"].flatten()
            msg_flat = np.array(obs["messages"], dtype=np.float32).flatten()
            obs = np.concatenate([grid_flat, rel_flat, msg_flat]).astype(np.float32)
        return obs

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        if self.window_surface is not None:
            try:
                pygame.display.quit()
            except pygame.error:
                pass  # display might already be uninitialized
            self.window_surface = None

        # Stop the clock
        if self.clock is not None:
            self.clock = None

        # Clear loaded assets (optional, helps with repeated reloads)
        self.hole_img = None
        self.cracked_hole_img = None
        self.ice_img = None
        self.ice_img_dark = None
        self.goal_img = None
        self.start_img = None
        self.elf_images = None
        self.agent_colored_elfs = {}

        # Quit pygame if no other windows are using it
        if pygame.get_init():
            pygame.quit()

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
        self.rewards = {agent: 0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        self.terminateds = {agent: False for agent in self.possible_agents}
        self.truncateds = {agent: False for agent in self.possible_agents}
        self.terminateds['__all__'] = False
        self.truncateds['__all__'] = False
        self.infos = {agent: {"dummy": {"yes": 1}} for agent in self.possible_agents}
        self.state = {agent: None for agent in self.possible_agents}
        self.observations = {agent: None for agent in self.possible_agents}
        self._was_done_last_step = {agent: False for agent in self.agents}
        """
        Our AgentSelector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.agent_messages = {agent: 0 for agent in self.possible_agents}
        self.agent_last_action = {agent: None for agent in self.possible_agents}
        self.current_step = 0
        self.game_engine.reset()

        for agent_id in self.possible_agents:
            self.observations[agent_id] = self.observe(agent_id)

        if self.render_mode == "human":
            self.render()

        return self.observations, self.infos

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
        self.current_step += 1
        if self.current_step >= self.max_steps:
            for a in self.possible_agents:
                self.truncateds[a] = True
                self.rewards[a] = self.reward_schedule[2]
            self.truncateds["__all__"] = True

        assert isinstance(action_bundle, dict), (
            f"Action bundle should be a dict, got ({type(action_bundle)}): {action_bundle}"
        )
        for agent_id in action_bundle:
            action, message = action_bundle[agent_id]
            if isinstance(action, np.int32):
                action = action.item()
            if isinstance(message, np.int32):
                message = message.item()

            new_x, new_y = self.game_engine.agent_movement(agent_id, action)
            reward = 0
            grid = self.game_engine.get_local_view(agent_id)
            relative_positions = self.game_engine.get_relative_positions(agent_id)

            # Handle new tile effects
            tile = self.game_engine.tile_at(new_x, new_y)
            if tile == Tile.EMPTY or tile == Tile.START:
                reward = self.reward_schedule[3]
            elif tile == Tile.GOAL:
                reward = self.reward_schedule[1]
            elif tile == Tile.HOLE:
                self.truncateds[agent_id] = True
                reward = self.reward_schedule[2]
                grid = np.zeros_like(grid)
                relative_positions = np.zeros_like(relative_positions)

            self.agent_messages[agent_id] = message
            self.rewards[agent_id] = reward
            self.agent_last_action[agent_id] = MovementAction(action)

        if self.render_mode == "human":
            self.render()

        just_done = {
            agent
            for agent in self.agents
            if (self.terminateds[agent] or self.truncateds[agent])
            and not self._was_done_last_step[agent]
        }

        obs = {}

        for agent in self.agents:
            if not (self.terminateds[agent] or self.truncateds[agent]):
                # Normal active agent
                obs[agent] = self.observe(agent)
            elif agent in just_done:
                # One final observation for done agents
                final_obs = np.zeros_like(self.observe(agent))
                obs[agent] = final_obs

        # Mark done agents so they won’t receive future obs
        for agent in just_done:
            self._was_done_last_step[agent] = True
        rewards = {agent: self.rewards[agent] for agent in obs}
        dones = {agent: self.terminateds[agent] for agent in obs}
        truncs = {agent: self.truncateds[agent] for agent in obs}
        dones["__all__"] = all(self.terminateds.values())
        return obs, rewards, dones, truncs, {}

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