import gymnasium
import pygame
from gymnasium.utils import seeding

from gym_multigrid.actions import Actions
from gym_multigrid.rendering import *
from gym_multigrid.window import Window
import numpy as np

from gym_multigrid.world_objects import Grid, COLOR_NAMES, TILE_PIXELS, World


class MultiGridEnv(gymnasium.Env):
    """
    2D grid world game environment
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    # Enumeration of possible actions

    def __init__(
        self,
        grid_size=None,
        width=None,
        height=None,
        max_steps=100,
        see_through_walls=False,
        agents=None,
        partial_obs=True,
        agent_view_size=7,
        actions_set=Actions,
        objects_set=None,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
        render_mode=None,
        screen_size: int = 480,
    ):
        self.agents = agents

        # Does the agents have partial or full observation?
        self.partial_obs = partial_obs

        # Can't set both grid_size and width/height
        if grid_size:
            assert width == None and height == None
            width = grid_size
            height = grid_size

        # Action enumeration for this environment
        self.actions = actions_set

        # Actions are discrete integer values
        self.action_space = gymnasium.spaces.Discrete(len(self.actions.available))

        self.objects = objects_set or World

        if partial_obs:
            self.observation_space = gymnasium.spaces.Box(
                low=0,
                high=255,
                shape=(agent_view_size, agent_view_size, self.objects.encode_dim),
                dtype="uint8",
            )

        else:
            self.observation_space = gymnasium.spaces.Box(
                low=0,
                high=255,
                shape=(width, height, self.objects.encode_dim),
                dtype="uint8",
            )

        self.ob_dim = np.prod(self.observation_space.shape)
        self.ac_dim = self.action_space.n

        # Range of possible rewards
        self.reward_range = (0, 1)

        # Window to use for human rendering mode
        self.window = None

        # Environment configuration
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.see_through_walls = see_through_walls

        # Initialize the state
        self.grid = None
        self.agent_pos = None
        self.reset()

        # rendering
        self.render_mode = render_mode
        self.screen_size = screen_size
        self.screen_width = self.screen_size
        self.screen_height = self.screen_size
        self.render_size = None
        self.highlight = highlight
        self.tile_size = tile_size
        self.agent_pov = agent_pov
        self.agent_view_size = agent_view_size
        self.clock = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed=seed)

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        for a in self.agents:
            assert a.pos is not None
            assert a.dir is not None

        # Item picked up, being carried, initially nothing
        for a in self.agents:
            a.carrying = None

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        if self.partial_obs:
            obs = self.gen_obs()
        else:
            obs = [
                self.grid.encode_for_agents(
                    world=self.world, agent_pos=self.agents[i].pos
                )
                for i in range(len(self.agents))
            ]
        obs = [self.objects.normalize_obs * ob for ob in obs]
        info = {}
        return obs, info

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    @property
    def steps_remaining(self):
        return self.max_steps - self.step_count

    def _gen_grid(self, width, height):
        assert False, "_gen_grid needs to be implemented by each environment"

    def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
        pass

    def _handle_build(self, i, rewards, fwd_pos, fwd_cell):
        pass

    def _handle_drop(self, i, rewards, fwd_pos, fwd_cell):
        pass

    def _handle_special_moves(self, i, rewards, fwd_pos, fwd_cell):
        pass

    def _handle_switch(self, i, rewards, fwd_pos, fwd_cell):
        pass

    def _reward(self, current_agent, rewards, reward=1):
        """
        Compute the reward to be given upon success
        """

        return 1 - 0.9 * (self.step_count / self.max_steps)

    def _rand_int(self, low, high):
        """
        Generate random integer in [low,high[
        """

        return self.np_random.integers(low, high)

    def _rand_float(self, low, high):
        """
        Generate random float in [low,high[
        """

        return self.np_random.uniform(low, high)

    def _rand_bool(self):
        """
        Generate random boolean value
        """

        return self.np_random.integers(0, 2) == 0

    def _rand_elem(self, iterable):
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_subset(self, iterable, num_elems):
        """
        Sample a random subset of distinct elements of a list
        """

        lst = list(iterable)
        assert num_elems <= len(lst)

        out = []

        while len(out) < num_elems:
            elem = self._rand_elem(lst)
            lst.remove(elem)
            out.append(elem)

        return out

    def _rand_color(self):
        """
        Generate a random color name (string)
        """

        return self._rand_elem(COLOR_NAMES)

    def _rand_pos(self, xLow, xHigh, yLow, yHigh):
        """
        Generate a random (x,y) position tuple
        """

        return (
            self.np_random.integers(xLow, xHigh),
            self.np_random.integers(yLow, yHigh),
        )

    def place_obj(self, obj, top=None, size=None, reject_fn=None, max_tries=math.inf):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError("rejection sampling failed in place_obj")

            num_tries += 1

            pos = np.array(
                (
                    self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                    self._rand_int(top[1], min(top[1] + size[1], self.grid.height)),
                )
            )

            # Don't place the object on top of another object
            if self.grid.get(*pos) != None:
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def put_obj(self, obj, i, j):
        """
        Put an object at a specific position in the grid
        """

        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)

    def place_agent(
        self, agent, top=None, size=None, rand_dir=True, max_tries=math.inf
    ):
        """
        Set the agent's starting point at an empty position in the grid
        """

        agent.pos = None
        pos = self.place_obj(agent, top, size, max_tries=max_tries)
        agent.pos = pos
        agent.init_pos = pos

        if rand_dir:
            agent.dir = self._rand_int(0, 4)

        agent.init_dir = agent.dir

        return pos

    def agent_sees(self, a, x, y):
        """
        Check if a non-empty grid position is visible to the agent
        """

        coordinates = a.relative_coords(x, y)
        if coordinates is None:
            return False
        vx, vy = coordinates

        obs = self.gen_obs()
        obs_grid, _ = Grid.decode(obs["image"])
        obs_cell = obs_grid.get(vx, vy)
        world_cell = self.grid.get(x, y)

        return obs_cell is not None and obs_cell.type == world_cell.type

    def step(self, actions):
        self.step_count += 1

        order = np.random.permutation(len(actions))

        rewards = np.zeros(len(actions))
        done = False

        for i in order:
            if (
                self.agents[i].terminated
                or self.agents[i].paused
                or not self.agents[i].started
                or actions[i] == self.actions.still
            ):
                continue

            # Get the position in front of the agent
            fwd_pos = self.agents[i].front_pos

            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)

            # Rotate left
            if actions[i] == self.actions.left:
                self.agents[i].dir -= 1
                if self.agents[i].dir < 0:
                    self.agents[i].dir += 4

            # Rotate right
            elif actions[i] == self.actions.right:
                self.agents[i].dir = (self.agents[i].dir + 1) % 4

            # Move forward
            elif actions[i] == self.actions.forward:
                if fwd_cell is not None:
                    if fwd_cell.type == "goal":
                        done = True
                        rewards[i] += self._reward(i, rewards, 1)
                    elif fwd_cell.type == "switch":
                        self._handle_switch(i, rewards, fwd_pos, fwd_cell)
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.grid.set(*self.agents[i].pos, None)
                    self.agents[i].pos = fwd_pos
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)

            elif "build" in self.actions.available and actions[i] == self.actions.build:
                self._handle_build(i, rewards, fwd_pos, fwd_cell)

            # Pick up an object
            elif actions[i] == self.actions.pickup:
                self._handle_pickup(i, rewards, fwd_pos, fwd_cell)

            # Drop an object
            elif actions[i] == self.actions.drop:
                self._handle_drop(i, rewards, fwd_pos, fwd_cell)

            # Toggle/activate an object
            elif actions[i] == self.actions.toggle:
                if fwd_cell:
                    fwd_cell.toggle(self, fwd_pos)

            # Done action (not used by default)
            elif actions[i] == self.actions.done:
                pass

            else:
                assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        if self.partial_obs:
            obs = self.gen_obs()
        else:
            obs = [
                self.grid.encode_for_agents(
                    world=self.world, agent_pos=self.agents[i].pos
                )
                for i in range(len(actions))
            ]

        obs = [self.objects.normalize_obs * ob for ob in obs]

        return obs, rewards, done, False, {}

    def gen_obs_grid(self):
        """
        Generate the sub-grid observed by the agents.
        This method also outputs a visibility mask telling us which grid
        cells the agents can actually see.
        """

        grids = []
        vis_masks = []

        for a in self.agents:
            topX, topY, botX, botY = a.get_view_exts()

            grid = self.grid.slice(self.objects, topX, topY, a.view_size, a.view_size)

            for i in range(a.dir + 1):
                grid = grid.rotate_left()

            # Process occluders and visibility
            # Note that this incurs some performance cost
            if not self.see_through_walls:
                vis_mask = grid.process_vis(
                    agent_pos=(a.view_size // 2, a.view_size - 1)
                )
            else:
                vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)

            grids.append(grid)
            vis_masks.append(vis_mask)

        return grids, vis_masks

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        grids, vis_masks = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        obs = [
            grid.encode_for_agents(
                self.objects, [grid.width // 2, grid.height - 1], vis_mask
            )
            for grid, vis_mask in zip(grids, vis_masks)
        ]

        return obs

    def get_obs_render(self, obs, tile_size=TILE_PIXELS // 2):
        """
        Render an agent observation for visualization
        """

        grid, vis_mask = Grid.decode(obs)

        # Render the whole grid
        img = grid.render(self.objects, tile_size, highlight_mask=vis_mask)

        return img

    def _render(self, close=False, highlight=False, tile_size=TILE_PIXELS):
        """
        Render the whole-grid human view
        """

        if close:
            if self.window:
                self.window.close()
            return

        if self.render_mode == "human" and not self.window:
            self.window = Window("gym_multigrid")
            self.window.show(block=False)

        if highlight:
            # Compute which cells are visible to the agent
            _, vis_masks = self.gen_obs_grid()

            highlight_masks = {
                (i, j): [] for i in range(self.width) for j in range(self.height)
            }

            for i, a in enumerate(self.agents):
                # Compute the world coordinates of the bottom-left corner
                # of the agent's view area
                f_vec = a.dir_vec
                r_vec = a.right_vec
                top_left = (
                    a.pos + f_vec * (a.view_size - 1) - r_vec * (a.view_size // 2)
                )

                # Mask of which cells to highlight

                # For each cell in the visibility mask
                for vis_j in range(0, a.view_size):
                    for vis_i in range(0, a.view_size):
                        # If this cell is not visible, don't highlight it
                        if not vis_masks[i][vis_i, vis_j]:
                            continue

                        # Compute the world coordinates of this cell
                        abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                        if abs_i < 0 or abs_i >= self.width:
                            continue
                        if abs_j < 0 or abs_j >= self.height:
                            continue

                        # Mark this cell to be highlighted
                        highlight_masks[abs_i, abs_j].append(i)

        # Render the whole grid
        img = self.grid.render(
            self.objects,
            agent_pos=self.agents[0].pos,
            agent_dir=self.agents[0].dir,
            tile_size=tile_size,
            highlight_mask=highlight_masks if highlight else None,
        )

        if self.render_mode == "human":
            self.window.show_img(img)

        return img

    def render(self):
        img = self.get_frame(self.highlight, self.tile_size, self.agent_pov)

        if self.render_mode == "human":
            img = np.transpose(img, axes=(1, 0, 2))
            if self.render_size is None:
                self.render_size = img.shape[:2]
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
                pygame.display.set_caption("minigrid")
            if self.clock is None:
                self.clock = pygame.time.Clock()
            surf = pygame.surfarray.make_surface(img)

            # Create background with mission description
            offset = surf.get_size()[0] * 0.1
            # offset = 32 if self.agent_pov else 64
            bg = pygame.Surface(
                (int(surf.get_size()[0] + offset), int(surf.get_size()[1] + offset))
            )
            bg.convert()
            bg.fill((255, 255, 255))
            bg.blit(surf, (offset / 2, 0))

            bg = pygame.transform.smoothscale(bg, (self.screen_size, self.screen_size))

            font_size = 22
            # text = self.mission
            # font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)
            # text_rect = font.get_rect(text, size=font_size)
            # text_rect.center = bg.get_rect().center
            # text_rect.y = bg.get_height() - font_size * 1.5
            # font.render_to(bg, text_rect, text, size=font_size)

            self.window.blit(bg, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return img

    def get_frame(
        self,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
    ):
        """Returns an RGB image corresponding to the whole environment or the agent's point of view.

        Args:

            highlight (bool): If true, the agent's field of view or point of view is highlighted with a lighter gray color.
            tile_size (int): How many pixels will form a tile from the NxM grid.
            agent_pov (bool): If true, the rendered frame will only contain the point of view of the agent.

        Returns:

            frame (np.ndarray): A frame of type numpy.ndarray with shape (x, y, 3) representing RGB values for the x-by-y pixel image.

        """

        if agent_pov:
            return self.get_pov_render(tile_size)
        else:
            return self.get_full_render(highlight, tile_size)

    def get_pov_render(self, tile_size):
        """
        Render an agent's POV observation for visualization
        """
        grid, vis_mask = self.gen_obs_grid()

        # Render the whole grid
        img = grid.render(
            tile_size,
            agent_pos=(self.agent_view_size // 2, self.agent_view_size - 1),
            agent_dir=3,
            highlight_mask=vis_mask,
        )

        return img

    def get_full_render(self, highlight, tile_size):
        """
        Render a non-paratial observation for visualization
        """
        # Compute which cells are visible to the agent
        _, vis_mask = self.gen_obs_grid()

        # Compute the world coordinates of the bottom-left corner
        # of the agent's view area
        f_vec = self.agents[0].dir_vec
        r_vec = self.agents[0].right_vec
        top_left = (
            self.agents[0].pos
            + f_vec * (self.agent_view_size - 1)
            - r_vec * (self.agent_view_size // 2)
        )

        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # For each cell in the visibility mask
        for vis_j in range(self.agent_view_size):
            for vis_i in range(self.agent_view_size):
                # If this cell is not visible, don't highlight it
                if not vis_mask[0][vis_i][vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                if abs_i < 0 or abs_i >= self.width:
                    continue
                if abs_j < 0 or abs_j >= self.height:
                    continue

                # Mark this cell to be highlighted
                highlight_mask[abs_i, abs_j] = True

        # Render the whole grid
        img = self.grid.render(
            world=self.objects,
            tile_size=tile_size,
            agent_pos=self.agents[0].pos,
            agent_dir=self.agents[0].dir,
            highlight_mask=highlight_mask if highlight else None,
        )

        return img
