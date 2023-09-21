import math

import numpy as np

from gym_multigrid.rendering import (
    fill_coords,
    point_in_circle,
    point_in_rect,
    point_in_line,
    point_in_triangle,
    rotate_fn, downsample, highlight_img,
)

# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32


# Map of color names to RGB values
COLORS = {
    "red": np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "purple": np.array([112, 39, 195]),
    "yellow": np.array([255, 255, 0]),
    "grey": np.array([100, 100, 100]),
}

COLOR_NAMES = sorted(list(COLORS.keys()))


# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]


class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, world, type, color):
        assert type in world.OBJECT_TO_IDX, type
        assert color in world.COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos = None

        # Current position of the object
        self.cur_pos = None

    def can_overlap(self):
        """Can the agent overlap with this?"""
        return False

    def can_pickup(self):
        """Can the agent pick this up?"""
        return False

    def can_contain(self):
        """Can this contain another object?"""
        return False

    def see_behind(self):
        """Can the agent see behind this object?"""
        return True

    def toggle(self, env, pos):
        """Method to trigger/toggle an action this object performs"""
        return False

    def encode(self, world, current_agent=False):
        """Encode the a description of this object as a 3-tuple of integers"""
        if world.encode_dim == 3:
            return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color], 0)
        else:
            return (
                world.OBJECT_TO_IDX[self.type],
                world.COLOR_TO_IDX[self.color],
                0,
                0,
                0,
                0,
            )

    @staticmethod
    def decode(type_idx, color_idx, state):
        assert False, "not implemented"

    def render(self, r):
        """Draw this object with the given renderer"""
        raise NotImplementedError


class ObjectGoal(WorldObj):
    def __init__(self, world, index, target_type="ball", reward=1, color=None):
        if color is None:
            super().__init__(world, "objgoal", world.IDX_TO_COLOR[index])
        else:
            super().__init__(world, "objgoal", world.IDX_TO_COLOR[color])
        self.target_type = target_type
        self.index = index
        self.reward = reward

    def can_overlap(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Goal(WorldObj):
    def __init__(self, world, index, reward=1, color=None):
        if color is None:
            super().__init__(world, "goal", world.IDX_TO_COLOR[index])
        else:
            super().__init__(world, "goal", world.IDX_TO_COLOR[color])
        self.index = index
        self.reward = reward

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Switch(WorldObj):
    def __init__(self, world):
        super().__init__(world, "switch", world.IDX_TO_COLOR[0])

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Floor(WorldObj):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self, world, color="blue"):
        super().__init__(world, "floor", color)

    def can_overlap(self):
        return True

    def render(self, r):
        # Give the floor a pale color
        c = COLORS[self.color]
        r.setLineColor(100, 100, 100, 0)
        r.setColor(*c / 2)
        r.drawPolygon(
            [(1, TILE_PIXELS), (TILE_PIXELS, TILE_PIXELS), (TILE_PIXELS, 1), (1, 1)]
        )


class Lava(WorldObj):
    def __init__(self, world):
        super().__init__(world, "lava", "red")

    def can_overlap(self):
        return True

    def render(self, img):
        c = (255, 128, 0)

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))


class Wall(WorldObj):
    def __init__(self, world, color="grey"):
        super().__init__(world, "wall", color)

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Door(WorldObj):
    def __init__(self, world, color, is_open=False, is_locked=False):
        super().__init__(world, "door", color)
        self.is_open = is_open
        self.is_locked = is_locked

    def can_overlap(self):
        """The agent can only walk over this cell when the door is open"""
        return self.is_open

    def see_behind(self):
        return self.is_open

    def toggle(self, env, pos):
        # If the player has the right key to open the door
        if self.is_locked:
            if isinstance(env.carrying, Key) and env.carrying.color == self.color:
                self.is_locked = False
                self.is_open = True
                return True
            return False

        self.is_open = not self.is_open
        return True

    def encode(self, world, current_agent=False):
        """Encode the a description of this object as a 3-tuple of integers"""

        # State, 0: open, 1: closed, 2: locked
        if self.is_open:
            state = 0
        elif self.is_locked:
            state = 2
        elif not self.is_open:
            state = 1

        return (
            world.OBJECT_TO_IDX[self.type],
            world.COLOR_TO_IDX[self.color],
            state,
            0,
            0,
            0,
        )

    def render(self, img):
        c = COLORS[self.color]

        if self.is_open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
            return

        # Door frame and door
        if self.is_locked:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * np.array(c))

            # Draw key slot
            fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), c)
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))

            # Draw door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)


class Key(WorldObj):
    def __init__(self, world, color="blue"):
        super(Key, self).__init__(world, "key", color)

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))


class Ball(WorldObj):
    def __init__(self, world, index=0, reward=1):
        super(Ball, self).__init__(world, "ball", world.IDX_TO_COLOR[index])
        self.index = index
        self.reward = reward

    def can_pickup(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])


class Box(WorldObj):
    def __init__(self, world, color, contains=None):
        super(Box, self).__init__(world, "box", color)
        self.contains = contains

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0, 0, 0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)

    def toggle(self, env, pos):
        # Replace the box by its contents
        env.grid.set(*pos, self.contains)
        return True


class Agent(WorldObj):
    def __init__(self, world, index=0, view_size=7):
        super(Agent, self).__init__(world, "agent", world.IDX_TO_COLOR[index])
        self.pos = None
        self.dir = None
        self.index = index
        self.view_size = view_size
        self.carrying = None
        self.terminated = False
        self.started = True
        self.paused = False

    def render(self, img):
        c = COLORS[self.color]
        tri_fn = point_in_triangle(
            (0.12, 0.19),
            (0.87, 0.50),
            (0.12, 0.81),
        )
        # Rotate the agent based on its direction
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * self.dir)
        fill_coords(img, tri_fn, c)

    def encode(self, world, current_agent=False):
        """Encode the a description of this object as a 3-tuple of integers"""
        if world.encode_dim == 3:
            return (
                world.OBJECT_TO_IDX[self.type],
                world.COLOR_TO_IDX[self.color],
                self.dir,
            )
        elif self.carrying:
            if current_agent:
                return (
                    world.OBJECT_TO_IDX[self.type],
                    world.COLOR_TO_IDX[self.color],
                    world.OBJECT_TO_IDX[self.carrying.type],
                    world.COLOR_TO_IDX[self.carrying.color],
                    self.dir,
                    1,
                )
            else:
                return (
                    world.OBJECT_TO_IDX[self.type],
                    world.COLOR_TO_IDX[self.color],
                    world.OBJECT_TO_IDX[self.carrying.type],
                    world.COLOR_TO_IDX[self.carrying.color],
                    self.dir,
                    0,
                )

        else:
            if current_agent:
                return (
                    world.OBJECT_TO_IDX[self.type],
                    world.COLOR_TO_IDX[self.color],
                    0,
                    0,
                    self.dir,
                    1,
                )
            else:
                return (
                    world.OBJECT_TO_IDX[self.type],
                    world.COLOR_TO_IDX[self.color],
                    0,
                    0,
                    self.dir,
                    0,
                )

    @property
    def dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """

        assert self.dir >= 0 and self.dir < 4
        return DIR_TO_VEC[self.dir]

    @property
    def right_vec(self):
        """
        Get the vector pointing to the right of the agent.
        """

        dx, dy = self.dir_vec
        return np.array((-dy, dx))

    @property
    def front_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """

        return self.pos + self.dir_vec

    def get_view_coords(self, i, j):
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        agent's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the agent's view size.
        """

        ax, ay = self.pos
        dx, dy = self.dir_vec
        rx, ry = self.right_vec

        # Compute the absolute coordinates of the top-left view corner
        sz = self.view_size
        hs = self.view_size // 2
        tx = ax + (dx * (sz - 1)) - (rx * hs)
        ty = ay + (dy * (sz - 1)) - (ry * hs)

        lx = i - tx
        ly = j - ty

        # Project the coordinates of the object relative to the top-left
        # corner onto the agent's own coordinate system
        vx = rx * lx + ry * ly
        vy = -(dx * lx + dy * ly)

        return vx, vy

    def get_view_exts(self):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        """

        # Facing right
        if self.dir == 0:
            topX = self.pos[0]
            topY = self.pos[1] - self.view_size // 2
        # Facing down
        elif self.dir == 1:
            topX = self.pos[0] - self.view_size // 2
            topY = self.pos[1]
        # Facing left
        elif self.dir == 2:
            topX = self.pos[0] - self.view_size + 1
            topY = self.pos[1] - self.view_size // 2
        # Facing up
        elif self.dir == 3:
            topX = self.pos[0] - self.view_size // 2
            topY = self.pos[1] - self.view_size + 1
        else:
            assert False, "invalid agent direction"

        botX = topX + self.view_size
        botY = topY + self.view_size

        return (topX, topY, botX, botY)

    def relative_coords(self, x, y):
        """
        Check if a grid position belongs to the agent's field of view, and returns the corresponding coordinates
        """

        vx, vy = self.get_view_coords(x, y)

        if vx < 0 or vy < 0 or vx >= self.view_size or vy >= self.view_size:
            return None

        return vx, vy

    def in_view(self, x, y):
        """
        check if a grid position is visible to the agent
        """

        return self.relative_coords(x, y) is not None


class Grid:
    """
    Represent a grid and operations on it
    """

    # Static cache of pre-renderer tiles
    tile_cache = {}

    def __init__(self, width, height):
        assert width >= 3
        assert height >= 3

        self.width = width
        self.height = height

        self.grid = [None] * width * height

    def __contains__(self, key):
        if isinstance(key, WorldObj):
            for e in self.grid:
                if e is key:
                    return True
        elif isinstance(key, tuple):
            for e in self.grid:
                if e is None:
                    continue
                if (e.color, e.type) == key:
                    return True
                if key[0] is None and key[1] == e.type:
                    return True
        return False

    def __eq__(self, other):
        grid1 = self.encode()
        grid2 = other.encode()
        return np.array_equal(grid2, grid1)

    def __ne__(self, other):
        return not self == other

    def copy(self):
        from copy import deepcopy

        return deepcopy(self)

    def set(self, i, j, v):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.grid[j * self.width + i] = v

    def get(self, i, j):
        assert i >= 0 and i < self.width, "i = %d, width = %d" % (i, self.width)
        assert j >= 0 and j < self.height, "j = %d, height = %d" % (j, self.height)
        return self.grid[j * self.width + i]

    def horz_wall(self, world, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, obj_type(world))

    def vert_wall(self, world, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, obj_type(world))

    def wall_rect(self, x, y, w, h):
        self.horz_wall(x, y, w)
        self.horz_wall(x, y + h - 1, w)
        self.vert_wall(x, y, h)
        self.vert_wall(x + w - 1, y, h)

    def rotate_left(self):
        """
        Rotate the grid to the left (counter-clockwise)
        """

        grid = Grid(self.height, self.width)

        for i in range(self.width):
            for j in range(self.height):
                v = self.get(i, j)
                grid.set(j, grid.height - 1 - i, v)

        return grid

    def slice(self, world, topX, topY, width, height):
        """
        Get a subset of the grid
        """

        grid = Grid(width, height)

        for j in range(0, height):
            for i in range(0, width):
                x = topX + i
                y = topY + j

                if x >= 0 and x < self.width and y >= 0 and y < self.height:
                    v = self.get(x, y)
                else:
                    v = Wall(world)

                grid.set(i, j, v)

        return grid

    @classmethod
    def render_tile(cls, world, obj, highlights=[], tile_size=TILE_PIXELS, subdivs=3):
        """
        Render a tile and cache the result
        """

        key = (*highlights, tile_size)
        key = obj.encode(world) + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(
            shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8
        )

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj != None:
            obj.render(img)

        # Highlight the cell  if needed
        if len(highlights) > 0:
            for h in highlights:
                highlight_img(
                    img, color=COLORS[world.IDX_TO_COLOR[h % len(world.IDX_TO_COLOR)]]
                )

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    def render(self, world, tile_size, highlight_masks=None):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)

                # agent_here = np.array_equal(agent_pos, (i, j))
                tile_img = Grid.render_tile(
                    world,
                    cell,
                    highlights=[] if highlight_masks is None else highlight_masks[i, j],
                    tile_size=tile_size,
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def encode(self, world, vis_mask=None):
        """
        Produce a compact numpy encoding of the grid
        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, world.encode_dim), dtype="uint8")

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)

                    if v is None:
                        array[i, j, 0] = world.OBJECT_TO_IDX["empty"]
                        array[i, j, 1] = 0
                        array[i, j, 2] = 0
                        if world.encode_dim > 3:
                            array[i, j, 3] = 0
                            array[i, j, 4] = 0
                            array[i, j, 5] = 0

                    else:
                        array[i, j, :] = v.encode(world)

        return array

    def encode_for_agents(self, world, agent_pos, vis_mask=None):
        """
        Produce a compact numpy encoding of the grid
        """
        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, world.encode_dim), dtype="uint8")

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)

                    if v is None:
                        array[i, j, 0] = world.OBJECT_TO_IDX["empty"]
                        array[i, j, 1] = 0
                        array[i, j, 2] = 0
                        if world.encode_dim > 3:
                            array[i, j, 3] = 0
                            array[i, j, 4] = 0
                            array[i, j, 5] = 0

                    else:
                        array[i, j, :] = v.encode(
                            world, current_agent=np.array_equal(agent_pos, (i, j))
                        )

        return array

    # @staticmethod
    # def decode(array):
    #     """
    #     Decode an array grid encoding back into a grid
    #     """
    #
    #     width, height, channels = array.shape
    #     assert channels == 3
    #
    #     vis_mask = np.ones(shape=(width, height), dtype=bool)
    #
    #     grid = Grid(width, height)
    #     for i in range(width):
    #         for j in range(height):
    #             type_idx, color_idx, state = array[i, j]
    #             v = WorldObj.decode(type_idx, color_idx, state)
    #             grid.set(i, j, v)
    #             vis_mask[i, j] = (type_idx != OBJECT_TO_IDX['unseen'])
    #
    #     return grid, vis_mask

    def process_vis(grid, agent_pos):
        mask = np.zeros(shape=(grid.width, grid.height), dtype=bool)

        mask[agent_pos[0], agent_pos[1]] = True

        for j in reversed(range(0, grid.height)):
            for i in range(0, grid.width - 1):
                if not mask[i, j]:
                    continue

                cell = grid.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i + 1, j] = True
                if j > 0:
                    mask[i + 1, j - 1] = True
                    mask[i, j - 1] = True

            for i in reversed(range(1, grid.width)):
                if not mask[i, j]:
                    continue

                cell = grid.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i - 1, j] = True
                if j > 0:
                    mask[i - 1, j - 1] = True
                    mask[i, j - 1] = True

        for j in range(0, grid.height):
            for i in range(0, grid.width):
                if not mask[i, j]:
                    grid.set(i, j, None)

        return mask
