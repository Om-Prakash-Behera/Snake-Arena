"""The core game engine having implementation of snake which has no idea of the gui 
and hence can be processed several times by an agent without major performance drops"""

from collections import deque
from enum import Enum, auto
from dataclasses import dataclass
import random

class Tile(Enum):
    EMPTY = auto()
    SNAKE_BODY = auto()
    SNAKE_HEAD = auto()
    WALL = auto()
    ITEM_APPLE = auto()

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

class Death(Enum):
    collided_with_wall = auto()
    collided_with_self = auto()

@dataclass(frozen=True)
class MoveRes:
    moved: bool
    ate: bool
    steps: int
    collided: bool
    won: bool
    done: bool
    score: int
    death_cause: Death | None


class SnakeEngine:
    def __init__(self, width: int = 16, height: int = 16, seed: int | None = None):
        self.width = width
        self.height = height

        self.reset(seed)


    def reset(self, seed: int | None = None):
        self.seed = seed
        self.randomizer = random.Random(seed)

        self._initialize_grid()

        self.steps = 0
        self.score = 0
        self.done = False
        self.won = False
        self.death_cause = None


    def _initialize_grid(self):
        # Initialize a grid of Tiles and set 
        self.grid = [[Tile.EMPTY for _ in range(self.width)] for _ in range(self.height)]
        for y in range(self.height):
            for x in range(self.width):
                if (x == 0 or y == 0 or x == self.width - 1 or y == self.height - 1):
                    self.grid[y][x] = Tile.WALL

        start_x, start_y = self.width // 2, self.height // 2

        # Snake initially at middle of screen pointing to right with length 3
        self.snake = deque([
            (start_x, start_y),
            (start_x - 1, start_y),
            (start_x - 2, start_y),
        ])
        self.direction = Direction.RIGHT

        # Copy data from snake deque to grid
        for x, y in self.snake:
            self.grid[y][x] = Tile.SNAKE_BODY
        self.grid[start_y][start_x] = Tile.SNAKE_HEAD

        # Spawn a single apple
        self.apple = self._spawn_item(Tile.ITEM_APPLE)


    def _spawn_item(self, item_type: Tile) -> tuple[int, int] | None:
        available = [(x, y) for y in range(self.height) for x in range(self.width) if self.grid[y][x] == Tile.EMPTY]

        if (not available):
            return None
        pos = self.randomizer.choice(available)
        x, y = pos
        self.grid[y][x] = item_type
        return (x, y)

    def _check_move(self, dir: Direction) -> bool:
        next_square = tuple(map(sum, zip(self.snake[0], dir.value)))

        if next_square == self.snake[-1]:
            return True
        
        if self.grid[next_square[1]][next_square[0]] in (Tile.WALL, Tile.SNAKE_BODY):
            return False
        
        return True

    def make_move(self, dir: Direction) -> MoveRes:

        # For stability
        if self.done:
            return MoveRes(
                moved=False,
                ate=False,
                steps=self.steps,
                collided=False,
                won=self.won,
                done=True,
                score=self.score,
                death_cause=self.death_cause
            )

        # Ignore a complete reversal
        opposite = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
        }
        if opposite[dir] == self.direction:
            dir = self.direction

        self.direction = dir
        self.steps += 1        
        next_x, next_y = tuple(map(sum, zip(self.snake[0], dir.value)))

        if not self._check_move(dir):
            self.done = True
            if self.grid[next_y][next_x] == Tile.WALL:
                self.death_cause = Death.collided_with_wall
            else:
                self.death_cause = Death.collided_with_self
            return MoveRes(
                moved=False,
                ate=False,
                steps=self.steps,
                collided=True,
                won=False,
                done=True,
                score=self.score,
                death_cause=self.death_cause
            )

        ate = (next_x, next_y) == self.apple

        # Pop tail if not eaten food
        if not ate:
            tail_x, tail_y = self.snake.pop()
            self.grid[tail_y][tail_x] = Tile.EMPTY

        # Push new head
        old_head_x, old_head_y = self.snake[0]
        self.grid[old_head_y][old_head_x] = Tile.SNAKE_BODY

        self.snake.appendleft((next_x, next_y))
        self.grid[next_y][next_x] = Tile.SNAKE_HEAD

        # If ate, now spawn new apple
        if ate:
            self.score += 1
            self.apple = self._spawn_item(Tile.ITEM_APPLE)

            if (self.apple is None):
                self.done = True
                self.won = True

        return MoveRes(
            moved=True,
            ate=ate,
            steps=self.steps,
            collided=False,
            won=self.won,
            done=self.done,
            score=self.score,
            death_cause=self.death_cause
        )      