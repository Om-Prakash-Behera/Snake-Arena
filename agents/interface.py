"""Information available to an agent"""

from dataclasses import dataclass
from engine.game import Direction, SnakeEngine, Tile

DIRECTIONS = (
    Direction.UP,
    Direction.DOWN,
    Direction.LEFT,
    Direction.RIGHT
)

# For type hint
Position = tuple[int, int]

@dataclass(frozen=True)
class Observation:
    # SNapshot presented to an agent
    width: int
    height: int
    snake: tuple[Position, ...]
    apple: Position | None
    direction: Direction
    walls: frozenset[Position]

class Agent:
    "For all agents"
    def __init__(self, width: int, height: int):
        ...
    def choose_action(self, observation: Observation) -> Direction:
        ...

def observe(engine: SnakeEngine) -> Observation:
    # Generate an observation from Current gameState
    walls = set()

    for y, row in enumerate(engine.grid):
        for x, tile in enumerate(row):
            if tile == Tile.WALL:
                walls.add((x, y))

    return Observation(
        width=engine.width,
        height=engine.height,
        snake=tuple(engine.snake),
        apple=engine.apple,
        direction=engine.direction,
        walls=frozenset(walls)
    )

def next_pos(position: Position, direction: Direction) -> Position:
    x, y = position
    dx, dy = direction.value
    return (x + dx, y + dy)

def is_opposite(a: Direction, b: Direction) -> bool:
    return (a.value[0] == -b.value[0] and a.value[1] == -b.value[1])

def is_legal(observation: Observation, direction: Direction) -> bool:
    head = observation.snake[0]
    next_square = next_pos(head, direction)

    x, y = next_square

    if not (0 <= x <= observation.width):
        return False

    if not (0 <= y <= observation.height):
        return False

    if next_square in observation.walls:
        return False

    if is_opposite(direction, observation.direction):
        return False

    if next_square in observation.snake[:-1]:
        return False

    return True