"""Snake moves in a hamiltonian cycle"""

from agents.interface import Agent, Observation, is_legal, Position
from engine.game import Direction

def make_cycle(width:int, height:int) -> list[Position]:
    # Constructs a hamiltonian cycle provided atleast one dim is even
    # Requirement can be explained by chessboard

    transpose = (height%2 == 1)
    w, h = (height, width) if transpose else (width, height)

    if (h%2 == 1):
        raise ValueError("Requirement: Atlease one dimention of grid must be even")

    cycle = [(x, 0) for x in range(w)]

    for y in range(1, h):
        if y % 2 == 1:
            cycle.extend(
                (x, y) for x in range(w - 1, 0, -1)
            )
        else:
            cycle.extend(
                (x, y) for x in range(1, w)
            )

    cycle.append((0, h - 1))

    for y in range(h - 2, 0, -1):
        cycle.append((0, y))

    if transpose:
        cycle = [(y, x) for x, y in cycle]

    return cycle

class HamiltonianAgent(Agent):

    def __init__(self, width: int, height: int) -> None:
        self.cycle = [(x + 1, y + 1) for x, y in make_cycle(width - 2, height - 2)]
        self.index = {pos: i for i, pos in enumerate(self.cycle)}
        self.cycle_started = False      # The 1st move is predetermined to ensure that snake moves into a hamiltonian cycle

    def choose_action(self, observation: Observation) -> Direction:

        if not self.cycle_started:
            self.cycle_started = True
            return Direction.UP


        head = observation.snake[0]

        next_index = (self.index[head] + 1) % len(self.cycle)
        next_sq = self.cycle[next_index]

        dx = next_sq[0] - head[0]
        dy = next_sq[1] - head[1]

        direction = None

        for d in Direction:
            if d.value == (dx, dy):
                direction = d
                break
            
        if direction is not None and is_legal(observation, direction):
            return direction

        return observation.direction