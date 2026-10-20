"""Greedy, but also take the reachable area after each directional movement as a heuristic"""
from collections import deque

from agents.interface import Agent, Observation, DIRECTIONS, is_legal, next_pos, Position
from engine.game import Direction

# ALPHA: +ve to encourage larger area reachable
# BETA: -ve to discourage larger dist from apple
ALPHA = 2.0
BETA = -3.0

def reachable_area(start: Position, blocked: set[Position], walls: frozenset[Position], width: int, height: int) -> int:
    # Count reaachable cells with BFS

    queue = deque([start])
    visited = {start}

    while queue:
        x, y = queue.popleft()

        for dir in DIRECTIONS:
            dx, dy = dir.value

            neighbour = x + dx, y + dy

            if not (0 <= neighbour[0] < width and 0 <= neighbour[1] < height):
                continue

            if neighbour in visited:
                continue

            if neighbour in blocked or neighbour in walls:
                continue

            visited.add(neighbour)
            queue.append(neighbour)

    return len(visited)



class FloodFillAgent(Agent):

    def choose_action(self, observation: Observation) -> Direction:
        if observation.apple is None:
            return observation.direction

        head = observation.snake[0]
        apple = observation.apple

        best_dir = None
        best_score = float("-inf")

        for direction in DIRECTIONS:
            if not is_legal(observation, direction):
                continue

            next_sq = next_pos(head, direction)
            eat = next_sq == apple

            if eat:
                new_snake = (next_sq) + observation.snake
            else:
                new_snake = (next_sq) + observation.snake[:-1]

            blocked = set(new_snake[1:])

            area = reachable_area(next_sq, blocked, observation.walls, observation.width, observation.height)

            new_len = len(new_snake)

            if apple is None:
                distance = 0
            else:
                distance = abs(next_sq[0] - apple[0]) + abs(next_sq[1] - apple[1])

            score = ALPHA * area + BETA * distance

            if score > best_score:
                best_score = score
                best_dir = direction

        if best_dir is not None:
            return best_dir

        # No legal move
        return observation.direction