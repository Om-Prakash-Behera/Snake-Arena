"""Find the direction which takes you closer to the current apple while still being a legal move"""

from agents.interface import Agent, Observation, DIRECTIONS, is_legal, next_pos

from engine.game import Direction

class GreedyAgent(Agent):

    def choose_action(self, observation: Observation) -> Direction:
        if observation.apple is None:
            return observation.direction

        head = observation.snake[0]
        apple = observation.apple

        # To comp manhatten dist for each possible dir
        def mh_dist(direction: Direction):
            x, y = next_pos(head, direction)
            ax, ay = apple
            return abs(x - ax) + abs(y - ay)

        candidates = sorted(DIRECTIONS, key=mh_dist)

        for direction in candidates:
            if is_legal(observation, direction):
                return direction

        # No legal move
        return observation.direction