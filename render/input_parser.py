"""Translate Pygame and agent commands into those understandable by the application"""

import pygame
from enum import Enum, auto
from engine.game import Direction


class Command(Enum):
    BACK = auto()
    QUIT = auto()
    RESET = auto()
    CONFIRM = auto()


def parse_pygame_event(event: pygame.event.Event) -> Direction | Command | None:
    # Convert a pygame event in to a game command

    if event.type == pygame.QUIT:
        return Command.QUIT

    if event.type != pygame.KEYDOWN:
        return None

    # Allows arrow-key and wasd movement
    direction_keys = {
        pygame.K_UP: Direction.UP,
        pygame.K_w: Direction.UP,
        pygame.K_DOWN: Direction.DOWN,
        pygame.K_s: Direction.DOWN,
        pygame.K_LEFT: Direction.LEFT,
        pygame.K_a: Direction.LEFT,
        pygame.K_RIGHT: Direction.RIGHT,
        pygame.K_d: Direction.RIGHT,
    }

    if event.key in direction_keys:
        return direction_keys[event.key]

    if event.key in (pygame.K_RETURN, pygame.K_SPACE):
        return Command.CONFIRM

    if event.key == pygame.K_r:
        return Command.RESET

    if event.key == pygame.K_ESCAPE:
        return Command.BACK

    return None