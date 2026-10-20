"""Pygame renderer for the snake engine

THis only handles drawing. DOesn't validate anything
"""

import pygame

from engine.game import SnakeEngine, Tile


class Renderer:

    def __init__(
        self,
        engine: SnakeEngine,
        screen: pygame.surface,
        tile_size: int = 32,
        status_height: int = 48,
    ) -> None:
        self.engine = engine
        self.screen = screen
        self.tile_size = tile_size
        self.status_height = status_height

        self.screen_width = engine.width * tile_size
        self.screen_height = status_height + engine.height * tile_size

        # Fonts are created once rather than once per frame.
        self.font = pygame.font.Font(None, 30)
        self.large_font = pygame.font.Font(None, 64)

        self.colors = {
            Tile.EMPTY: (28, 32, 38),
            Tile.WALL: (90, 98, 110),
            Tile.SNAKE_BODY: (46, 180, 82),
            Tile.SNAKE_HEAD: (102, 225, 102),
            Tile.ITEM_APPLE: (225, 55, 55),
        }

    def draw(self) -> None:
        """Render one complete frame using the engine's current state."""

        self.screen.fill((18, 20, 24))
        self._draw_status()
        self._draw_board()

        if self.engine.done:
            self._draw_end_overlay()

        # Pygame has now received all drawing commands for this frame.
        # flip() makes the finished frame visible in the window.
        pygame.display.flip()

    def _draw_status(self) -> None:
        if self.engine.won:
            status = "You win!"
        elif self.engine.done:
            status = "Game over"
        else:
            status = "Playing"

        label = self.font.render(
            f"Score: {self.engine.score}    {status}",
            True,
            (240, 240, 240),
        )
        self.screen.blit(label, (10, 10))

    def _draw_board(self) -> None:
        """Draw every grid cell from ``engine.grid[y][x]``."""

        for y, row in enumerate(self.engine.grid):
            for x, tile in enumerate(row):
                rect = self._tile_rect(x, y)

                if tile == Tile.ITEM_APPLE:
                    # A circle is visually clearer than a full red square.
                    pygame.draw.rect(self.screen, self.colors[Tile.EMPTY], rect)
                    pygame.draw.circle(
                        self.screen,
                        self.colors[Tile.ITEM_APPLE],
                        rect.center,
                        max(3, self.tile_size // 3),
                    )
                else:
                    pygame.draw.rect(self.screen, self.colors[tile], rect)

                # A thin border makes individual cells visible without
                # changing the engine's logical board.
                pygame.draw.rect(self.screen, (12, 14, 18), rect, 1)

    def _draw_end_overlay(self) -> None:
        """Show a hint over the board when no more moves are possible."""

        overlay = pygame.Surface(
            (self.screen_width, self.screen_height - self.status_height),
            pygame.SRCALPHA,
        )
        overlay.fill((0, 0, 0, 135))
        self.screen.blit(overlay, (0, self.status_height))

        message = "YOU WIN" if self.engine.won else "GAME OVER"
        text = self.large_font.render(message, True, (255, 255, 255))
        hint = self.font.render("R: Restart    Enter/Space: Menu", True, (255, 255, 255))

        board_center_x = self.screen_width // 2
        board_center_y = self.status_height + (
            self.engine.height * self.tile_size
        ) // 2
        self.screen.blit(text, text.get_rect(center=(board_center_x, board_center_y - 24)))
        self.screen.blit(hint, hint.get_rect(center=(board_center_x, board_center_y + 28)))

    def _tile_rect(self, x: int, y: int) -> pygame.Rect:
        # Get screen rectangle corresponding to board coordinate (x, y)

        return pygame.Rect(
            x * self.tile_size,
            self.status_height + y * self.tile_size,
            self.tile_size,
            self.tile_size,
        )