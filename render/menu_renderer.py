"""Menu renderor to select various modes."""

import pygame

class MenuRenderer:

    def __init__(self, screen: pygame.surface):
        self.screen = screen

        self.width = 800
        self.height = 600

        self.title_font = pygame.font.Font(None, 72) 
        self.option_font = pygame.font.Font(None, 38) 
        self.hint_font = pygame.font.Font(None, 24)

        self.background_color = (18, 20, 24) 
        self.text_color = (235, 238, 242) 
        self.selected_color = (102, 225, 102) 
        self.disabled_color = (100, 105, 115)

        self.selected_option = 0

        self.options = [
            "Human Play",
            "AI Spectator (Greedy)",
            "AI Spectator (Flood Fill)",
            "AI Spectator (Hamiltonian)",
            "AI Spectator (DQN)",
            "Quit",
        ]

    def move_selection(self, direction: int) -> None:
        available = [0, 1, 2, 3, 4, 5]
        current_index = available.index(self.selected_option)
        current_index = (current_index + direction) % len(available)
        self.selected_option = available[current_index]
    
    def draw(self) -> None:

        self.screen.fill(self.background_color)

        width, height = self.screen.get_size()

        title = self.title_font.render(
            "SNAKE ARENA", True, self.text_color
        )
        self.screen.blit(
            title,
            title.get_rect(center=(width // 2, height // 3)),
        )

        for index, option in enumerate(self.options):
            # if index == 1:
            #     color = self.disabled_color
            # el
            if index == self.selected_option:
                color = self.selected_color
            else:
                color = self.text_color

            label = self.option_font.render(option, True, color)

            self.screen.blit(
                label,
                label.get_rect(
                    center=(width // 2, height // 2 + index * 35)
                ),
            )    

        hint = self.hint_font.render(
            "UP / DOWN: Select     ENTER: Confirm",
            True,
            self.text_color,
        )

        self.screen.blit(
            hint,
            hint.get_rect(
                center=(width // 2, height - 45)
            ),
        )

        pygame.display.flip()