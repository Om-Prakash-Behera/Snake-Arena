"""Application loop that connects input, engine updates, and rendering."""

from enum import Enum, auto

import pygame

from engine.game import Direction, SnakeEngine
from render.input_parser import Command, parse_pygame_event
from render.renderer import Renderer
from render.menu_renderer import MenuRenderer

from agents.interface import observe
from agents.greedy import GreedyAgent
from agents.flood_fill import FloodFillAgent
from agents.hamiltonian import HamiltonianAgent
from agents.dqn import DQNAgent


class AppState(Enum):
    # States of the Application State Machine

    MENU = auto()
    PLAYING = auto()
    SPECTATING = auto()
    GAME_OVER = auto()
    WON = auto()
    EXIT = auto()


class App:
    # Coordinates everything

    def __init__(self, width: int = 16, height: int = 16, moves_per_second: int = 8, seed=None) -> None:
        # NO sound in game. SO dont initialize mixer
        pygame.display.init()
        pygame.font.init()

        # self.seed = random.Random()
        self.width = width
        self.height = height
        self.seed = seed
        self.engine = SnakeEngine(width=width, height=height, seed=seed)
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Snake Arena")

        self.renderer = Renderer(self.engine, self.screen)
        self.menu_renderer = MenuRenderer(self.screen)
        
        self.clock = pygame.time.Clock()
        self.move_interval = 1000 / moves_per_second
        self.time_since_move = 0.0

        self.state = AppState.MENU
        self.play_mode = AppState.PLAYING
        self.agent = None
        self.AI = None
        self.current_direction = self.engine.direction
        self.running = True

    def run(self) -> None:
        # Game Loop

        # self._set_screen_size(self.renderer.screen_width, self.renderer.screen_height)
        try:
            while self.running:
                self._handle_events()

                if self.state in (AppState.PLAYING, AppState.SPECTATING):
                    self._move_when_due()

                if self.state == AppState.MENU:
                    self.menu_renderer.draw()
                else:
                    self.renderer.draw()

                # Limits max fps to 60
                elapsed_ms = self.clock.tick(60)
                if self.state in (AppState.PLAYING, AppState.SPECTATING):
                    self.time_since_move += elapsed_ms  
        finally:
            # This runs even if the loop exits dur to error
            pygame.quit()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            command = parse_pygame_event(event)

            if command == Command.QUIT:
                self.state = AppState.EXIT
                self.running = False
                continue

            if self.state == AppState.MENU:
                if command == Command.BACK:
                    self.state = AppState.EXIT
                    self.running = False

                elif command == Direction.UP:
                    self.menu_renderer.move_selection(-1)

                elif command == Direction.DOWN:
                    self.menu_renderer.move_selection(1)

                elif command == Command.CONFIRM:
                    self._confirm_menu_selection()

                continue

            if self.state in (AppState.PLAYING, AppState.SPECTATING):
                if command == Command.BACK:
                    self._return_to_menu()

                elif (isinstance(command, Direction) and self.state == AppState.PLAYING):
                    self.current_direction = command

                continue

            if self.state in (AppState.GAME_OVER, AppState.WON):
                if command == Command.RESET:
                    self._reset_game(self.play_mode, self.AI)

                elif command in (Command.CONFIRM, Command.BACK):
                    self._return_to_menu()

    def _confirm_menu_selection(self) -> None:
        selection = self.menu_renderer.selected_option

        if selection == 0:
            self._reset_game(AppState.PLAYING)

        elif selection == 1:
            self._reset_game(AppState.SPECTATING, GreedyAgent)

        elif selection == 2:
            self._reset_game(AppState.SPECTATING, FloodFillAgent)

        elif selection == 3:
            self._reset_game(AppState.SPECTATING, HamiltonianAgent)

        elif selection == 4:
            self._reset_game(AppState.SPECTATING, DQNAgent)

        else:
            self.state = AppState.EXIT
            self.running = False

    def _move_when_due(self) -> None:
        # Rate of movement handled independently of frame rate

        if self.time_since_move < self.move_interval:
            return

        self.time_since_move -= self.move_interval

        if self.state == AppState.SPECTATING:
            observation = observe(self.engine)
            action = self.agent.choose_action(observation)
        else:
            action = self.current_direction

        result = self.engine.make_move(action)

        if result.done:
            self.state = AppState.WON if result.won else AppState.GAME_OVER

    def _set_screen_size(self, width: int = 16, height: int = 16) -> None:
        self.screen = pygame.display.set_mode((width, height))
        self.renderer.screen = self.screen
        

    def _reset_game(self, mode: AppState, AI=None) -> None:

        self.engine.reset(self.seed)
        self.play_mode = mode
        self.current_direction = self.engine.direction
        self.time_since_move = 0.0

        if mode == AppState.SPECTATING:
            self.agent = AI(self.width, self.height)
            self.AI = AI
        else:
            self.agent = None
            self.AI = None

        self._set_screen_size(self.renderer.screen_width, self.renderer.screen_height)

        self.state = mode

    def _return_to_menu(self) -> None:
        self.state = AppState.MENU
        self.agent = None
        self.time_since_move = 0.0

        self.menu_renderer.selected_option = 0
        self._set_screen_size(self.menu_renderer.width, self.menu_renderer.height)