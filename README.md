# Snake Arena

A Snake game with a clean separation between the **engine**, the **renderer** and the **agents**. You can play it yourself, watch AI agents play, or benchmark agents headlessly. Agents range from classic heuristics (greedy, flood fill, Hamiltonian cycle) to a **Double Dueling DQN** trained with reinforcement learning that works on many board sizes.

## Features

- Pygame front end with a menu, human play and AI spectator modes
- Headless benchmark runner with summary table and CSV export
- Four agents behind one interface: Greedy, Flood Fill, Hamiltonian, and DQN
- DQN trained on randomly sized boards, so one model plays many grid sizes

## Project structure

```
.
├── main.py                 # Entry point (GUI or headless benchmark)
├── engine/
│   └── game.py             # SnakeEngine, Direction, Tile, Death, MoveRes
├── agents/
│   ├── interface.py        # Agent base class, Observation, observe(), is_legal()
│   ├── greedy.py           # GreedyAgent
│   ├── flood_fill.py       # FloodFillAgent
│   ├── hamiltonian.py      # HamiltonianAgent
│   └── dqn.py              # DQNAgent + training / evaluation CLI
├── render/
│   ├── app.py              # Application loop and state machine
│   ├── renderer.py         # Game board drawing
│   ├── menu_renderer.py    # Main menu
│   └── input_parser.py     # Pygame events -> Direction / Command
└── misc/
    └── run.py              # Headless benchmark (run_episode, run_benchmark)
```

## Installation

Use latest version of Python

```bash
pip install pygame numpy torch
```

`torch` is only needed for the DQN agent. `gymnasium` and `matplotlib` are only needed for the FrozenLake reference scripts.

## Running the game

```bash
python main.py                      # GUI, 16x16 board
python main.py --w 20 --h 12 --m 12 # custom size, 12 moves per second
python main.py --s 42               # fixed seed (same apple sequence for every game)
```

### Menu and controls

| Key | Action |
|---|---|
| Up / Down | Move menu selection |
| Enter / Space | Confirm |
| Arrow keys or WASD | Steer the snake (Human Play) |
| R | Restart (after game over / win) |
| Esc | Back to menu / quit from menu |

Menu modes: Human Play, AI Spectator (Greedy), AI Spectator (Flood Fill), AI Spectator (Hamiltonian), Quit. Add a DQN entry as described under [Adding an agent](#adding-an-agent).

## Headless benchmark

```bash
python main.py --headless --agents greedy flood hamiltonian --episodes 20 --w 16 --h 16
python main.py --headless --agents greedy dqn --episodes 50 --max-steps 10000 --output results/run1.csv
```

| Flag | Default | Meaning |
|---|---|---|
| `--headless` | off | Run the benchmark instead of the GUI |
| `--agents` | greedy flood hamiltonian | Agents to compare |
| `--episodes` | 20 | Episodes per agent |
| `--max-steps` | 10000 | Step cap per episode (counts as a timeout) |
| `--w`, `--h` | 16, 16 | Board size, **including** the wall border |
| `--s` | 0 | First seed; episode *i* uses seed + *i* |
| `--m` | 8 | Moves per second (GUI only) |
| `--output` | none | Write per-episode results to CSV |

The summary table shows win rate, average / median / best score, average steps and timeouts. The CSV has one row per episode (seed, agent, size, score, steps, won, timed_out, death_cause).

## The engine

`SnakeEngine(width, height, seed)` in `engine/game.py`:

- `width` and `height` include a 1-cell wall border, so the default 16x16 board has a 14x14 playable area. The minimum width is 6.
- The snake starts at the centre with length 3, heading right.
- `make_move(direction)` returns a `MoveRes` (moved, ate, steps, collided, won, done, score, death_cause). A 180° reversal is ignored and the snake continues straight.
- Moving into the cell the tail is leaving is legal.
- The game is won when the snake fills the board (no space to spawn an apple).
- A seed makes apple placement fully reproducible.

## Agents

All agents implement `Agent.choose_action(observation) -> Direction`. An `Observation` holds the board size, snake (head first), apple, current direction and the set of wall cells; `observe(engine)` builds one.

| Agent | Idea |
|---|---|
| **Greedy** | Takes the legal move that minimises Manhattan distance to the apple. Fast, but traps itself as it grows. |
| **Flood Fill** | Scores each legal move by reachable free area (BFS) and distance to the apple, so it avoids sealing itself in. |
| **Hamiltonian** | Follows a fixed cycle through every playable cell. Never dies, but needs at least one even playable dimension (`width-2` or `height-2`) and is slow. |
| **DQN** | Learned policy, described below. |

### Double Dueling DQN

`agents/dqn.py` contains the agent and its trainer.

- **Double DQN:** the online network picks the next action and the target network evaluates it. The target is updated by Polyak averaging.
- **Dueling head:** `Q = V + A - mean(A)` over legal moves.
- **n-step returns** (n = 3) for better credit assignment.
- **Action masking:** immediately fatal moves are excluded in training and at play time, so the network learns long-term planning rather than basic collision avoidance.
- **Reward shaping:** +1 per apple, a small bonus/penalty for moving towards/away from the apple, a small step cost, and -1 when no safe move remains.
- **Board-size independence:** the network is fully convolutional and boards are padded with walls onto a fixed canvas. Head-relative coordinates are computed inside the network and features are pooled with masks, so one set of weights works on any board size.
- **Input encoding** (4 planes): walls/padding, body (value = moves until the cell is vacated, scaled by length), head, apple.

#### Train

Run from the project root so the package imports resolve.

```bash
python -m agents.dqn --minutes 3 --out agents/test.pt        # dry run
python -m agents.dqn --minutes 80 --min-size 7 --max-size 16 # full run
```

Training is limited by wall-clock time (default 80 minutes) and uses a GPU automatically if available. Each episode uses a random board size in `[min-size, max-size]` (including walls, non-square allowed). Set `--max-size` to at least the largest board you want to play. It logs every minute, evaluates every 10 minutes and writes:

- `agents/dqn_snake.pt`: best checkpoint (loaded by `DQNAgent`)
- `agents/dqn_snake_last.pt`: final weights

#### Resume or fine-tune

```bash
python -m agents.dqn --resume agents/dqn_snake.pt --out agents/dqn_snake_v2.pt \
    --minutes 40 --eps-start 0.3 --lr 2e-4
```

Use a different `--out` so the new run does not overwrite the old best. Add `--max-size 24` to extend to larger boards. The replay buffer is not saved, so it is rebuilt at the start of each run.

#### Evaluate

```bash
python -m agents.dqn --eval --out agents/dqn_snake.pt \
    --eval-sizes 8x8 12x12 16x16 20x20 --eval-episodes 10
```

Or benchmark it against the other agents with `python main.py --headless --agents greedy flood dqn ...`.

#### Training flags

| Flag | Default | Meaning |
|---|---|---|
| `--minutes` | 80 | Wall-clock training budget |
| `--min-size`, `--max-size` | 7, 16 | Board size range incl. walls; `--max-size` is the canvas size |
| `--envs` | 8 | Parallel environments |
| `--batch` | 64 | Batch size |
| `--lr` | 5e-4 | Starting learning rate (decays to 5e-5) |
| `--eps-start` | 1.0 | Initial exploration rate |
| `--device` | auto | `cpu` / `cuda` |
| `--seed` | 0 | Random seed |
| `--out` | `agents/dqn_snake.pt` | Best-checkpoint path |
| `--resume` | none | Checkpoint to continue from |

## Adding an agent

1. Create `agents/my_agent.py` with a class that subclasses `Agent`, accepts `(width, height)` and implements `choose_action(observation) -> Direction`.
2. Register it in the `AGENTS` dict in `misc/run.py` (headless benchmark).
3. For the GUI, add a label to `MenuRenderer.options`, extend `move_selection` if the option count changes, and handle the new index in `App._confirm_menu_selection`, for example `self._reset_game(AppState.SPECTATING, DQNAgent)`.

## Reference material

`frozen_lake_dql.py` and `frozen_lake_dql_cnn.py` are simple DQN examples on Gymnasium's FrozenLake (one-hot MLP and a small CNN). They served as references for the Snake DQN and are independent of the game.