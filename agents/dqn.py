"""Double Dueling DQN agent for Snake.

Usage:
Train (time-budgeted, default ~80 min wall clock):
    python -m agents.dqn --minutes 80 --min-size 7 --max-size 16

Evaluate a saved checkpoint on any board sizes:
    python -m agents.dqn --eval --eval-sizes 8x8 12x12 16x16 20x20

"""

from __future__ import annotations

import argparse
import copy
import random
import time
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.interface import Agent, Observation, DIRECTIONS, Position, is_opposite, observe
from engine.game import Direction, SnakeEngine

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "dqn_snake.pt"

N_ACTIONS = len(DIRECTIONS)  # UP, DOWN, LEFT, RIGHT (order of DIRECTIONS)
IN_CHANNELS = 4              # walls/padding, body age, head, apple

# Rewards
R_APPLE = 1.0
R_WIN = 1.0
R_DEATH = -1.0      # given when the snake has no safe move left
R_SHAPE = 0.01      # per unit of Manhattan distance gained towards the apple
R_STEP = 0.001      # small per-step cost


# State encoding (shared by training and inference so they cannot drift apart)

_WALL_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def _wall_plane(width: int, height: int, canvas: int) -> np.ndarray:
    key = (width, height, canvas)
    plane = _WALL_CACHE.get(key)
    if plane is None:
        plane = np.full((canvas, canvas), 255, dtype=np.uint8)  # padding == wall
        plane[1:height - 1, 1:width - 1] = 0
        _WALL_CACHE[key] = plane
    return plane


def encode_state(width: int, height: int, snake: tuple[Position, ...],
                 apple: Position | None, canvas: int) -> np.ndarray:
    # uint8 array (4, canvas, canvas), indexed [channel, y, x].

    # ch0: wall (border + padding outside the real board)
    # ch1: body; value = (moves until that cell is vacated) / snake length
    # ch2: head
    # ch3: apple
    
    state = np.zeros((IN_CHANNELS, canvas, canvas), dtype=np.uint8)
    state[0] = _wall_plane(width, height, canvas)

    n = len(snake)
    pts = np.asarray(snake, dtype=np.int64)
    if n > 1:
        age = (n - np.arange(1, n)) / n
        state[1, pts[1:, 1], pts[1:, 0]] = np.rint(age * 255).astype(np.uint8)
    state[2, pts[0, 1], pts[0, 0]] = 255
    if apple is not None:
        state[3, apple[1], apple[0]] = 255
    return state


def legal_mask(width: int, height: int, snake: tuple[Position, ...],
               direction: Direction) -> np.ndarray:
    # bool[4]: moves that do not immediately kill the snake (tail cell is allowed).
    hx, hy = snake[0]
    body = set(snake[:-1])
    mask = np.zeros(N_ACTIONS, dtype=bool)
    for i, d in enumerate(DIRECTIONS):
        x, y = hx + d.value[0], hy + d.value[1]
        if not (1 <= x <= width - 2 and 1 <= y <= height - 2):
            continue
        if is_opposite(d, direction) or (x, y) in body:
            continue
        mask[i] = True
    return mask


# Network

class DuelingQNet(nn.Module):
    # Fully convolutional dueling Q-network, independent of board size

    COORD_SCALE = 8.0

    def __init__(self, canvas: int = 16, channels: int = 32, hidden: int = 128):
        super().__init__()
        self.canvas, self.channels, self.hidden = canvas, channels, hidden

        def conv(i, o, dilation):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=dilation, dilation=dilation), nn.ReLU()
            )

        # Dilations 1,1,2,4,1 -> receptive field of 19 cells.
        self.body = nn.Sequential(
            conv(IN_CHANNELS + 2, channels, 1),
            conv(channels, channels, 1),
            conv(channels, channels, 2),
            conv(channels, channels, 4),
            conv(channels, channels, 1),
        )
        self.trunk = nn.Sequential(nn.Linear(3 * channels, hidden), nn.ReLU())
        self.value = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, N_ACTIONS))

    def forward(self, x: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        B, _, H, W = x.shape
        wall, head, apple = x[:, 0:1], x[:, 2:3], x[:, 3:4]

        # Coordinates relative to the head, computed in-network.
        xs = torch.arange(W, device=x.device, dtype=x.dtype).view(1, 1, 1, W)
        ys = torch.arange(H, device=x.device, dtype=x.dtype).view(1, 1, H, 1)
        hx = (head * xs).sum(dim=(1, 2, 3)).view(B, 1, 1, 1)
        hy = (head * ys).sum(dim=(1, 2, 3)).view(B, 1, 1, 1)
        dx = ((xs - hx) / self.COORD_SCALE).expand(B, 1, H, W)
        dy = ((ys - hy) / self.COORD_SCALE).expand(B, 1, H, W)

        f = self.body(torch.cat([x, dx, dy], dim=1))

        free = 1.0 - wall
        pooled = torch.cat(
            [
                (f * head).sum(dim=(2, 3)),
                (f * apple).sum(dim=(2, 3)),
                (f * free).sum(dim=(2, 3)) / free.sum(dim=(2, 3)).clamp(min=1.0),
            ],
            dim=1,
        )
        h = self.trunk(pooled)
        v, a = self.value(h), self.advantage(h)

        lf = legal.float()
        a_mean = (a * lf).sum(1, keepdim=True) / lf.sum(1, keepdim=True).clamp(min=1.0)
        return v + a - a_mean


# Agent (inference)

_MODEL_CACHE: dict[str, DuelingQNet] = {}


def load_model(path: str | Path | None = None, device: str = "cpu") -> DuelingQNet:
    path = Path(path or DEFAULT_CHECKPOINT)
    key = f"{path.resolve()}::{device}"
    if key not in _MODEL_CACHE:
        if not path.exists():
            raise FileNotFoundError(
                f"No DQN checkpoint at {path}. Train one with: python -m agents.dqn"
            )
        ckpt = torch.load(path, map_location=device)
        model = DuelingQNet(ckpt["canvas"], ckpt["channels"], ckpt["hidden"]).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


class DQNAgent(Agent):
    """Greedy policy over the trained network, restricted to non-fatal moves."""

    def __init__(self, width: int, height: int,
                 checkpoint: str | Path | None = None,
                 model: DuelingQNet | None = None) -> None:
        super().__init__(width, height)
        self.model = model if model is not None else load_model(checkpoint)
        self.device = next(self.model.parameters()).device

    def choose_action(self, observation: Observation) -> Direction:
        w, h = observation.width, observation.height
        legal = legal_mask(w, h, observation.snake, observation.direction)
        if not legal.any():
            return observation.direction  # trapped: rip snek

        canvas = max(self.model.canvas, w, h)
        state = encode_state(w, h, observation.snake, observation.apple, canvas)

        legal_t = torch.from_numpy(legal).to(self.device).unsqueeze(0)
        with torch.inference_mode():
            q = self.model(torch.from_numpy(state).to(self.device).unsqueeze(0), legal_t)[0]
        q = q.masked_fill(~legal_t[0], float("-inf"))
        return DIRECTIONS[int(q.argmax())]


# Training

@dataclass
class TrainConfig:
    minutes: float = 80.0           # wall-clock budget (includes periodic evals)
    min_size: int = 7               # board size range, INCLUDING the wall border
    max_size: int = 16              # also the canvas size
    num_envs: int = 8
    batch_size: int = 64
    buffer_size: int = 100_000
    learn_start: int = 5_000
    steps_per_update: int = 4       # env steps per gradient step
    gamma: float = 0.99
    n_step: int = 3
    lr_start: float = 5e-4
    lr_end: float = 5e-5
    tau: float = 0.005              # soft target update rate
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.85    # fraction of the time budget spent decaying epsilon
    channels: int = 32
    hidden: int = 128
    eval_every_min: float = 10.0
    eval_episodes: int = 20
    seed: int = 0
    device: str = "auto"
    out: str = str(DEFAULT_CHECKPOINT)
    resume: str | None = None


class ReplayBuffer:
    # Uniform replay storing uint8 states and n-step transitions

    def __init__(self, capacity: int, canvas: int):
        self.capacity = capacity
        self.s = np.zeros((capacity, IN_CHANNELS, canvas, canvas), dtype=np.uint8)
        self.s2 = np.zeros_like(self.s)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.disc = np.zeros(capacity, dtype=np.float32)   # gamma ** n for this sample
        self.done = np.zeros(capacity, dtype=np.float32)
        self.l = np.zeros((capacity, N_ACTIONS), dtype=bool)
        self.l2 = np.zeros((capacity, N_ACTIONS), dtype=bool)
        self.pos = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(self, s, a, r, s2, l, l2, done, disc, ) -> None:
        i = self.pos
        self.s[i], self.a[i], self.r[i], self.s2[i] = s, a, r, s2
        self.l[i], self.l2[i] = l, l2
        self.done[i], self.disc[i] = float(done), disc
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, nprng: np.random.Generator, device):
        idx = nprng.integers(0, self.size, batch)

        def t(arr):
            return torch.from_numpy(arr[idx]).to(device)

        return t(self.s), t(self.a), t(self.r), t(self.s2), t(self.l), t(self.l2), t(self.disc), t(self.done)


class _TrainEnv:
    # One SnakeEngine with random board size, reward shaping and n-step buffer

    def __init__(self, cfg: TrainConfig, rng: random.Random, canvas: int):
        self.cfg, self.rng, self.canvas = cfg, rng, canvas
        self.nbuf: deque = deque()
        self.reset()

    def reset(self) -> None:
        w = self.rng.randint(self.cfg.min_size, self.cfg.max_size)
        h = self.rng.randint(self.cfg.min_size, self.cfg.max_size)
        self.engine = SnakeEngine(w, h, self.rng.randrange(2 ** 31))
        self.width, self.height = w, h
        self.since_apple = 0
        self.stall_limit = 2 * (w - 2) * (h - 2)
        self.nbuf.clear()
        self._refresh()

    def _refresh(self) -> None:
        snake = tuple(self.engine.snake)
        self.state = encode_state(self.width, self.height, snake, self.engine.apple, self.canvas)
        self.legal = legal_mask(self.width, self.height, snake, self.engine.direction)

    def step(self, action: int) -> tuple[float, bool, bool]:
        # Returns (reward, terminal, truncated
        eng = self.engine
        hx, hy = eng.snake[0]
        ax, ay = eng.apple
        d_old = abs(hx - ax) + abs(hy - ay)

        res = eng.make_move(DIRECTIONS[action])
        self.since_apple += 1

        if res.collided:  # cannot happen with masking, kept for safety
            self._refresh()
            return R_DEATH, True, False
        if res.won:
            self._refresh()
            return R_APPLE + R_WIN, True, False

        if res.ate:
            reward = R_APPLE
            self.since_apple = 0
        else:
            nx, ny = eng.snake[0]
            d_new = abs(nx - ax) + abs(ny - ay)
            reward = R_SHAPE * (d_old - d_new) - R_STEP

        self._refresh()
        if not self.legal.any():          # every next move is fatal -> doomed
            return reward + R_DEATH, True, False
        return reward, False, self.since_apple >= self.stall_limit

    def record(self, replay: ReplayBuffer, s, a, r, l, done: bool, trunc: bool) -> None:
        # Push n-step transitions; self.state/legal must already be the NEXT state
        cfg = self.cfg
        self.nbuf.append((s, a, r, l))
        flush = done or trunc
        if not flush and len(self.nbuf) < cfg.n_step:
            return
        items = list(self.nbuf)
        for st in (range(len(items)) if flush else (0,)):
            seq = items[st:]
            ret = sum(cfg.gamma ** j * it[2] for j, it in enumerate(seq))
            replay.add(items[st][0], items[st][1], ret, self.state, items[st][3],
                       self.legal, done, cfg.gamma ** len(seq))
        if flush:
            self.nbuf.clear()
        else:
            self.nbuf.popleft()


def _learn(online, target, opt, replay, cfg, nprng, device) -> torch.Tensor:
    s, a, r, s2, l, l2, disc, done = replay.sample(cfg.batch_size, nprng, device)

    q = online(s, l).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        # Double DQN: online net selects, target net evaluates.
        best = online(s2, l2).masked_fill(~l2, float("-inf")).argmax(1, keepdim=True)
        q2 = target(s2, l2).gather(1, best).squeeze(1)
        y = r + disc * (1.0 - done) * q2

    loss = F.smooth_l1_loss(q, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), 10.0)
    opt.step()

    with torch.no_grad():  # Polyak averaging
        for pt, p in zip(target.parameters(), online.parameters()):
            pt.mul_(1.0 - cfg.tau).add_(p, alpha=cfg.tau)
    return loss.detach()


def evaluate(model: DuelingQNet, boards, episodes: int = 4, seed: int = 10_000,
             verbose: bool = True) -> float:
    # Greedy play through the real Agent/observe path. Returns mean fraction of
    # the maximum possible score, averaged over boards
    was_training = model.training
    model.eval()
    fractions = []
    for (w, h) in boards:
        scores = []
        for ep in range(episodes):
            engine = SnakeEngine(w, h, seed + ep)
            agent = DQNAgent(w, h, model=model)
            stall, limit = 0, 2 * (w - 2) * (h - 2)
            while not engine.done and stall < limit:
                res = engine.make_move(agent.choose_action(observe(engine)))
                stall = 0 if res.ate else stall + 1
            scores.append(engine.score)
        max_score = (w - 2) * (h - 2) - 3
        fractions.append(float(np.mean(scores)) / max_score)
        if verbose:
            print(f"    {w:>2}x{h:<2} avg score {np.mean(scores):6.1f}  best {max(scores):>4}"
                  f"  ({100 * fractions[-1]:.1f}% of max)")
    model.train(was_training)
    return float(np.mean(fractions))


def save_checkpoint(path: str | Path, model: DuelingQNet) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "canvas": model.canvas,
         "channels": model.channels, "hidden": model.hidden},
        path,
    )


def train(cfg: TrainConfig) -> None:
    if cfg.min_size < 6:
        raise ValueError("min_size must be >= 6 (the engine spawns a length-3 snake at the centre)")
    if cfg.max_size < cfg.min_size:
        raise ValueError("max_size must be >= min_size")

    rng = random.Random(cfg.seed)
    nprng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if cfg.device == "auto" else cfg.device))
    canvas = cfg.max_size

    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device)
        cfg.channels, cfg.hidden = ckpt["channels"], ckpt["hidden"]
    online = DuelingQNet(canvas, cfg.channels, cfg.hidden).to(device)
    if cfg.resume:
        online.load_state_dict(ckpt["model"])
    target = copy.deepcopy(online)
    for p in target.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(online.parameters(), lr=cfg.lr_start)

    replay = ReplayBuffer(cfg.buffer_size, canvas)
    envs = [_TrainEnv(cfg, rng, canvas) for _ in range(cfg.num_envs)]

    # boards = [b for b in [(8, 8), (12, 12), (16, 16), (9, 14)] if max(b) <= canvas]
    boards = [(16, 16)]
    boards = boards or [(canvas, canvas)]

    out = Path(cfg.out)
    last_path = out.with_name(out.stem + "_last" + out.suffix)

    print(f"Training on {device} | boards {cfg.min_size}-{cfg.max_size} (incl. walls) | "
          f"budget {cfg.minutes:.0f} min | params "
          f"{sum(p.numel() for p in online.parameters()):,}")

    t0 = time.time()
    budget = cfg.minutes * 60
    last_log = last_eval = t0
    best = -1.0
    recent: deque = deque(maxlen=200)
    env_steps = updates = pending = 0
    loss_acc = torch.zeros((), device=device)
    loss_n = 0
    neg_inf = float("-inf")

    while True:
        elapsed = time.time() - t0
        if elapsed >= budget:
            break
        frac = elapsed / budget
        eps = cfg.eps_start + (cfg.eps_end - cfg.eps_start) * min(1.0, frac / cfg.eps_decay_frac)
        lr = cfg.lr_start + (cfg.lr_end - cfg.lr_start) * frac
        for g in opt.param_groups:
            g["lr"] = lr

        # act: one batched forward pass for all envs 
        states = np.stack([e.state for e in envs])
        legals = np.stack([e.legal for e in envs])
        legal_t = torch.from_numpy(legals).to(device)
        with torch.no_grad():
            q = online(torch.from_numpy(states).to(device), legal_t).masked_fill(~legal_t, neg_inf)
            greedy = q.argmax(1).cpu().numpy()

        for i, e in enumerate(envs):
            s, l = e.state, e.legal
            if rng.random() < eps:
                a = rng.choice(np.flatnonzero(l).tolist())
            else:
                a = int(greedy[i])
            r, done, trunc = e.step(a)
            e.record(replay, s, a, r, l, done, trunc)
            if done or trunc:
                recent.append(e.engine.score)
                e.reset()

        env_steps += len(envs)

        # learn 
        if len(replay) >= cfg.learn_start:
            pending += len(envs)
            while pending >= cfg.steps_per_update:
                pending -= cfg.steps_per_update
                loss_acc += _learn(online, target, opt, replay, cfg, nprng, device)
                loss_n += 1
                updates += 1

        now = time.time()
        if now - last_log >= 60:
            avg = np.mean(recent) if recent else 0.0
            print(f"[{(now - t0) / 60:5.1f}m] env steps {env_steps:>9,} | updates {updates:>8,} | "
                  f"eps {eps:.2f} | lr {lr:.1e} | train score {avg:5.2f} | "
                  f"loss {loss_acc.item() / max(loss_n, 1):.4f}")
            loss_acc.zero_()
            loss_n = 0
            last_log = now

        if now - last_eval >= cfg.eval_every_min * 60:
            print("  eval:")
            score = evaluate(online, boards, cfg.eval_episodes)
            if score > best:
                best = score
                save_checkpoint(out, online)
                print(f"  new best ({100 * best:.1f}%) -> saved {out.name}")
            last_eval = time.time()

    print("Final eval:")
    score = evaluate(online, boards, cfg.eval_episodes)
    save_checkpoint(last_path, online)
    if score >= best:
        best = score
        save_checkpoint(out, online)
    print(f"Done in {(time.time() - t0) / 60:.1f} min. Best {100 * best:.1f}% -> {out}")


def _parse_sizes(items: list[str]) -> list[tuple[int, int]]:
    sizes = []
    for item in items:
        w, h = item.lower().split("x")
        sizes.append((int(w), int(h)))
    return sizes


def main() -> None:
    d = TrainConfig()
    p = argparse.ArgumentParser(description="Train / evaluate the Double Dueling DQN Snake agent")
    p.add_argument("--minutes", type=float, default=d.minutes)
    p.add_argument("--min-size", type=int, default=d.min_size)
    p.add_argument("--max-size", type=int, default=d.max_size)
    p.add_argument("--envs", type=int, default=d.num_envs)
    p.add_argument("--batch", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr_start)
    p.add_argument("--eps-start", type=float, default=d.eps_start)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--device", default=d.device)
    p.add_argument("--out", default=d.out)
    p.add_argument("--resume", default=None, help="checkpoint to continue from")
    p.add_argument("--eval", action="store_true", help="only evaluate --out")
    p.add_argument("--eval-sizes", nargs="+", default=["8x8", "12x12", "16x16", "20x20"])
    p.add_argument("--eval-episodes", type=int, default=10)
    args = p.parse_args()

    if args.eval:
        model = load_model(args.out)
        evaluate(model, _parse_sizes(args.eval_sizes), args.eval_episodes)
        return

    train(TrainConfig(
        minutes=args.minutes, min_size=args.min_size, max_size=args.max_size,
        num_envs=args.envs, batch_size=args.batch, lr_start=args.lr,
        eps_start=args.eps_start, seed=args.seed, device=args.device,
        out=args.out, resume=args.resume,
    ))


if __name__ == "__main__":
    main()