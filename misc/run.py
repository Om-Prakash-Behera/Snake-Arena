"""For headless running of agents"""
import csv
from dataclasses import dataclass, asdict
from statistics import median
from pathlib import Path

from engine.game import SnakeEngine, Death, MoveRes

from agents.interface import Agent, observe
from agents.greedy import GreedyAgent
from agents.flood_fill import FloodFillAgent
from agents.hamiltonian import HamiltonianAgent
from agents.dqn import DQNAgent

AGENTS = {
    "greedy": GreedyAgent,
    "flood": FloodFillAgent,
    "hamiltonian": HamiltonianAgent,
    "dqn": DQNAgent
}


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    agent_name: str
    width: int
    height: int
    score: int
    steps: int
    won: bool
    timed_out: bool
    death_cause: Death | None

@dataclass(frozen=True)
class AgentSummary:
    agent_name: str
    episodes: int
    win_rate: float
    avg_score: float
    median_score: float
    best_score: int
    avg_steps: float
    timeouts: int
    self_collisions: int
    wall_collisions: int

def run_episode(agent_name: str, seed: int, width: int = 16, height: int = 16, max_steps: int | None = None) -> EpisodeResult:
    # Single run on an agent on a seed
    if agent_name not in AGENTS:
        raise ValueError(f"Unknown agent: {agent_name}")
    
    engine = SnakeEngine(width, height, seed)
    agent = AGENTS[agent_name](width, height)

    mx = float("inf") if max_steps is None else max_steps
    while not engine.done and engine.steps <= mx:
        observation = observe(engine)
        action = agent.choose_action(observation)
        res = engine.make_move(action)

    return EpisodeResult(
        seed=seed,
        agent_name=agent_name,
        width=width,
        height=height,
        score=res.score,
        steps=res.steps,
        won=res.won,
        timed_out=not res.done,
        death_cause=res.death_cause
    )

def summarize(results: list[EpisodeResult]):
    summaries = []
    for name in dict.fromkeys(result.agent_name for result in results):
        group = [result for result in results if result.agent_name == name]
        scores = [result.score for result in group]
        count = len(group)

        summaries.append(
            AgentSummary(
                agent_name=name,
                episodes=count,
                win_rate=100 * sum(result.won for result in group) / count,
                avg_score=sum(scores) / count,
                median_score=float(median(scores)),
                best_score=max(scores),
                avg_steps=sum(result.steps for result in group) / count,
                timeouts=sum(result.timed_out for result in group),
                self_collisions=sum(result.death_cause == Death.collided_with_self for result in group),
                wall_collisions=sum(result.death_cause == Death.collided_with_wall for result in group)
            )
        )

    return summaries

def print_summary(summaries):
    # Print comparison table
    if not summaries:
        print("No results.")
        return

    headers = (
        f"{'Agent':<14} {'Episodes':>8} {'Win %':>8} "
        f"{'Avg score':>10} {'Median':>8} {'Best':>6} "
        f"{'Avg steps':>10} {'Timeouts':>9}"
    )

    print("\n" + headers)
    print("-" * len(headers))

    for result in summaries:
        print(
            f"{result.agent_name:<14} {result.episodes:>8} "
            f"{result.win_rate:>7.1f}% "
            f"{result.avg_score:>10.2f} "
            f"{result.median_score:>8.1f} "
            f"{result.best_score:>6} "
            f"{result.avg_steps:>10.1f} "
            f"{result.timeouts:>9}"
        )

    print()


def save_csv(results: EpisodeResult, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = list(EpisodeResult.__dataclass_fields__)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)  # asdict convert a dataclass into a dict


def run_benchmark(
    agents=("greedy", "flood", "hamiltonian", "dqn"),
    episodes=20,
    seed=0,
    width=16,
    height=16,
    max_steps=10_000,
    output=None,
):
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if width < 2 or height < 2:
        raise ValueError("width and height must be at least 2")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if not agents:
        raise ValueError("At least one agent must be selected")


    unknown = set(agents) - AGENTS.keys()
    if unknown:
        raise ValueError(f"Unknown agents: {sorted(unknown)}")

    results = []
    total = len(agents) * episodes

    for agent_name in agents:
        print(f"\nRunning {agent_name}...")

        for episode in range(episodes):
            episode_seed = seed + episode

            result = run_episode(
                agent_name=agent_name,
                seed=episode_seed,
                width=width,
                height=height,
                max_steps=max_steps,
            )
            results.append(result)

            completed = len(results)
            # \r moves cursor to start of line, allowing overwrite
            # end = "" prevents moving to nwline
            # flush=True displays output immediately without waiting fro buffer
            print(
                f"\r  Episode {episode + 1:>4}/{episodes:<4}"
                f" | score: {result.score:>6}"
                f" | steps: {result.steps:>6}"
                f" | progress: {completed:>4}/{total:<4}",
                end="",
                flush=True,
            )

        print()

    print_summary(summarize(results))

    # Store res ina file if provided
    if output is not None:
        save_csv(results, output)
        print(f"Results saved to {output}")

    return results