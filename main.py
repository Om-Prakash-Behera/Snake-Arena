
"""Entry point for the Snake game and headless benchmark."""

import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--headless", action="store_true")
parser.add_argument("--agents", nargs="+", default=["greedy", "flood", "hamiltonian", "dqn"])
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--max-steps", type=int, default=10_000)

parser.add_argument("--w", type=int, default=16)
parser.add_argument("--h", type=int, default=16)
parser.add_argument("--m", type=int, default=8)
parser.add_argument("--s", type=int, default=None)
parser.add_argument("--output", default=None)

args = parser.parse_args()

if args.headless:
    from misc.run import run_benchmark

    run_benchmark(
        agents=args.agents,
        episodes=args.episodes,
        seed=args.s if args.s is not None else 0,
        width=args.w,
        height=args.h,
        max_steps=args.max_steps,
        output=args.output,
    )
else:
    from render.app import App
    kwargs = {
        "width": args.w,
        "height": args.h,
        "moves_per_second": args.m,
    }

    if args.s is not None:
        kwargs["seed"] = args.s

    App(**kwargs).run()
