import argparse
import json
import os
from datetime import datetime
import torch as t
from environment import QuantumBoardGameEnv, TicTacToeBoard
from spaces import ComplexProjectiveActionSpace
from mcts import MCTS


def get_env_int(name, default):
    val = os.getenv(name)
    return int(val) if val is not None else default


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_line(msg, logfile=None):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    if logfile is not None:
        with open(logfile, "a") as f:
            f.write(line + "\n")
            f.flush()


def action_summary(action: t.Tensor) -> dict:
    support = (action.abs() > 1e-8).nonzero(as_tuple=False).flatten().tolist()
    amps = {int(i): [float(action[i].real.item()), float(action[i].imag.item())] for i in support}
    return {
        "support": support,
        "norm": float(t.linalg.vector_norm(action).item()),
        "amps": amps,
    }


def extract_best_trajectory(mcts: MCTS, max_depth=None):
    trajectory = []
    env = mcts.env

    while not env.terminal:
        if max_depth is not None and len(env.state) >= max_depth:
            break

        root_children = mcts.tree[()]
        if len(root_children) == 0:
            break

        best_child, best_action = mcts.select_action()
        trajectory.append(
            {
                "depth": int(len(env.state)),
                "player": int(env.player),
                "abstract_child": tuple(int(x) for x in best_child),
                "visits": int(mcts.visits[best_child]),
                "mean_value": (mcts.values[best_child] / max(1, mcts.visits[best_child])).tolist(),
                "summary": action_summary(best_action),
            }
        )

        next_env = env.move(best_action)
        mcts.reroot(best_child, next_env)
        env = next_env

    return trajectory, env


def run_one(
    seed: int,
    k: int,
    num_simulations: int,
    n_workers: int,
    preview_depth: int,
    logfile=None,
    summary_every: int | None = 50,
    summary_depth: int | None = None,
    top_n: int = 3,
):
    log_line(f"seed={seed} started k={k} nsim={num_simulations} n_workers={n_workers}", logfile)

    t.manual_seed(seed)

    env = QuantumBoardGameEnv(TicTacToeBoard).create_initial_state()
    action_space = ComplexProjectiveActionSpace(d_action=9, seed=seed)

    mcts = MCTS(
        env=env,
        action_space=action_space,
        exploration_weight=1.0,
        widening_factor=2.0,
        num_simulations=num_simulations,
        seed=seed,
        show_progress=False,
    )

    stats = mcts.run(
        k=k,
        n_workers=n_workers,
        log_enabled=True,
        logfile=logfile,
        summary_every=summary_every,
        summary_depth=summary_depth,
        top_n=top_n,
    )
    trajectory, terminal_env = extract_best_trajectory(mcts, max_depth=preview_depth)

    result = {
        "seed": seed,
        "stats": stats,
        "trajectory_preview": trajectory,
        "terminal_reached_in_preview": bool(terminal_env.terminal),
        "preview_depth": preview_depth,
        "final_num_actions_in_preview": int(len(terminal_env.state)),
        "payoff": None if terminal_env.payoff is None else terminal_env.payoff.tolist(),
    }

    final_root_top = stats.get("final_root_top_strategies", [])
    final_depth_summary = stats.get("final_branch_counts_by_depth", [])

    log_line(
        f"seed={seed} finished "
        f"payoff={result['payoff']} "
        f"trajectory_len={len(trajectory)} "
        f"terminal_reached_in_preview={result['terminal_reached_in_preview']}",
        logfile,
    )

    for rank, strat in enumerate(final_root_top, start=1):
        log_line(
            f"seed={seed} final_root_top_{rank} "
            f"child={strat['child']} "
            f"visits={strat['visits']} "
            f"visit_fraction={strat['visit_fraction']:.6f} "
            f"mean_value={strat['mean_value']} "
            f"support={strat['action_summary']['support']}",
            logfile,
        )

    for depth_info in final_depth_summary:
        log_line(
            f"seed={seed} final_depth={depth_info['depth']} "
            f"num_nodes={depth_info['num_nodes']} "
            f"total_visits={depth_info['total_visits']}",
            logfile,
        )

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--num-simulations", type=int, default=None)
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--preview-depth", type=int, default=3)
    parser.add_argument("--summary-every", type=int, default=50)
    parser.add_argument("--summary-depth", type=int, default=-1)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--logfile", type=str, default="mcts_progress.log")
    parser.add_argument("--jsonfile", type=str, default="mcts_results.jsonl")
    args = parser.parse_args()

    k = args.k if args.k is not None else get_env_int("MCTS_K", 15)
    num_simulations = (
        args.num_simulations
        if args.num_simulations is not None
        else get_env_int("MCTS_NSIM", 500)
    )
    n_workers = args.n_workers if args.n_workers is not None else k

    seeds = list(range(args.start_seed, args.start_seed + args.num_runs))

    with open(args.logfile, "w") as f:
        f.write("")
    with open(args.jsonfile, "w") as f:
        f.write("")

    log_line(
        f"job started seeds={seeds} "
        f"k={k} nsim={num_simulations} n_workers={n_workers} "
        f"preview_depth={args.preview_depth} "
        f"summary_every={args.summary_every} "
        f"summary_depth={'all' if args.summary_depth < 0 else args.summary_depth} "
        f"top_n={args.top_n}",
        args.logfile,
    )

    for seed in seeds:
        result = run_one(
            seed=seed,
            k=k,
            num_simulations=num_simulations,
            n_workers=n_workers,
            preview_depth=args.preview_depth,
            logfile=args.logfile,
            summary_every=args.summary_every,
            summary_depth=None if args.summary_depth < 0 else args.summary_depth,
            top_n=args.top_n,
        )

        with open(args.jsonfile, "a") as f:
            f.write(json.dumps(result) + "\n")
            f.flush()

        log_line(f"seed={seed} appended to {args.jsonfile}", args.logfile)

    log_line("job completed", args.logfile)

