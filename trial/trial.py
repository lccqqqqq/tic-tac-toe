import argparse
from pathlib import Path
from datetime import datetime
import torch as t
import random

from environment_test import QuantumBoardGameEnv, TicTacToeBoard
from spaces_test import ComplexProjectiveActionSpace
from mcts_timer import MCTS

uniqueString = ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=4))

def append_log(logfile: Path, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with logfile.open("a") as f:
        f.write(f"[{timestamp}] {message}\n")
        f.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--num-simulations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--exploration-weight", type=float, default=1.0)
    parser.add_argument("--widening-factor", type=float, default=2.0)
    parser.add_argument("--profile-log", type=str, default=None)
    parser.add_argument("--run-log", type=str, default=None)
    args = parser.parse_args()

    if args.k <= 0:
        raise ValueError("k must be positive")
    if args.num_simulations <= 0:
        raise ValueError("num_simulations must be positive")

    if args.profile_log is None:
        args.profile_log = f"mcts_profile_k{args.k}_nsim{args.num_simulations}_{uniqueString}.txt"
    if args.run_log is None:
        args.run_log = f"mcts_run_k{args.k}_nsim{args.num_simulations}_{uniqueString}.log"

    profile_log = Path(args.profile_log).resolve()
    run_log = Path(args.run_log).resolve()

    with run_log.open("w") as f:
        f.write("===== Quantum Tic-Tac-Toe MCTS Run =====\n")
        f.flush()

    append_log(run_log, f"cwd={Path.cwd()}")
    append_log(run_log, f"profile_log={profile_log}")
    append_log(run_log, f"run_log={run_log}")
    append_log(run_log, f"seed={args.seed}")
    append_log(run_log, f"k={args.k}")
    append_log(run_log, f"num_simulations={args.num_simulations}")
    append_log(run_log, f"exploration_weight={args.exploration_weight}")
    append_log(run_log, f"widening_factor={args.widening_factor}")

    t.manual_seed(args.seed)

    env = QuantumBoardGameEnv(TicTacToeBoard).create_initial_state()
    action_space = ComplexProjectiveActionSpace(d_action=9, seed=args.seed)

    append_log(run_log, "environment and action space created")

    mcts = MCTS(
        env=env,
        action_space=action_space,
        exploration_weight=args.exploration_weight,
        widening_factor=args.widening_factor,
        num_simulations=args.num_simulations,
        seed=args.seed,
        show_progress=False,
    )

    append_log(run_log, "starting mcts.run()")
    stats = mcts.run(k=args.k, profile=True, logfile=str(profile_log))
    append_log(run_log, f"run finished stats={stats}")

    root_children = mcts.tree[()]
    append_log(run_log, f"num_root_children={len(root_children)}")

    if len(root_children) == 0:
        append_log(run_log, "no root children were created")
        return

    best_child, best_action = mcts.select_action()
    support = (best_action.abs() > 1e-8).nonzero(as_tuple=False).flatten().tolist()

    append_log(run_log, f"best_child={best_child}")
    append_log(run_log, f"best_action_norm={float(t.linalg.vector_norm(best_action).item()):.6f}")
    append_log(run_log, f"best_action_support={support}")

    for idx in support:
        append_log(run_log, f"best_action_site_{idx}={best_action[idx]}")

    next_env = env.move(best_action)
    branch_count = len(next_env._state_dict) if hasattr(next_env, "_state_dict") else -1

    append_log(run_log, f"post_move_terminal={next_env.terminal}")
    append_log(run_log, f"post_move_player={next_env.player}")
    append_log(run_log, f"post_move_branch_count={branch_count}")

    if next_env.terminal:
        append_log(run_log, f"post_move_payoff={next_env.payoff}")

    append_log(run_log, "job completed")


if __name__ == "__main__":
    # python -u run_qttt_mcts.py --k 20 --num-simulations 10
    main()