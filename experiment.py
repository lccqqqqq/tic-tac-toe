from game import Board, QuantumGameEnv
from agent_cl import MCTSAgent, MinimaxAgent, RandomAgent  # RandomAgent may not be defined here; keep minimal imports
from agent_qu import MCTSQAgent, MCTSQAgentVanillaGradient, RandomQAgent, QAgent
from tqdm import tqdm
import os
import json
import argparse
import sys
import numpy as np
import pandas as pd
import time


def gather_game_stats(player_x, player_o, num_games: int = 100):
    print(
        f"Player X: {player_x.__class__.__name__}, Player O: {player_o.__class__.__name__}"
    )
    sys.stdout.flush()

    game_stats: list[dict] = []
    wins_x = 0
    wins_o = 0
    draws = 0

    for _ in tqdm(range(num_games)):
        board = Board.create_initial_board()
        actions: list[int] = []
        steps: list[dict] = []

        # play the game, agents get actions in turn
        while not board.terminal:
            current = player_x if board.player == 1 else player_o
            # Snapshot before move
            snapshot_before = {
                "board": board.board.tolist(),
                "player": int(board.player),
            }

            # Try to pass current board if the agent supports it
            action = None
            try:
                action = current.get_action(board=board)
            except TypeError:
                # Fallback for agents without board kwarg
                action = current.get_action()

            if action is None:
                # No legal moves or agent abstained; stop to avoid infinite loop
                break

            actions.append(int(action))
            new_board = board.move(action)
            steps.append({
                "player": int(board.player),
                "action": int(action),
                "board_before": snapshot_before["board"],
                "board_after": new_board.board.tolist(),
            })
            board = new_board

        # update aggregate results (do not store per-game result)
        result = int(board.payoff) if board.terminal else 0
        if result > 0:
            wins_x += 1
        elif result < 0:
            wins_o += 1
        else:
            draws += 1
        # store only actions as the game stats
        game_stats.append({"actions": actions})

    # print statistics of the game, expected wins
    expected_x = (wins_x - wins_o) / float(num_games)

    print(f"Games: {num_games}, X wins: {wins_x}, O wins: {wins_o}, Draws: {draws}")
    print(f"Expected value for X (1=win,-1=loss): {expected_x:.3f}")

    # save the game stats to a file, the name should reflect the agents
    os.makedirs("Data", exist_ok=True)
    fname = f"Data/stats_{player_x.__class__.__name__}_vs_{player_o.__class__.__name__}.json"
    with open(fname, "w") as f:
        json.dump({
            "player_x": player_x.__class__.__name__,
            "player_o": player_o.__class__.__name__,
            "num_games": num_games,
            "wins_x": wins_x,
            "wins_o": wins_o,
            "draws": draws,
            "expected_x": expected_x,
            "games": game_stats,  # each entry contains only {"actions": [...]}
        }, f, indent=2)
    print(f"Saved stats to {fname}")

def simulate_quantum_game(agent_x: QAgent, agent_o: QAgent, num_games: int = 100):
    """
    simulate the quantum tic-tac-toe.
    Sanity check: payoff of MCTS
    """
    print(f"Quantum Player X: {agent_x.__class__.__name__}, Quantum Player O: {agent_o.__class__.__name__}")
    sys.stdout.flush()

    game_stats: list[dict] = []
    acc_payoff = 0.0
    failed_games = 0

    for game_idx in tqdm(range(num_games)):
        try:
            # Initialize quantum game environment
            env = QuantumGameEnv.create_initial_state()
            actions: list[complex] = []
            steps: list[dict] = []

            # Play the quantum game
            while not env.terminal:
                current = agent_x if env.current_player == 1 else agent_o
                
                # Get quantum action (normalized complex vector)
                try:
                    action = current.get_action(env=env)
                except TypeError:
                    # Fallback for agents without env parameter
                    action = current.get_action()

                if action is None:
                    break

                # Store action and game state information
                actions.append(action.copy() if hasattr(action, 'copy') else action)
                
                # Snapshot before move
                state_info = {
                    "player": int(env.current_player),
                    "num_states": len(env.state),
                    "total_amplitude": float(np.sum(np.abs(list(env.state.values()))**2))
                }

                # Apply quantum action
                try:
                    new_env = env.move(action)
                    steps.append({
                        "player": int(env.current_player),
                        "action": action.tolist(),
                        "state_before": state_info,
                        "state_after": {
                            "num_states": len(new_env.state),
                            "total_amplitude": float(np.sum(np.abs(list(new_env.state.values()))**2))
                        }
                    })
                    env = new_env
                except ValueError as e:
                    # Handle quantum mechanics violations
                    print(f"Game {game_idx} failed: {e}")
                    failed_games += 1
                    break

            if env.terminal:
                # Calculate final payoff (from player 1's perspective)
                payoff = float(env.payoff)
                acc_payoff += payoff
                # Store game statistics
                game_stats.append({
                    "player_x": agent_x.__class__.__name__,
                    "player_o": agent_o.__class__.__name__,
                    "actions": [np.asarray(a) for a in actions],
                    "payoff": payoff,  # Always from player 1's perspective
                })

        except Exception as e:
            print(f"Unexpected error in game {game_idx}: {e}")
            failed_games += 1
            raise e

    # Calculate statistics (from player 1's perspective)
    print(f"Failed games: {failed_games}")
    sys.stdout.flush()
    valid_games = num_games - failed_games
    expected_payoff = acc_payoff / float(max(1, valid_games))

    os.makedirs("Data", exist_ok=True)
    pickle_name = (
        f"Data/quantum_stats_{agent_x.__class__.__name__}_vs_{agent_o.__class__.__name__}_timestamp_{time.time()}.pkl"
    )

    results_payload = {
        "player_x": agent_x.__class__.__name__,
        "player_o": agent_o.__class__.__name__,
        "num_games": num_games,
        "valid_games": valid_games,
        "failed_games": failed_games,
        "expected_payoff": expected_payoff,
        "games": pd.DataFrame(game_stats),
    }

    pd.to_pickle(results_payload, pickle_name)
    print(f"Saved quantum stats to {pickle_name}")

    return expected_payoff




def main():
    parser = argparse.ArgumentParser(description="Run head-to-head experiments between agents")
    parser.add_argument("--agent-x", choices=["mcts", "minimax", "random"], default="mcts", help="Agent type for X (player 1)")
    parser.add_argument("--agent-o", choices=["mcts", "minimax", "random"], default="mcts", help="Agent type for O (player -1)")
    parser.add_argument("--games", type=int, default=100, help="Number of games to play")
    # MCTS params
    parser.add_argument("--sims-x", type=int, default=200, help="MCTS simulations for X")
    parser.add_argument("--sims-o", type=int, default=200, help="MCTS simulations for O")
    parser.add_argument("--exploration", type=float, default=2.0, help="UCT exploration weight for both MCTS agents")
    parser.add_argument("--horizon", type=int, default=None, help="Rollout horizon for both MCTS agents (None for full)" )
    # Minimax params
    parser.add_argument("--depth-x", type=int, default=None, help="Minimax depth for X")
    parser.add_argument("--depth-o", type=int, default=None, help="Minimax depth for O")
    parser.add_argument("--tiebreak", action="store_true", help="Enable random tiebreak in minimax")

    args = parser.parse_args()

    # Initial board for agent constructors
    init_board = Board.create_initial_board()

    def build_agent(which: str, is_x: bool):
        if which == "mcts":
            sims = args.sims_x if is_x else args.sims_o
            return MCTSAgent(init_board, exploration_weight=args.exploration, num_simulations=sims, rollout_horizon=args.horizon)
        elif which == "minimax":
            depth = args.depth_x if is_x else args.depth_o
            return MinimaxAgent(init_board, max_depth=depth, random_tiebreak=args.tiebreak)
        else:
            return RandomAgent(init_board)

    ax = build_agent(args.agent_x, True)
    ao = build_agent(args.agent_o, False)

    gather_game_stats(ax, ao, num_games=args.games)


def main_quantum():
    parser = argparse.ArgumentParser(
        description="Run head-to-head experiments between quantum agents"
    )
    parser.add_argument(
        "--agent-x",
        choices=["mctsq", "mctsq_grad", "random"],
        default="mctsq",
        help="Quantum agent type for X (player 1)",
    )
    parser.add_argument(
        "--agent-o",
        choices=["mctsq", "mctsq_grad", "random"],
        default="mctsq",
        help="Quantum agent type for O (player -1)",
    )
    parser.add_argument("--games", type=int, default=100, help="Number of games to play")
    parser.add_argument(
        "--sims-x",
        type=int,
        default=200,
        help="MCTSQ simulations for X",
    )
    parser.add_argument(
        "--sims-o",
        type=int,
        default=200,
        help="MCTSQ simulations for O",
    )
    parser.add_argument(
        "--exploration",
        type=float,
        default=2.0,
        help="UCT exploration weight for MCTSQ agents",
    )
    parser.add_argument(
        "--widening",
        type=float,
        default=2.0,
        help="Progressive widening factor for MCTSQ agents",
    )
    parser.add_argument(
        "--grad-lr",
        type=float,
        default=0.1,
        help="Learning rate for gradient-based MCTSQ agents",
    )
    parser.add_argument(
        "--grad-interval",
        type=int,
        default=1,
        help="Simulations between gradient updates in gradient MCTSQ agents",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient norm clip for gradient MCTSQ agents (set negative to disable)",
    )

    args = parser.parse_args()

    def build_agent(which: str, is_x: bool):
        base_env = QuantumGameEnv.create_initial_state()
        if which == "mctsq":
            sims = args.sims_x if is_x else args.sims_o
            return MCTSQAgent(
                base_env,
                exploration_weight=args.exploration,
                num_simulations=sims,
                widening_factor=args.widening,
            )
        if which == "mctsq_grad":
            sims = args.sims_x if is_x else args.sims_o
            clip_value = args.grad_clip if args.grad_clip >= 0 else None
            return MCTSQAgentVanillaGradient(
                base_env,
                exploration_weight=args.exploration,
                num_simulations=sims,
                widening_factor=args.widening,
                learning_rate=args.grad_lr,
                update_every_simulation=max(1, args.grad_interval),
                clip_grad=clip_value,
            )
        else:
            return RandomQAgent(base_env)

    ax = build_agent(args.agent_x, True)
    ao = build_agent(args.agent_o, False)

    expected_payoff = simulate_quantum_game(ax, ao, num_games=args.games)
    print(f"Expected value for X (quantum): {expected_payoff:.3f}")


if __name__ == "__main__":
    if "--quantum" in sys.argv:
        argv = [arg for arg in sys.argv if arg != "--quantum"]
        sys.argv = argv
        main_quantum()
    else:
        main()
