"""
Comprehensive comparison of game solvers on Loss Game environments.

This module provides a systematic testing framework to compare different solver algorithms
(MCTS variants and local update methods) on both 1D and 3D loss game environments.
"""

import sys
import os
# Add parent directory to path so we can import from root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import pickle
import torch as t
from torch.distributions import Normal

from environment import LossGame1DEnv, LossGame3DEnv
from spaces import BoxActionSpace
from mcts import MCTS, GradientMCTS, MCTSwithLocalUpdate
from solver import LocalStochasticMinimaxUpdate
from evaluator import Evaluator


# =============================================================================
# Configuration
# =============================================================================

# Game configuration
GAME_CONFIGS = {
    "LossGame1D": {
        "env_cls": LossGame1DEnv,
        "env_kwargs": {"num_stages": 3, "num_players": 3, "d_action": 1, "preserve_grad": True},
        "action_space_kwargs": {"d_action": 1, "lows": [-5.0], "highs": [5.0]},
        "eval_dist": Normal(t.zeros(1), t.ones(1)),
    },
    "LossGame3D": {
        "env_cls": LossGame3DEnv,
        "env_kwargs": {"num_stages": 3, "num_players": 3, "d_action": 3, "preserve_grad": True},
        "action_space_kwargs": {"d_action": 3, "lows": [-5.0]*3, "highs": [5.0]*3},
        "eval_dist": Normal(t.zeros(3), t.ones(3)),
    },
}

# Solver configurations
SOLVER_CONFIGS = {
    "MCTS": {
        "simulation_checkpoints": [100, 200, 400, 800, 1600],  # Progressive runs
        "exploration_weight": 1.0,
        "widening_factor": 2.0,
    },
    "GradientMCTS": {
        "simulation_checkpoints": [100, 200, 400, 800, 1600],
        "gradient_update_frequency": 0.25,
        "learning_rate": 1e-3,
        "clip_grad_norm": 1.0,
        "exploration_weight": 1.0,
        "widening_factor": 2.0,
    },
    "MCTSwithLocalUpdate": {
        "simulation_checkpoints": [100, 200, 400, 800, 1600],
        "update_frequency": 0.25,
        "learning_rate": 1e-3,
        "max_branching_factor": 5,
        "exploration_weight": 1.0,
        "widening_factor": 2.0,
    },
    "LocalStochasticMinimaxUpdate": {
        "max_iter": 2000,
        "eval_interval": 100,
        "learning_rate": 1e-3,
        "max_branching_factor": 5,
    },
}

# Evaluation configuration
EVAL_CONFIG = {
    "branching_factor": 10,
    "seed": 42,
}


# =============================================================================
# Helper Functions
# =============================================================================

def _rollout_random_terminal_state(env, action_space, seed):
    """Rollout random actions to reach terminal state.

    Args:
        env: Environment instance (should be at initial state)
        action_space: Action space for sampling
        seed: Random seed

    Returns:
        Terminal environment state
    """
    current_env = env.create_initial_state()
    while not current_env.terminal:
        action = action_space.sample()
        current_env = current_env.move(action)
    return current_env


def _get_best_terminal_path(mcts, target_depth):
    """Find the best complete path through the MCTS tree to a terminal node.

    Follows the most visited children from root until reaching target depth.

    Args:
        mcts: MCTS instance with completed search
        target_depth: Depth to reach (typically num_stages for the game)

    Returns:
        Tuple representing the complete path (e.g., (2, 1, 4) for 3 stages)
    """
    current_abstract = ()

    for depth in range(target_depth):
        # Get children of current node
        if current_abstract not in mcts.tree or len(mcts.tree[current_abstract]) == 0:
            raise ValueError(f"No children found at depth {depth}. Tree may not be deep enough.")

        # Select most visited child
        children = mcts.tree[current_abstract]
        best_child = max(children, key=lambda child: mcts.visits[child])
        current_abstract = best_child

    return current_abstract


def _reconstruct_terminal_from_tree_path(root_env, mcts, abstract_node):
    """Reconstruct terminal environment from MCTS tree path.

    Args:
        root_env: Root environment (initial state)
        mcts: MCTS instance with completed search
        abstract_node: Abstract node tuple representing path through tree

    Returns:
        Terminal environment state reached by following the path
    """
    actions = []
    for depth in range(len(abstract_node)):
        parent_abstract = abstract_node[:depth]
        child_idx = abstract_node[depth]
        action = mcts.actions[parent_abstract][child_idx]
        actions.append(action)

    env = root_env.create_initial_state()
    for action in actions:
        env = env.move(action)
    return env


# =============================================================================
# Solver Runner Functions
# =============================================================================

def run_mcts_progressive(game_config, solver_config, seed=42):
    """Run MCTS with progressive simulation counts.

    Args:
        game_config: Dictionary with game configuration
        solver_config: Dictionary with solver configuration
        seed: Random seed

    Returns:
        List of result dictionaries, one per checkpoint
    """
    results = []

    for num_sims in solver_config["simulation_checkpoints"]:
        print(f"  Running MCTS with {num_sims} simulations...")

        # Create fresh environment and action space
        env = game_config["env_cls"](**game_config["env_kwargs"])
        action_space = BoxActionSpace(**game_config["action_space_kwargs"], seed=seed)

        # Initialize and run MCTS
        mcts = MCTS(
            env, action_space,
            exploration_weight=solver_config["exploration_weight"],
            widening_factor=solver_config["widening_factor"],
            num_simulations=num_sims,
            seed=seed
        )

        start_time = time.time()
        mcts.run()
        elapsed = time.time() - start_time

        # Get best complete path to terminal node
        best_terminal_path = _get_best_terminal_path(mcts, env.num_stages)

        # Reconstruct terminal environment from tree path
        terminal_env = _reconstruct_terminal_from_tree_path(env, mcts, best_terminal_path)

        # Evaluate regret
        evaluator = Evaluator(
            terminal_env,
            EVAL_CONFIG["branching_factor"],
            game_config["eval_dist"],
            seed=EVAL_CONFIG["seed"]
        )
        eval_result = evaluator.evaluate()

        results.append({
            "iteration": num_sims,
            "payoff": terminal_env.payoff.clone(),
            "regret": eval_result["regret"].clone(),
            "time": elapsed,
        })

        print(f"    Completed in {elapsed:.2f}s - Payoff (P0): {terminal_env.payoff[0].item():.4f}, Regret (P0): {eval_result['regret'][0].item():.4f}")

    return results


def run_gradient_mcts_progressive(game_config, solver_config, seed=42):
    """Run GradientMCTS with progressive simulation counts.

    Args:
        game_config: Dictionary with game configuration
        solver_config: Dictionary with solver configuration
        seed: Random seed

    Returns:
        List of result dictionaries, one per checkpoint
    """
    results = []

    for num_sims in solver_config["simulation_checkpoints"]:
        print(f"  Running GradientMCTS with {num_sims} simulations...")

        # Create fresh environment and action space
        env = game_config["env_cls"](**game_config["env_kwargs"])
        action_space = BoxActionSpace(**game_config["action_space_kwargs"], seed=seed)

        # Initialize and run GradientMCTS
        mcts = GradientMCTS(
            env, action_space,
            gradient_update_frequency=solver_config["gradient_update_frequency"],
            learning_rate=solver_config["learning_rate"],
            clip_grad_norm=solver_config["clip_grad_norm"],
            exploration_weight=solver_config["exploration_weight"],
            widening_factor=solver_config["widening_factor"],
            num_simulations=num_sims,
            seed=seed
        )

        start_time = time.time()
        mcts.run()
        elapsed = time.time() - start_time

        # Get best complete path to terminal node
        best_terminal_path = _get_best_terminal_path(mcts, env.num_stages)

        # Reconstruct terminal environment from tree path
        terminal_env = _reconstruct_terminal_from_tree_path(env, mcts, best_terminal_path)

        # Evaluate regret
        evaluator = Evaluator(
            terminal_env,
            EVAL_CONFIG["branching_factor"],
            game_config["eval_dist"],
            seed=EVAL_CONFIG["seed"]
        )
        eval_result = evaluator.evaluate()

        results.append({
            "iteration": num_sims,
            "payoff": terminal_env.payoff.clone(),
            "regret": eval_result["regret"].clone(),
            "time": elapsed,
        })

        print(f"    Completed in {elapsed:.2f}s - Payoff (P0): {terminal_env.payoff[0].item():.4f}, Regret (P0): {eval_result['regret'][0].item():.4f}")

    return results


def run_mcts_local_update_progressive(game_config, solver_config, seed=42):
    """Run MCTSwithLocalUpdate with progressive simulation counts.

    Args:
        game_config: Dictionary with game configuration
        solver_config: Dictionary with solver configuration
        seed: Random seed

    Returns:
        List of result dictionaries, one per checkpoint
    """
    results = []

    for num_sims in solver_config["simulation_checkpoints"]:
        print(f"  Running MCTSwithLocalUpdate with {num_sims} simulations...")

        # Create fresh environment and action space
        env = game_config["env_cls"](**game_config["env_kwargs"])
        action_space = BoxActionSpace(**game_config["action_space_kwargs"], seed=seed)

        # Initialize and run MCTSwithLocalUpdate
        mcts = MCTSwithLocalUpdate(
            env, action_space,
            update_frequency=solver_config["update_frequency"],
            learning_rate=solver_config["learning_rate"],
            max_branching_factor=solver_config["max_branching_factor"],
            exploration_weight=solver_config["exploration_weight"],
            widening_factor=solver_config["widening_factor"],
            num_simulations=num_sims,
            seed=seed
        )

        start_time = time.time()
        mcts.run()
        elapsed = time.time() - start_time

        # Get best complete path to terminal node
        best_terminal_path = _get_best_terminal_path(mcts, env.num_stages)

        # Reconstruct terminal environment from tree path
        terminal_env = _reconstruct_terminal_from_tree_path(env, mcts, best_terminal_path)

        # Evaluate regret
        evaluator = Evaluator(
            terminal_env,
            EVAL_CONFIG["branching_factor"],
            game_config["eval_dist"],
            seed=EVAL_CONFIG["seed"]
        )
        eval_result = evaluator.evaluate()

        results.append({
            "iteration": num_sims,
            "payoff": terminal_env.payoff.clone(),
            "regret": eval_result["regret"].clone(),
            "time": elapsed,
        })

        print(f"    Completed in {elapsed:.2f}s - Payoff (P0): {terminal_env.payoff[0].item():.4f}, Regret (P0): {eval_result['regret'][0].item():.4f}")

    return results


def run_local_update_iterative(game_config, solver_config, seed=42):
    """Run standalone LocalStochasticMinimaxUpdate iteratively.

    Similar to the pattern in main.py but adapted for loss game.

    Args:
        game_config: Dictionary with game configuration
        solver_config: Dictionary with solver configuration
        seed: Random seed

    Returns:
        List of result dictionaries, one per evaluation checkpoint
    """
    print(f"  Running LocalStochasticMinimaxUpdate for {solver_config['max_iter']} iterations...")

    # Create environment and action space
    env = game_config["env_cls"](**game_config["env_kwargs"])
    action_space = BoxActionSpace(**game_config["action_space_kwargs"], seed=seed)

    # Rollout random terminal state as starting point
    env = _rollout_random_terminal_state(env, action_space, seed)

    # Initialize updater
    updater = LocalStochasticMinimaxUpdate(
        env, action_space,
        solver_config["learning_rate"],
        solver_config["max_branching_factor"],
        seed=seed
    )

    results = []
    start_time = time.time()

    for iter_num in range(solver_config["max_iter"]):
        # Update environment
        env = updater.update()

        # Evaluate at intervals
        if iter_num % solver_config["eval_interval"] == 0:
            evaluator = Evaluator(
                env,
                EVAL_CONFIG["branching_factor"],
                game_config["eval_dist"],
                seed=EVAL_CONFIG["seed"]
            )
            eval_result = evaluator.evaluate()

            elapsed = time.time() - start_time

            results.append({
                "iteration": iter_num,
                "payoff": env.payoff.clone(),
                "regret": eval_result["regret"].clone(),
                "time": elapsed,
            })

            print(f"    Iter {iter_num}/{solver_config['max_iter']} - Payoff (P0): {env.payoff[0].item():.4f}, Regret (P0): {eval_result['regret'][0].item():.4f}")

    # Final evaluation if not already done
    if (solver_config["max_iter"] - 1) % solver_config["eval_interval"] != 0:
        evaluator = Evaluator(
            env,
            EVAL_CONFIG["branching_factor"],
            game_config["eval_dist"],
            seed=EVAL_CONFIG["seed"]
        )
        eval_result = evaluator.evaluate()

        elapsed = time.time() - start_time

        results.append({
            "iteration": solver_config["max_iter"] - 1,
            "payoff": env.payoff.clone(),
            "regret": eval_result["regret"].clone(),
            "time": elapsed,
        })

        print(f"    Final - Payoff (P0): {env.payoff[0].item():.4f}, Regret (P0): {eval_result['regret'][0].item():.4f}")

    return results


# =============================================================================
# Comparison Framework
# =============================================================================

def compare_solvers_on_game(game_name, solvers_to_run=None, seed=42):
    """Run all specified solvers on a game and collect results.

    Args:
        game_name: Name of game from GAME_CONFIGS
        solvers_to_run: List of solver names, or None for all
        seed: Random seed

    Returns:
        Dictionary mapping solver names to their result lists
    """
    if solvers_to_run is None:
        solvers_to_run = ["MCTS", "GradientMCTS", "MCTSwithLocalUpdate", "LocalStochasticMinimaxUpdate"]

    game_config = GAME_CONFIGS[game_name]
    results = {}

    for solver_name in solvers_to_run:
        print(f"\nRunning {solver_name} on {game_name}...")
        solver_config = SOLVER_CONFIGS[solver_name]

        if solver_name == "LocalStochasticMinimaxUpdate":
            results[solver_name] = run_local_update_iterative(game_config, solver_config, seed)
        elif solver_name == "MCTS":
            results[solver_name] = run_mcts_progressive(game_config, solver_config, seed)
        elif solver_name == "GradientMCTS":
            results[solver_name] = run_gradient_mcts_progressive(game_config, solver_config, seed)
        elif solver_name == "MCTSwithLocalUpdate":
            results[solver_name] = run_mcts_local_update_progressive(game_config, solver_config, seed)
        else:
            raise ValueError(f"Unknown solver: {solver_name}")

    return results


def save_results(results, game_name, timestamp=None):
    """Save results to disk.

    Args:
        results: Dictionary of solver results
        game_name: Name of game
        timestamp: Optional timestamp string, generated if not provided

    Returns:
        Path to saved file
    """
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

    os.makedirs("Data/loss_game_comparison", exist_ok=True)
    filename = f"Data/loss_game_comparison/{game_name}_{timestamp}.pkl"

    payload = {
        "game_name": game_name,
        "timestamp": timestamp,
        "results": results,
        "game_config": GAME_CONFIGS[game_name],
        "solver_configs": SOLVER_CONFIGS,
        "eval_config": EVAL_CONFIG,
    }

    with open(filename, "wb") as f:
        pickle.dump(payload, f)

    print(f"\nResults saved to {filename}")
    return filename


# =============================================================================
# Main Entry Point for Direct Execution
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test game solvers on loss games")
    parser.add_argument("--game", choices=["LossGame1D", "LossGame3D"], default="LossGame1D",
                       help="Which game to test")
    parser.add_argument("--solvers", nargs="+",
                       choices=["MCTS", "GradientMCTS", "MCTSwithLocalUpdate", "LocalStochasticMinimaxUpdate"],
                       default=None, help="Which solvers to run (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print(f"{'='*80}")
    print(f"Running solver comparison on {args.game}")
    print(f"{'='*80}")

    results = compare_solvers_on_game(args.game, args.solvers, seed=args.seed)
    filename = save_results(results, args.game)

    print(f"\n{'='*80}")
    print(f"Experiment complete! Results saved to: {filename}")
    print(f"{'='*80}")
