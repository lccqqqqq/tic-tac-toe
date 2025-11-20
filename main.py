"""Training loop for local minimax updates on the simple-board quantum game."""

import argparse
import os
import time
import torch as t
import numpy as np
from dataclasses import dataclass, field
from typing import Any
from environment import GameEnv, QuantumBoardGameEnv, SimpleBoard, Board, TicTacToeBoard
from solver import LocalStochasticMinimaxUpdate, LocalUpdate
from evaluator import Evaluator, _ComplexSphereDistribution
from spaces import ActionSpace, ComplexProjectiveActionSpace
import wandb
from torch.distributions import Distribution
from tqdm import tqdm
from rich.table import Table
from rich.console import Console

@dataclass
class GameSolverConfig:
    game_cls: type[GameEnv] = QuantumBoardGameEnv
    board_cls: type[Board] | None = SimpleBoard
    game_kwargs: dict[str, Any] = field(default_factory=dict)

    save_dir: str = "simpleboard_localalg_xl"
    solver_cls: type[LocalUpdate] = LocalStochasticMinimaxUpdate
    learning_rate: float = 0.001
    max_branching_factor: int = 10

    evaluator_cls: type[Evaluator] = Evaluator
    eval_branching_factor: int = 30
    eval_interval: int = 1000

    action_space: ActionSpace | None = None
    evaluator_dist: Distribution | None = None

    max_iter: int = 100000
    seed: int | None = None

    use_wandb: bool = False
    project_name: str = "quantum-games"
    run_name: str | None = None

    def print_config(self):
        """Print configuration parameters in a rich table."""
        console = Console()
        table = Table(title="Game Solver Configuration")

        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        table.add_column("Type", style="green")

        # Game configuration
        table.add_row("Game Class", self.game_cls.__name__, str(type(self.game_cls)))
        table.add_row("Board Class", self.board_cls.__name__ if self.board_cls else "None", str(type(self.board_cls)))
        table.add_row("Game Kwargs", str(self.game_kwargs), str(type(self.game_kwargs)))
        table.add_row("Save Directory", self.save_dir, str(type(self.save_dir)))

        # Solver configuration
        table.add_row("Solver Class", self.solver_cls.__name__, str(type(self.solver_cls)))
        table.add_row("Learning Rate", str(self.learning_rate), str(type(self.learning_rate)))
        table.add_row("Max Branching Factor", str(self.max_branching_factor), str(type(self.max_branching_factor)))

        # Evaluator configuration
        table.add_row("Evaluator Class", self.evaluator_cls.__name__, str(type(self.evaluator_cls)))
        table.add_row("Eval Branching Factor", str(self.eval_branching_factor), str(type(self.eval_branching_factor)))
        table.add_row("Eval Interval", str(self.eval_interval), str(type(self.eval_interval)))

        # Training configuration
        table.add_row("Max Iterations", str(self.max_iter), str(type(self.max_iter)))
        table.add_row("Random Seed", str(self.seed), str(type(self.seed)))

        # Logging configuration
        table.add_row("Use WandB", str(self.use_wandb), str(type(self.use_wandb)))
        table.add_row("Project Name", self.project_name, str(type(self.project_name)))
        table.add_row("Run Name", str(self.run_name), str(type(self.run_name)))

        console.print(table)
    
    
TICTACTOE_CONFIG = GameSolverConfig(
    game_cls=QuantumBoardGameEnv,
    board_cls=TicTacToeBoard,
    game_kwargs={"board_cls": TicTacToeBoard},
    save_dir="tic_tac_toe_localalg",
    learning_rate=0.002,
    max_branching_factor=2,
    eval_interval=100,
    eval_branching_factor=2,
    max_iter=5000,
    use_wandb=True,
    project_name="quantum-games",
    run_name="tic_tac_toe_localalg",
)


def _build_environment(config: GameSolverConfig) -> GameEnv:
    kwargs = dict(config.game_kwargs)
    if config.board_cls is not None:
        kwargs.setdefault("board_cls", config.board_cls)
    return config.game_cls(**kwargs)


def _default_action_space(env: GameEnv, seed: int | None) -> ActionSpace:
    if hasattr(env, "board_cls"):
        num_sites = env.board_cls.num_sites
        return ComplexProjectiveActionSpace(d_action=num_sites, seed=seed)
    raise ValueError("Unable to infer action space for the provided environment")


def _default_evaluator_distribution(env: GameEnv, seed: int | None):
    if hasattr(env, "board_cls"):
        return _ComplexSphereDistribution(env.board_cls.num_sites, seed=seed)
    raise ValueError("Unable to infer evaluator distribution for the provided environment")


def _rollout_random_terminal_state(env: GameEnv, action_space: ActionSpace) -> GameEnv:
    sampled_env = env
    for _ in range(env.num_stages):
        action = action_space.sample()
        if isinstance(action, t.Tensor):
            action_to_play = action.clone()
        else:
            action_to_play = t.as_tensor(action)
        sampled_env = sampled_env.move(action_to_play)
    return sampled_env


def _clone_action_sequence(state) -> list:
    cloned = []
    for action in state:
        if isinstance(action, t.Tensor):
            cloned.append(action.detach().clone().cpu())
        else:
            cloned.append(t.as_tensor(action).clone())
    return cloned


def _clone_value(value):
    if isinstance(value, t.Tensor):
        return value.detach().clone().cpu()
    return t.tensor(value, dtype=t.float32)


def _save_game_history(game_history: dict[str, list], config: GameSolverConfig) -> str:
    metadata = {
        "seed": config.seed,
        "max_iter": config.max_iter,
        "eval_interval": config.eval_interval,
        "learning_rate": config.learning_rate,
        "max_branching_factor": config.max_branching_factor,
    }
    payload = {"metadata": metadata, "history": game_history}

    os.makedirs(f"Data/{config.save_dir}", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = config.run_name or "run"
    filename = f"Data/{config.save_dir}/game_history_{run_name}_{timestamp}.pt"
    t.save(payload, filename)
    return filename


def main(config: GameSolverConfig):
    # Print configuration table at start
    config.print_config()

    # Generate different random seeds for each component to ensure randomness
    import random
    import time
    import os
    # Use time + process ID + random component to ensure uniqueness across parallel processes
    base_seed = int((time.time() * 1000000 + os.getpid() * 12345) % (2**32 - 1))
    random.seed(base_seed)  # Seed the random module itself
    action_seed = random.randint(0, 2**32 - 1)
    eval_seed = random.randint(0, 2**32 - 1)
    solver_seed = random.randint(0, 2**32 - 1)

    env = _build_environment(config)
    action_space = config.action_space or _default_action_space(env, action_seed)
    env = _rollout_random_terminal_state(env, action_space)

    evaluator_dist = config.evaluator_dist or _default_evaluator_distribution(env, eval_seed)

    updater = config.solver_cls(
        env,
        action_space,
        config.learning_rate,
        config.max_branching_factor,
        seed=solver_seed,
    )

    if config.use_wandb:
        wandb.init(project=config.project_name, name=config.run_name)
    
    game_history = {
        "payoff": [],
        "actions": [],
        "regret": [],
    }
    progress = tqdm(range(config.max_iter), desc="Optimizing", unit="iter", mininterval=30.0)
    for iter in progress:
        env = updater.update()
        game_history["payoff"].append(_clone_value(env.payoff))
        game_history["actions"].append(_clone_action_sequence(env.state))
        regret_value = None

        if isinstance(env.payoff, t.Tensor):
            payoff_scalar = env.payoff[0].item() if env.payoff.ndim > 0 else env.payoff.item()
        else:
            payoff_scalar = float(env.payoff)

        progress_metrics = {"payoff": payoff_scalar}

        if config.use_wandb:
            wandb.log({"payoff": payoff_scalar})
        if iter % config.eval_interval == 0:
            eval_iter_seed = random.randint(0, 2**32 - 1)
            evaluator = config.evaluator_cls(
                env,
                config.eval_branching_factor,
                evaluator_dist,
                seed=eval_iter_seed,
            )
            result = evaluator.evaluate()
            regret_value = _clone_value(result["regret"])
            if isinstance(regret_value, t.Tensor):
                progress_metrics["regret"] = regret_value.mean().item()
            elif regret_value is not None:
                progress_metrics["regret"] = float(regret_value)

            if config.use_wandb:
                if isinstance(result["regret"], t.Tensor):
                    regret_payload = result["regret"].tolist()
                else:
                    regret_payload = float(result["regret"])
                wandb.log({"regret": regret_payload})

        progress.set_postfix(progress_metrics)

        game_history["regret"].append(regret_value)

    save_path = _save_game_history(game_history, config)
    return game_history, save_path


if __name__ == "__main__":
    use_tic_tac_toe = True
    
    
    
    parser = argparse.ArgumentParser(description="Run local minimax updates on the quantum simple-board game")
    parser.add_argument("--max-iter", type=int, default=GameSolverConfig.max_iter, help="Number of optimization iterations")
    parser.add_argument("--learning-rate", type=float, default=GameSolverConfig.learning_rate, help="Local update step size")
    parser.add_argument("--max-branching-factor", type=int, default=GameSolverConfig.max_branching_factor, help="Number of sampled actions per decision node")
    parser.add_argument("--eval-interval", type=int, default=GameSolverConfig.eval_interval, help="Iterations between evaluator regret checks")
    parser.add_argument("--eval-branching-factor", type=int, default=GameSolverConfig.eval_branching_factor, help="Evaluator branching factor")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--project-name", type=str, default=GameSolverConfig.project_name, help="Weights & Biases project name")
    parser.add_argument("--run-name", type=str, default=GameSolverConfig.run_name, help="Weights & Biases run name")
    parser.add_argument("--save-dir", type=str, default=GameSolverConfig.save_dir, help="Directory to save game history")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (omit for stochastic runs)")

    args = parser.parse_args()

    if not use_tic_tac_toe:
        config = GameSolverConfig(
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            save_dir=args.save_dir,
            max_branching_factor=args.max_branching_factor,
            eval_interval=args.eval_interval,
            eval_branching_factor=args.eval_branching_factor,
            use_wandb=args.use_wandb,
            project_name=args.project_name,
            run_name=args.run_name,
            seed=args.seed,
        )
    else:
        print("Warning: overwriting terminal input parameters. Using Tic-Tac-Toe config.")
        config = TICTACTOE_CONFIG

    _, saved_path = main(config)
    print(f"Game history saved to {saved_path}")







    
