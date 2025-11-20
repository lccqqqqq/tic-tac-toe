"""Utilities for locally sampling extensive-form games and evaluating regret.

This module provides an :class:`Evaluator` that discretizes a continuous-action
game using user supplied torch distributions, solves the sampled tree with a
minimax search, and reports per-player regret relative to the current
trajectory. It also includes a couple of executable demos covering the loss
game and the quantum simple-board environment, which can serve as lightweight
sanity checks when run as a script.
"""

import torch as t
from environment import GameEnv, LossGame1DEnv, QuantumBoardGameEnv, SimpleBoard, TicTacToeBoard
from torch.distributions import Distribution, Normal
from copy import deepcopy

class Evaluator:
    """Approximate a continuous-action game with random sampling and solve it.

    Args:
        env: Terminal environment representing the current trajectory.
        branching_factor: Number of candidate actions to sample per stage.
        dist: Torch distribution used to draw actions (shape must match env).
        seed: Optional torch RNG seed used to deterministically rebuild trees.
    """
    def __init__(self, env: GameEnv, branching_factor: int, dist: Distribution, seed: int | None = None):
        self.env = deepcopy(env)
        self.branching_factor = branching_factor
        self.dist = dist
        self.seed = seed
        self.tree_actions: list[list[t.Tensor]] | None = None

    def _create_local_game_tree(self):
        """Sample a local game tree rooted at the initial state of the env."""
        num_stages = getattr(self.env, "num_stages", None)
        if num_stages is None:
            raise ValueError("Environment must define num_stages for tree construction")

        # try to infer action specification from existing state or environment metadata
        if len(getattr(self.env, "state", [])) > 0:
            reference_action = self.env.state[0]
            action_shape = tuple(reference_action.shape)
            action_dtype = reference_action.dtype
            action_device = reference_action.device
        elif hasattr(self.env, "d_action"):
            action_shape = (self.env.d_action,)
            action_dtype = t.float32
            action_device = t.device("cpu")
        elif hasattr(self.env, "board_cls"):
            action_shape = (self.env.board_cls.num_sites,)
            action_dtype = t.complex64
            action_device = t.device("cpu")
        else:
            raise ValueError("Unable to infer action specification from environment")

        if self.branching_factor <= 0:
            raise ValueError("Branching factor must be positive")

        original_rng_state = None
        if self.seed is not None:
            original_rng_state = t.random.get_rng_state()
            t.random.manual_seed(self.seed)

        tree_actions: list[list[t.Tensor]] = []

        try:
            for stage in range(num_stages):
                raw_samples = self.dist.sample((self.branching_factor,))

                # handle distributions whose event shape includes stage information
                if raw_samples.dim() == len(action_shape) + 2 and raw_samples.shape[1] == num_stages:
                    stage_samples = raw_samples[:, stage]
                elif raw_samples.dim() == len(action_shape) + 1:
                    stage_samples = raw_samples
                elif raw_samples.dim() == len(action_shape) and self.branching_factor == 1:
                    stage_samples = raw_samples.unsqueeze(0)
                else:
                    raise ValueError(
                        f"Distribution sample shape {tuple(raw_samples.shape)} incompatible with action shape {action_shape}"
                    )

                stage_candidates: list[t.Tensor] = []
                for sample in stage_samples:
                    action_tensor = sample.reshape(action_shape).to(dtype=action_dtype, device=action_device)
                    stage_candidates.append(action_tensor.clone())

                if not stage_candidates:
                    raise ValueError(f"No actions sampled for stage {stage}")

                tree_actions.append(stage_candidates)

        finally:
            if original_rng_state is not None:
                t.random.set_rng_state(original_rng_state)

        self.tree_actions = tree_actions
        return tree_actions

    def _minimax(self):
        """Solve the sampled tree with standard minimax recursion."""
        if self.tree_actions is None:
            raise ValueError("Local game tree has not been constructed")

        def recurse(env: GameEnv):
            if env.terminal:
                return env.payoff, env.state

            stage = len(getattr(env, "state", ()))
            actions_at_stage = self.tree_actions[stage]
            if not actions_at_stage:
                raise RuntimeError(f"No candidate actions stored for stage {stage}")

            best_value = None
            best_state = None

            for action in actions_at_stage:
                child_env = env.move(action.clone()) if isinstance(action, t.Tensor) else env.move(action)
                value, state = recurse(child_env)

                if best_value is None or value[env.player] >= best_value[env.player]:
                    best_value = value
                    best_state = state

            return best_value, best_state

        initial_env = self.env.create_initial_state()
        return recurse(initial_env)

    def evaluate(self):
        """Compute minimax value and regret for the current environment."""
        self._create_local_game_tree()
        minimax_payoff, minimax_state = self._minimax()

        if getattr(self.env, "payoff", None) is not None:
            baseline_payoff = self.env.payoff.clone()
        else:
            baseline_env = self.env.create_initial_state()
            for action in getattr(self.env, "state", ()):  # replay existing trajectory if provided
                baseline_env = baseline_env.move(action)
            if baseline_env.payoff is None:
                raise ValueError("Baseline environment is not terminal; cannot compute regret")
            baseline_payoff = baseline_env.payoff.clone()

        baseline_payoff = baseline_payoff.to(dtype=minimax_payoff.dtype)
        regret = minimax_payoff - baseline_payoff

        return {
            "regret": regret,
            "optimal_payoff": minimax_payoff,
            "baseline_payoff": baseline_payoff,
            "optimal_state": minimax_state,
            "optimal_actions": self._extract_actions(minimax_state),
        }

    def _extract_actions(self, state):
        """Return a list of action tensors extracted from a terminal state."""
        if isinstance(state, tuple) or isinstance(state, list):
            return [action.clone() if isinstance(action, t.Tensor) else action for action in state]
        return state
    

class _ComplexSphereDistribution(Distribution):
    """Samples complex vectors uniformly over the unit sphere."""
    arg_constraints = {}
    has_rsample = False

    def __init__(self, num_sites: int, seed: int | None = None):
        self.num_sites = num_sites
        self.seed = seed
        self._generator = t.Generator().manual_seed(seed) if seed is not None else None
        super().__init__(validate_args=False)

    def sample(self, sample_shape=t.Size()):
        if not isinstance(sample_shape, t.Size):
            sample_shape = t.Size(sample_shape)
        shape = sample_shape + (self.num_sites,)
        if self._generator is None:
            real = t.randn(shape, dtype=t.float32)
            imag = t.randn(shape, dtype=t.float32)
        else:
            real = t.randn(shape, generator=self._generator, dtype=t.float32)
            imag = t.randn(shape, generator=self._generator, dtype=t.float32)
        actions = t.complex(real, imag)
        flat = actions.reshape(-1, self.num_sites)

        # Normalize to unit sphere
        norms = t.linalg.vector_norm(flat, dim=-1, keepdim=True).clamp_min(1e-12)
        normalized = flat / norms

        # Make first component real (complex projective space)
        first_component = normalized[:, 0]
        phases = first_component / first_component.abs().clamp_min(1e-12)
        normalized = normalized * phases.conj().unsqueeze(-1)

        return normalized.reshape(actions.shape).to(dtype=t.complex64)


def _loss_game_demo():
    """Quick demonstration using the 1D loss game environment."""
    env = LossGame1DEnv(num_stages=3, num_players=3, d_action=1)
    staged_env = env.create_initial_state()
    baseline_actions = []
    for _ in range(env.num_stages):
        action = t.randn((env.d_action,), dtype=t.float32)
        baseline_actions.append(action.clone())
        staged_env = staged_env.move(action)

    evaluator = Evaluator(
        env=staged_env,
        branching_factor=5,
        dist=Normal(t.zeros(env.d_action), t.ones(env.d_action)),
        seed=None,
    )
    result = evaluator.evaluate()

    print("Loss game baseline actions:", baseline_actions)
    print("Loss game baseline payoff:", result["baseline_payoff"])
    print("Loss game optimal payoff:", result["optimal_payoff"])
    print("Loss game regret:", result["regret"])


def _quantum_board_demo():
    """Quick demonstration on the quantum simple-board environment."""
    env = QuantumBoardGameEnv(board_cls=SimpleBoard).create_initial_state()
    dist = _ComplexSphereDistribution(SimpleBoard.num_sites, seed=None)
    staged_env = env
    baseline_actions = []
    for _ in range(env.num_stages):
        action = dist.sample()
        baseline_actions.append(action.clone())
        staged_env = staged_env.move(action)

    evaluator = Evaluator(
        env=staged_env,
        branching_factor=4,
        dist=_ComplexSphereDistribution(SimpleBoard.num_sites, seed=None),
        seed=None,
    )
    result = evaluator.evaluate()

    print("Quantum board baseline actions:")
    for action in baseline_actions:
        print(action)
    print("Quantum board baseline payoff:", result["baseline_payoff"])
    print("Quantum board optimal payoff:", result["optimal_payoff"])
    print("Quantum board regret:", result["regret"])


def _tictactoe_demo():
    """Demonstration using the classical 3x3 Tic-Tac-Toe board."""
    env = QuantumBoardGameEnv(board_cls=TicTacToeBoard).create_initial_state()
    dist = _ComplexSphereDistribution(TicTacToeBoard.num_sites, seed=None)

    staged_env = env
    baseline_actions = []
    for _ in range(env.num_stages):
        action = dist.sample()
        baseline_actions.append(action.clone())
        staged_env = staged_env.move(action)

    evaluator = Evaluator(
        env=staged_env,
        branching_factor=2,
        dist=_ComplexSphereDistribution(TicTacToeBoard.num_sites, seed=None),
        seed=None,
    )
    result = evaluator.evaluate()

    print("TicTacToe baseline actions:")
    for action in baseline_actions:
        print(action)
    print("TicTacToe baseline payoff:", result["baseline_payoff"])
    print("TicTacToe optimal payoff:", result["optimal_payoff"])
    print("TicTacToe regret:", result["regret"])


if __name__ == "__main__":
    # _loss_game_demo()
    # _quantum_board_demo()
    _tictactoe_demo()
    # Can you extrapolate the strategy configuration from the discrete set of strategies?
