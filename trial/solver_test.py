from site import venv
import torch as t
from torch import Tensor
from abc import ABC, abstractmethod
from environment import GameEnv
from spaces import ActionSpace, BoxActionSpace, ComplexProjectiveActionSpace
import math
from copy import deepcopy

# add 10.1.2025, integration with different action spaces

class LocalUpdate(ABC):
    @abstractmethod
    def update(self):
        pass

class VanillaGradientUpdate(LocalUpdate):
    def __init__(self, env: GameEnv, learning_rate: float, clip_grad_norm: float):
        self.env = env
        self.learning_rate = learning_rate
        self.clip_grad_norm = clip_grad_norm

    def update(self):
        assert self.env.terminal, f"The game state is not terminal, unable to gather gradient signals."

        # Handle both list and tuple state types
        state_is_tuple = isinstance(self.env.state, tuple)
        state_list = list(self.env.state) if state_is_tuple else self.env.state

        player_grads = []
        for player in range(self.env.num_players):
            player_grads.append(
                t.autograd.grad(self.env.payoff[player], state_list[player].requires_grad_(True), retain_graph=True)[0]
            )

        with t.no_grad():
            for player in range(self.env.num_players):
                if t.linalg.vector_norm(player_grads[player]) >= self.clip_grad_norm:
                    player_grads[player] = player_grads[player] * (self.clip_grad_norm / t.linalg.vector_norm(player_grads[player]))

                state_list[player] = state_list[player] + self.learning_rate * player_grads[player]

                # Re-normalize for constrained action spaces (e.g., complex unit sphere)
                action_norm = t.linalg.vector_norm(state_list[player])
                if action_norm > 1e-10:  # Avoid division by zero
                    state_list[player] = state_list[player] / action_norm

        # Convert back to tuple if needed
        self.env.state = tuple(state_list) if state_is_tuple else state_list

        # recalculate the payoffs
        self.env.terminal = self.env._is_terminal()
        self.env.payoff = self.env._calculate_payoff()

        return self.env

class LocalMinimaxUpdate(LocalUpdate):
    """
    Implement local updates by considering the entire game tree.
    """
    def __init__(self, env: GameEnv, learning_rate: float, max_branching_factor: int):
        self.env = env
        assert self.env.terminal, f"The game state is not terminal, unable to gather gradient signals."
        self.learning_rate = learning_rate
        self.max_branching_factor = max_branching_factor
    
    def _create_local_game_tree(self):
        # this is the most vanilla method...
        # future: randomize
        self.tree_actions = []
        for stage in range(self.env.num_stages):
            action = self.env.state[stage]
            eye = t.eye(self.env.d_action, dtype=action.dtype, device=action.device)
            candidates = [action]
            for basis in eye:
                plus_action = (action + self.learning_rate * basis).clone()
                minus_action = (action - self.learning_rate * basis).clone()
                if action.requires_grad:
                    plus_action.requires_grad_(True)
                    minus_action.requires_grad_(True)
                candidates.extend([plus_action, minus_action])
            self.tree_actions.append(candidates)
        return self.tree_actions
    
    def _get_stage(self, env: GameEnv):
        return len(env.state)
    
    def _minimax(self, env: GameEnv):
        if env.terminal:
            return env.payoff, env.state
        
        # initialize to extreme values so the first child always wins the comparison
        best_value = None
        best_state = None
        for child_action in self.tree_actions[self._get_stage(env)]:
            child_env = env.move(child_action) # No in-place modifications
            value, state = self._minimax(child_env)
            if best_value is None or value[env.player] >= best_value[env.player]:
                best_value = value
                best_state = state

        return best_value, best_state
    
    def update(self):
        self._create_local_game_tree()
        venv = self.env.create_initial_state() # This should not change existing environment
        value, state = self._minimax(venv)
        
        self.env.state = state
        if hasattr(self.env, "_state_dict"):
            self.env._state_dict = self.env._get_state_dict()

        self.env.terminal = self.env._is_terminal()
        # TODO: update the state dict also!!
        self.env.payoff = self.env._calculate_payoff() if self.env.terminal else None

        return self.env

class LocalStochasticMinimaxUpdate(LocalMinimaxUpdate):
    def __init__(self, env: GameEnv, action_space: ActionSpace, learning_rate: float, max_branching_factor: int, seed: int | None = None):
        super().__init__(env, learning_rate, max_branching_factor)
        self.action_space = action_space
        self.seed = seed
        self.rng = t.Generator().manual_seed(self.seed) if self.seed is not None else t.Generator()
    
    def _create_local_game_tree(self):
        self.tree_actions = []
        # for the quantum game board, the state argument is a dictionary from board to amplitude
        for stage in range(self.env.num_stages):
            action = self.env.state[stage]
            random_actions = self.action_space.sample_local(self.max_branching_factor - 1, action, self.learning_rate)
            candidates = [action]
            # candidates = []
            for rand_action in random_actions:
                if isinstance(rand_action, t.Tensor):
                    candidates.append(rand_action)
                else:
                    candidates.append(t.tensor(rand_action, dtype=action.dtype))
            self.tree_actions.append(candidates)
        return self.tree_actions

class RandomMinimaxUpdate(LocalMinimaxUpdate):
    def __init__(self, env: GameEnv, max_branching_factor: int, seed: int | None = None):
        super().__init__(env, None, max_branching_factor) # setting learning rates to None
        self.env = env
        self.seed = seed
        self.max_branching_factor = max_branching_factor
        self.rng = t.Generator().manual_seed(self.seed) if self.seed is not None else t.Generator()
    
    def _create_local_game_tree(self):
        self.tree_actions = []
        action_lows = t.as_tensor(
            getattr(self.env, "action_lows", [-5.0 for _ in range(self.env.d_action)]),
            dtype=t.float32,
        )
        action_highs = t.as_tensor(
            getattr(self.env, "action_highs", [5.0 for _ in range(self.env.d_action)]),
            dtype=t.float32,
        )
        for stage in range(self.env.num_stages):
            action = self.env.state[stage] # shape (d_action,)
            random_actions = t.rand(
                (self.max_branching_factor - 1, self.env.d_action),
                generator=self.rng,
                dtype=t.float32,
            ) * (action_highs - action_lows) + action_lows

            candidates = [action]
            for rand_action in random_actions:
                candidates.append(rand_action.clone())
            self.tree_actions.append(candidates)
        return self.tree_actions



class Verifier(RandomMinimaxUpdate):
    def __init__(self, env: GameEnv, branching_ratio: int, num_trees: int, seed: int | None = None):
        super().__init__(env, branching_ratio, seed)
        self.num_trees = num_trees
    
    def _update_without_replacement(self):
        original_env = deepcopy(self.env)
        updated_env = super().update()
        self.env = original_env
        return updated_env
    
    def compute_counterfactual_regret(self, reduction: None | str = None):
        """
        Given an environment with actions, this function tries to verify that the path of actions remain an SPNE in a bunch of randomized discretizations of the game tree.
        """
        regrets = []
        updated_envs = []
        for _ in range(self.num_trees):
            updated_env = self._update_without_replacement()
            regrets.append(updated_env.payoff - self.env.payoff)
            updated_envs.append(updated_env)

        if reduction is None:
            return regrets, updated_envs
        elif reduction == "mean":
            return t.stack(regrets).mean(dim=0), updated_envs
        elif reduction == "max":
            max_regret_env = updated_envs[t.argmax(t.stack(regrets))]
            return t.stack(regrets).max(dim=0), max_regret_env
        else:
            raise ValueError(f"Unknown reduction method: {reduction}")
