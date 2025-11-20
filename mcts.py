from environment import GameEnv
from spaces import ActionSpace
from solver import RandomMinimaxUpdate, LocalStochasticMinimaxUpdate
from typing import Union
import torch as t
from torch import Tensor
from collections import defaultdict
import math
import random
from copy import deepcopy

class MCTS:

    NodeType = GameEnv
    AbstractNodeType = tuple[int]

    def __init__(
        self,
        env: GameEnv,
        action_space: ActionSpace,
        exploration_weight: float = 1.0,
        widening_factor: float = 2.0,
        num_simulations: int = 100,
        seed: int | None = None,
        show_progress: bool = False,
    ):
        self.env = env
        self.exploration_weight = exploration_weight
        self.widening_factor = widening_factor
        self.num_simulations = num_simulations
        self.seed = seed
        self.rng = t.Generator().manual_seed(self.seed) if self.seed is not None else t.Generator()
        self.action_space = action_space
        self.show_progress = show_progress

        # Dynamic variables
        self.tree: dict[MCTS.AbstractNodeType, list[MCTS.AbstractNodeType]] = defaultdict(list)
        self.visits: dict[MCTS.AbstractNodeType, int] = defaultdict(int)
        self.values: dict[MCTS.AbstractNodeType, Tensor] = defaultdict(lambda: t.zeros((self.env.num_players,), dtype=t.float32))
        self.actions: dict[MCTS.AbstractNodeType, list[Tensor]] = defaultdict(list)

    def _get_tree_path_actions(self, abstract_node: AbstractNodeType) -> list[Tensor]:
        """Extract actions along the tree path to this node."""
        actions = []
        for depth in range(len(abstract_node)):
            parent_abstract = abstract_node[:depth]
            child_idx = abstract_node[depth]
            action = self.actions[parent_abstract][child_idx]
            actions.append(action)
        return actions
    
    def _widening_threshold(self, n_parent: int) -> int:
        """
        Given a node, if it's number of children is less than the computed threshold, we need to expand a new child at the node
        """
        return max(1, math.floor(self.widening_factor * math.sqrt(max(0, n_parent))))
    
    def _reset(self):
        """
        Reset the dynamical variables
        """
        self.tree = defaultdict(list)
        self.visits = defaultdict(int)
        self.values = defaultdict(lambda: t.zeros((self.env.num_players,), dtype=t.float32))
        self.actions = defaultdict(list)
    
    def _tree_traversal(self):
        """
        Traverse the existing tree, with progressive widening for continuous action spaces.
        """

        # Initialize the tree traversal data
        current_node: GameEnv = self.env
        current_abstract_node: MCTS.AbstractNodeType = ()

        while not current_node.terminal:
            # Check progressive widening criteria
            num_children = len(self.tree[current_abstract_node])
            if num_children == 0 or num_children < self._widening_threshold(self.visits[current_abstract_node]):
                new_abstract_node = current_abstract_node + (num_children,)
                # The action label should be the same as the number of children
                new_action = self.action_space.sample()
                new_node = current_node.move(new_action)

                # expand
                self.tree[current_abstract_node].append(new_abstract_node)
                self.actions[current_abstract_node].append(new_action)

                return new_node, new_abstract_node

            # select a child
            next_abstract_node = self._uct_select(current_abstract_node, current_node)
            next_node = current_node.move(self.actions[current_abstract_node][next_abstract_node[-1]])

            current_node = next_node
            current_abstract_node = next_abstract_node
        
        return current_node, current_abstract_node
    
    def _uct_select(self, abstract_node: AbstractNodeType, node: NodeType):
        """
        Select a child as argmax UCT
        """
        best_score = -float("inf")
        best_children = []

        for child in self.tree[abstract_node]:
            q_value = (self.values[child][node.player] / max(1, self.visits[child])).item()
            exploration_term = self.exploration_weight * math.sqrt(math.log(self.visits[abstract_node]) / max(1, self.visits[child]))
            uct_score = q_value + exploration_term
            if uct_score > best_score:
                best_score = uct_score
                best_children = [child]
            elif uct_score == best_score:
                best_children.append(child)

        return random.choice(best_children)
    
    def _rollout(self, node: NodeType):
        current_node = node
        while not current_node.terminal:
            current_node = current_node.move(self.action_space.sample())

        return current_node.payoff
    
    def _backpropagate(self, abstract_node: AbstractNodeType, value: Tensor):
        for i in range(len(abstract_node) + 1):
            abstract_node_i = abstract_node[:i]
            self.visits[abstract_node_i] += 1
            self.values[abstract_node_i] += value
    
    def run(self):
        iterator = range(self.num_simulations)
        if self.show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="MCTS simulations")
            except ImportError:
                pass  # Fall back to no progress bar if tqdm not installed

        for _ in iterator:
            leaf_node, abstract_leaf_node = self._tree_traversal()
            # This leaf is either itself a terminal node or a leaf node.
            value = self._rollout(leaf_node)
            self._backpropagate(abstract_leaf_node, value)
        
    def select_action(self):
        most_visited_child = max(self.tree[()], key=lambda child: self.visits[child])
        return most_visited_child, self.actions[()][most_visited_child[-1]]
    
    def reroot(self, new_abstract_root: AbstractNodeType, new_root: NodeType):
        self.env = new_root

        # Copying the subtree data
        new_tree = defaultdict(list)
        new_visits = defaultdict(int)
        new_values = defaultdict(lambda: t.zeros((self.env.num_players,), dtype=t.float32))
        new_actions = defaultdict(list)

        for abstract_node in self.tree.keys():
            if abstract_node == ():
                continue

            if abstract_node[0] == new_abstract_root[0]:
                new_tree[abstract_node[1:]] = [child[1:] for child in self.tree[abstract_node]]
                new_visits[abstract_node[1:]] = self.visits[abstract_node]
                new_values[abstract_node[1:]] = self.values[abstract_node]
                new_actions[abstract_node[1:]] = self.actions[abstract_node]
        
        self.tree = new_tree
        self.visits = new_visits
        self.values = new_values
        self.actions = new_actions


class GradientMCTS(MCTS):
    """
    MCTS with Variational Gradient updates (VG-MCTS).
    Applies gradient-based improvements to actions in the tree using rollout trajectories.
    """

    def __init__(
        self,
        env: GameEnv,
        action_space: ActionSpace,
        gradient_update_frequency: float = 0.25,
        learning_rate: float = 1e-4,
        clip_grad_norm: float = 1.0,
        exploration_weight: float = 1.0,
        widening_factor: float = 2.0,
        num_simulations: int = 100,
        seed: int | None = None,
        show_progress: bool = False,
    ):
        super().__init__(
            env=env,
            action_space=action_space,
            exploration_weight=exploration_weight,
            widening_factor=widening_factor,
            num_simulations=num_simulations,
            seed=seed,
            show_progress=show_progress,
        )
        self.gradient_update_frequency = gradient_update_frequency
        self.learning_rate = learning_rate
        self.clip_grad_norm = clip_grad_norm

    def _should_apply_gradient_update(self) -> int:
        """
        Determine how many gradient updates to apply.
        Returns: 0 (skip), 1 (single), or >1 (multiple updates)
        """
        if self.gradient_update_frequency == 0:
            return 0
        elif self.gradient_update_frequency >= 1:
            return int(math.ceil(self.gradient_update_frequency))
        else:  # 0 < freq < 1
            return 1 if random.random() < self.gradient_update_frequency else 0

    def _rollout(self, leaf_node: GameEnv, abstract_leaf: MCTS.AbstractNodeType) -> tuple[Tensor, GameEnv]:
        """
        Perform rollout from leaf to terminal, reconstructing full trajectory from root.
        Returns: (payoff, full_terminal_env)
        """
        # Build complete environment from root
        full_env = self.env.create_initial_state()

        # Collect all actions for the full trajectory
        all_actions = []

        # First, replay tree path actions
        tree_actions = self._get_tree_path_actions(abstract_leaf)
        all_actions.extend(tree_actions)

        # Build environment up to leaf
        for action in tree_actions:
            full_env = full_env.move(action)

        # Continue rollout to terminal, collecting actions
        while not full_env.terminal:
            action = self.action_space.sample()
            all_actions.append(action)
            full_env = full_env.move(action)

        # Reconstruct environment with gradient-enabled actions for gradient updates
        if self.gradient_update_frequency > 0:
            full_env_grad = self.env.create_initial_state()
            for action in all_actions:
                # Ensure actions require gradients
                action_grad = action.detach().requires_grad_(True)
                full_env_grad = full_env_grad.move(action_grad)
            return full_env.payoff, full_env_grad

        return full_env.payoff, full_env

    def _reconstruct_with_gradients(self, abstract_leaf: MCTS.AbstractNodeType, terminal_env: GameEnv) -> GameEnv:
        """
        Reconstruct the full trajectory with gradient-enabled actions.
        Uses updated tree actions and rollout actions from terminal environment.
        """
        # Get updated tree actions from the MCTS tree
        tree_actions = self._get_tree_path_actions(abstract_leaf)

        # Get rollout actions from terminal environment (everything after tree portion)
        num_tree_actions = len(tree_actions)
        rollout_actions = list(terminal_env.state[num_tree_actions:])

        # Combine tree actions (potentially updated) with rollout actions
        all_actions = tree_actions + rollout_actions

        # Reconstruct environment with gradient-enabled actions
        full_env_grad = self.env.create_initial_state()
        for action in all_actions:
            action_grad = action.detach().requires_grad_(True)
            full_env_grad = full_env_grad.move(action_grad)

        return full_env_grad

    def _apply_gradient_update(
        self, abstract_leaf: MCTS.AbstractNodeType, terminal_env: GameEnv
    ) -> Tensor:
        """
        Apply gradient update to actions in the tree using VanillaGradientUpdate.
        Uses full trajectory for gradient computation, but only updates tree actions.
        Returns: Updated payoff
        """
        from solver import VanillaGradientUpdate

        assert terminal_env.terminal, "Can only compute gradients on terminal states"

        # Apply gradient update to full trajectory
        updater = VanillaGradientUpdate(terminal_env, self.learning_rate, self.clip_grad_norm)
        updated_env = updater.update()

        # Update only the tree portion of actions (first len(abstract_leaf) actions)
        num_tree_actions = len(abstract_leaf)
        for depth in range(num_tree_actions):
            parent_abstract = abstract_leaf[:depth]
            child_idx = abstract_leaf[depth]
            self.actions[parent_abstract][child_idx] = updated_env.state[depth].detach()

        # Return detached payoff to avoid gradient tracking in tree values
        return updated_env.payoff.detach() if isinstance(updated_env.payoff, Tensor) and updated_env.payoff.requires_grad else updated_env.payoff

    def run(self):
        """Run MCTS with gradient updates."""
        iterator = range(self.num_simulations)
        if self.show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="MCTS simulations")
            except ImportError:
                pass  # Fall back if tqdm not available

        for _ in iterator:
            leaf_node, abstract_leaf = self._tree_traversal()

            # Rollout to terminal with full trajectory reconstruction
            terminal_payoff, terminal_env = self._rollout(leaf_node, abstract_leaf)

            # Apply gradient updates if enabled
            num_updates = self._should_apply_gradient_update()
            for _ in range(num_updates):
                # Reconstruct gradient-enabled environment for each update
                # (computational graph is consumed after each gradient computation)
                updated_terminal_env = self._reconstruct_with_gradients(abstract_leaf, terminal_env)
                terminal_payoff = self._apply_gradient_update(abstract_leaf, updated_terminal_env)

            # Backpropagate with (potentially improved) payoff
            self._backpropagate(abstract_leaf, terminal_payoff)


class MCTSwithLocalUpdate(MCTS):
    """
    MCTS with Local Stochastic Minimax updates.
    Applies local minimax optimization to actions in the tree using rollout trajectories.
    """

    def __init__(
        self,
        env: GameEnv,
        action_space: ActionSpace,
        update_frequency: float = 0.25,
        learning_rate: float = 1e-4,
        max_branching_factor: int = 5,
        exploration_weight: float = 1.0,
        widening_factor: float = 2.0,
        num_simulations: int = 100,
        seed: int | None = None,
        show_progress: bool = False,
    ):
        super().__init__(
            env=env,
            action_space=action_space,
            exploration_weight=exploration_weight,
            widening_factor=widening_factor,
            num_simulations=num_simulations,
            seed=seed,
            show_progress=show_progress,
        )
        self.update_frequency = update_frequency
        self.learning_rate = learning_rate
        self.max_branching_factor = max_branching_factor

    def _should_apply_local_update(self) -> int:
        """
        Determine how many local updates to apply.
        Returns: 0 (skip), 1 (single), or >1 (multiple updates)
        """
        if self.update_frequency == 0:
            return 0
        elif self.update_frequency >= 1:
            return int(math.ceil(self.update_frequency))
        else:  # 0 < freq < 1
            return 1 if random.random() < self.update_frequency else 0

    def _rollout(self, leaf_node: GameEnv, abstract_leaf: MCTS.AbstractNodeType) -> tuple[Tensor, GameEnv]:
        """
        Perform rollout from leaf to terminal, reconstructing full trajectory from root.
        Returns: (payoff, full_terminal_env)
        """
        # Build complete environment from root
        full_env = self.env.create_initial_state()

        # Collect all actions for the full trajectory
        all_actions = []

        # First, replay tree path actions
        tree_actions = self._get_tree_path_actions(abstract_leaf)
        all_actions.extend(tree_actions)

        # Build environment up to leaf
        for action in tree_actions:
            full_env = full_env.move(action)

        # Continue rollout to terminal, collecting actions
        while not full_env.terminal:
            action = self.action_space.sample()
            all_actions.append(action)
            full_env = full_env.move(action)

        # Reconstruct full trajectory environment
        if self.update_frequency > 0:
            full_env_complete = self.env.create_initial_state()
            for action in all_actions:
                full_env_complete = full_env_complete.move(action)
            return full_env.payoff, full_env_complete

        return full_env.payoff, full_env

    def _reconstruct_terminal_env(self, abstract_leaf: MCTS.AbstractNodeType, terminal_env: GameEnv) -> GameEnv:
        """
        Reconstruct the full trajectory environment.
        Uses updated tree actions and rollout actions from terminal environment.
        """
        # Get updated tree actions from the MCTS tree
        tree_actions = self._get_tree_path_actions(abstract_leaf)

        # Get rollout actions from terminal environment (everything after tree portion)
        num_tree_actions = len(tree_actions)
        rollout_actions = list(terminal_env.state[num_tree_actions:])

        # Combine tree actions (potentially updated) with rollout actions
        all_actions = tree_actions + rollout_actions

        # Reconstruct environment
        full_env = self.env.create_initial_state()
        for action in all_actions:
            full_env = full_env.move(action)

        return full_env

    def _apply_local_update(
        self, abstract_leaf: MCTS.AbstractNodeType, terminal_env: GameEnv
    ) -> Tensor:
        """
        Apply local stochastic minimax update to actions in the tree.
        Uses full trajectory for optimization, but only updates tree actions.
        Returns: Updated payoff
        """
        from solver import LocalStochasticMinimaxUpdate

        assert terminal_env.terminal, "Can only apply local updates on terminal states"

        # Apply local stochastic minimax update to full trajectory
        updater = LocalStochasticMinimaxUpdate(
            terminal_env,
            self.action_space,
            self.learning_rate,
            self.max_branching_factor,
            self.seed
        )
        updated_env = updater.update()

        # Update only the tree portion of actions (first len(abstract_leaf) actions)
        num_tree_actions = len(abstract_leaf)
        for depth in range(num_tree_actions):
            parent_abstract = abstract_leaf[:depth]
            child_idx = abstract_leaf[depth]
            self.actions[parent_abstract][child_idx] = updated_env.state[depth].detach() if isinstance(updated_env.state[depth], Tensor) else updated_env.state[depth]

        # Return payoff
        return updated_env.payoff.detach() if isinstance(updated_env.payoff, Tensor) and updated_env.payoff.requires_grad else updated_env.payoff

    def run(self):
        """Run MCTS with local stochastic minimax updates."""
        iterator = range(self.num_simulations)
        if self.show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="MCTS simulations")
            except ImportError:
                pass  # Fall back if tqdm not available

        for _ in iterator:
            leaf_node, abstract_leaf = self._tree_traversal()

            # Rollout to terminal with full trajectory reconstruction
            terminal_payoff, terminal_env = self._rollout(leaf_node, abstract_leaf)

            # Apply local updates if enabled
            num_updates = self._should_apply_local_update()
            for _ in range(num_updates):
                # Reconstruct terminal environment for each update
                updated_terminal_env = self._reconstruct_terminal_env(abstract_leaf, terminal_env)
                terminal_payoff = self._apply_local_update(abstract_leaf, updated_terminal_env)

            # Backpropagate with (potentially improved) payoff
            self._backpropagate(abstract_leaf, terminal_payoff)
