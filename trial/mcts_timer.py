from environment_test import GameEnv
from spaces_test import ActionSpace
from solver_test import RandomMinimaxUpdate, LocalStochasticMinimaxUpdate
from typing import Union
import torch as t
from torch import Tensor
from collections import defaultdict
import math
import random
from copy import deepcopy
from multiprocessing import Pool
from time import perf_counter

from datetime import datetime

def logmsg(message, profile=None, logfile=None):
    if not profile:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(logfile, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
        f.flush()

# ----------------------------
# Leaf parallel rollout worker
# ----------------------------

# def _rollout_worker(args):
#     env, action_space = args
#     node = env
#     while not node.terminal:
#         node = node.move(action_space.sample())
#     return node.payoff

def _rollout_worker(args):
    env, action_space = args

    node = env
    initial_depth = len(env.state) if hasattr(env, "state") else 0

    step_count = 0
    move_time_total = 0.0
    sample_time_total = 0.0
    branch_sizes = []

    rollout_start = perf_counter()

    while not node.terminal:
        if hasattr(node, "_state_dict"):
            branch_sizes.append(len(node._state_dict))

        t0 = perf_counter()
        action = action_space.sample()
        sample_time_total += perf_counter() - t0

        t1 = perf_counter()
        node = node.move(action)
        move_time_total += perf_counter() - t1

        step_count += 1

    rollout_time = perf_counter() - rollout_start

    final_branch_size = len(node._state_dict) if hasattr(node, "_state_dict") else 1
    max_branch_size = max(branch_sizes) if branch_sizes else final_branch_size
    avg_branch_size = sum(branch_sizes) / len(branch_sizes) if branch_sizes else float(final_branch_size)

    return {
        "payoff": node.payoff,
        "initial_depth": initial_depth,
        "rollout_time": rollout_time,
        "sample_time_total": sample_time_total,
        "move_time_total": move_time_total,
        "step_count": step_count,
        "final_branch_size": final_branch_size,
        "max_branch_size": max_branch_size,
        "avg_branch_size": avg_branch_size,
    }

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

    def _select_k_leaves(self, k):
        leaves = []
        virtual_visits = defaultdict(int)

        for _ in range(k):
            node = self.env
            abstract_node = ()

            path = []  # track path for virtual updates

            while not node.terminal:
                num_children = len(self.tree[abstract_node])

                # use virtual visits in widening
                effective_visits = self.visits[abstract_node] + virtual_visits[abstract_node]

                if num_children == 0 or num_children < self._widening_threshold(effective_visits):
                    new_abstract = abstract_node + (num_children,)
                    action = self.action_space.sample()
                    new_node = node.move(action)

                    self.tree[abstract_node].append(new_abstract)
                    self.actions[abstract_node].append(action)

                    leaves.append((new_node, new_abstract))

                    # apply virtual visit to path
                    for p in path:
                        virtual_visits[p] += 1

                    break

                # use virtual-aware UCT
                next_abstract = self._uct_select_with_virtual(
                    abstract_node, node, virtual_visits
                )

                action = self.actions[abstract_node][next_abstract[-1]]

                node = node.move(action)
                abstract_node = next_abstract
                path.append(abstract_node)

            else:
                # terminal node case
                leaves.append((node, abstract_node))
                for p in path:
                    virtual_visits[p] += 1

        return leaves
    
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
    
    def _uct_select_with_virtual(self, abstract_node, node, virtual_visits):
        best_score = -float("inf")
        best_children = []

        for child in self.tree[abstract_node]:
            # include virtual visits
            child_visits = self.visits[child] + virtual_visits[child]
            parent_visits = self.visits[abstract_node] + virtual_visits[abstract_node]

            q_value = (self.values[child][node.player] / max(1, child_visits)).item()
            exploration = self.exploration_weight * math.sqrt(
                math.log(max(1, parent_visits)) / max(1, child_visits)
            )
            score = q_value + exploration

            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
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
    
    def run_old(self):
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

    def run(self, k=4, profile=True, logfile="mcts_profile.txt"):
        from multiprocessing import Pool
        from time import perf_counter
        from datetime import datetime

        if k <= 0:
            raise ValueError("k must be a positive integer")

        def logmsg(message):
            if not profile:
                return
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(logfile, "a") as f:
                f.write(f"[{timestamp}] {message}\n")

        if profile:
            with open(logfile, "w") as f:
                f.write("===== MCTS Profiling Log =====\n")

        total_start = perf_counter()

        select_time_total = 0.0
        rollout_pool_time_total = 0.0
        backprop_time_total = 0.0

        worker_rollout_time_total = 0.0
        worker_sample_time_total = 0.0
        worker_move_time_total = 0.0
        worker_step_total = 0
        worker_final_branch_total = 0
        worker_max_branch_global = 0
        worker_avg_branch_accum = 0.0

        depth_rollout_times = {}
        depth_counts = {}
        depth_step_totals = {}
        depth_avg_branch_totals = {}
        depth_max_branch = {}

        total_batches = 0
        total_rollouts = 0

        logmsg(f"starting run num_simulations={self.num_simulations} k={k}")

        with Pool() as pool:
            for batch_idx in range(self.num_simulations):
                select_start = perf_counter()
                leaves = self._select_k_leaves(k)
                dt_select = perf_counter() - select_start
                select_time_total += dt_select

                inputs = [(leaf[0], self.action_space) for leaf in leaves]

                rollout_start = perf_counter()
                results = pool.map(_rollout_worker, inputs)
                dt_rollout_pool = perf_counter() - rollout_start
                rollout_pool_time_total += dt_rollout_pool

                backprop_start = perf_counter()
                for (_, abstract_leaf), result in zip(leaves, results):
                    self._backpropagate(abstract_leaf, result["payoff"])
                dt_backprop = perf_counter() - backprop_start
                backprop_time_total += dt_backprop

                for result in results:
                    worker_rollout_time_total += result["rollout_time"]
                    worker_sample_time_total += result["sample_time_total"]
                    worker_move_time_total += result["move_time_total"]
                    worker_step_total += result["step_count"]
                    worker_final_branch_total += result["final_branch_size"]
                    worker_avg_branch_accum += result["avg_branch_size"]
                    worker_max_branch_global = max(worker_max_branch_global, result["max_branch_size"])

                    d = result["initial_depth"]
                    depth_rollout_times[d] = depth_rollout_times.get(d, 0.0) + result["rollout_time"]
                    depth_counts[d] = depth_counts.get(d, 0) + 1
                    depth_step_totals[d] = depth_step_totals.get(d, 0) + result["step_count"]
                    depth_avg_branch_totals[d] = depth_avg_branch_totals.get(d, 0.0) + result["avg_branch_size"]
                    depth_max_branch[d] = max(depth_max_branch.get(d, 0), result["max_branch_size"])

                total_batches += 1
                total_rollouts += len(leaves)

                logmsg(
                    f"batch={batch_idx + 1} "
                    f"select={dt_select:.6f}s "
                    f"pool_rollout={dt_rollout_pool:.6f}s "
                    f"backprop={dt_backprop:.6f}s "
                    f"rollouts_in_batch={len(leaves)}"
                )

        total_time = perf_counter() - total_start
        overhead_time = max(
            0.0,
            total_time - (select_time_total + rollout_pool_time_total + backprop_time_total),
        )

        if profile:
            with open(logfile, "a") as f:
                f.write("\n===== SUMMARY =====\n")
                f.write(f"num_simulations_batches: {self.num_simulations}\n")
                f.write(f"batch_size_k: {k}\n")
                f.write(f"total_batches: {total_batches}\n")
                f.write(f"total_rollouts: {total_rollouts}\n")
                f.write(f"expected_total_rollouts: {self.num_simulations * k}\n")
                f.write(f"total_time_s: {total_time:.6f}\n")
                f.write(f"select_time_s: {select_time_total:.6f}\n")
                f.write(f"pool_rollout_time_s: {rollout_pool_time_total:.6f}\n")
                f.write(f"backprop_time_s: {backprop_time_total:.6f}\n")
                f.write(f"overhead_time_s: {overhead_time:.6f}\n")
                f.write(f"avg_batch_time_s: {total_time / max(1, total_batches):.6f}\n")
                f.write(f"avg_select_per_batch_s: {select_time_total / max(1, total_batches):.6f}\n")
                f.write(f"avg_pool_rollout_per_batch_s: {rollout_pool_time_total / max(1, total_batches):.6f}\n")
                f.write(f"avg_backprop_per_batch_s: {backprop_time_total / max(1, total_batches):.6f}\n")

                if total_time > 0:
                    f.write(f"select_frac: {select_time_total / total_time:.6f}\n")
                    f.write(f"pool_rollout_frac: {rollout_pool_time_total / total_time:.6f}\n")
                    f.write(f"backprop_frac: {backprop_time_total / total_time:.6f}\n")
                    f.write(f"overhead_frac: {overhead_time / total_time:.6f}\n")

                f.write("\n===== ROLLOUT INTERNALS =====\n")
                f.write(f"worker_rollout_time_sum_s: {worker_rollout_time_total:.6f}\n")
                f.write(f"worker_sample_time_sum_s: {worker_sample_time_total:.6f}\n")
                f.write(f"worker_move_time_sum_s: {worker_move_time_total:.6f}\n")
                f.write(f"worker_total_steps: {worker_step_total}\n")
                f.write(f"avg_steps_per_rollout: {worker_step_total / max(1, total_rollouts):.6f}\n")
                f.write(f"avg_rollout_time_s: {worker_rollout_time_total / max(1, total_rollouts):.6f}\n")
                f.write(f"avg_sample_time_s: {worker_sample_time_total / max(1, total_rollouts):.6f}\n")
                f.write(f"avg_move_time_s: {worker_move_time_total / max(1, total_rollouts):.6f}\n")
                f.write(f"avg_final_branch_size: {worker_final_branch_total / max(1, total_rollouts):.6f}\n")
                f.write(f"avg_branch_size_during_rollout: {worker_avg_branch_accum / max(1, total_rollouts):.6f}\n")
                f.write(f"max_branch_size_seen: {worker_max_branch_global}\n")

                f.write("\n===== DEPTH ANALYSIS =====\n")
                for d in sorted(depth_counts.keys()):
                    count = depth_counts[d]
                    avg_rollout = depth_rollout_times[d] / max(1, count)
                    avg_steps = depth_step_totals[d] / max(1, count)
                    avg_branch = depth_avg_branch_totals[d] / max(1, count)
                    max_branch = depth_max_branch[d]
                    f.write(
                        f"depth={d} "
                        f"count={count} "
                        f"avg_rollout_time_s={avg_rollout:.6f} "
                        f"avg_steps={avg_steps:.6f} "
                        f"avg_branch_size={avg_branch:.6f} "
                        f"max_branch_size={max_branch}\n"
                    )

        logmsg("run completed")

        return {
            "num_simulations_batches": self.num_simulations,
            "batch_size_k": k,
            "total_batches": total_batches,
            "total_rollouts": total_rollouts,
            "total_time": total_time,
            "select_time_total": select_time_total,
            "rollout_pool_time_total": rollout_pool_time_total,
            "backprop_time_total": backprop_time_total,
            "overhead_time": overhead_time,
            "worker_total_steps": worker_step_total,
            "max_branch_size_seen": worker_max_branch_global,
            "depth_counts": depth_counts,
        }
        
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
        from test.solver_test import VanillaGradientUpdate

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
        from test.solver_test import LocalStochasticMinimaxUpdate

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
