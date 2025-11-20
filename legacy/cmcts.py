## An implementation of MCTS algorithm
# MCTS proceeds on an abstract game tree
# The abstract game tree can be mapped to an actual game class where various properties of the game, such as payoff assignement and game states are defined customly
# The continuous action space is discretized into a finite number of actions -> the need to sample with low discrepancy, especially in high dimensions

# The VG-MCTS algorithm updates a particular branch each time the path is visited. We would like to test the following:

# 1. VG-MCTS performs better than the standard MCTS algorithm
# 2. The VG-MCTS with differential rollback algorithm performs better than the vanilla VG_MCTS where the players update the actions based solely on its local gradient.

from games import *
from DBI import *
from typing import Union

scaling = lambda x: 2 * x ** (1 / 2)  # criteria of progressive widening
gradient_update_frequency = (
    1 / 4
)  # relative frequency between updating the gradient and the tree search simulation
learning_rate = 1e-2


class cMCTS:
    def __init__(
        self,
        node: AbstractGameNode,
        agent: GameRealization,
        # meta_data: GameMetaData,
        exploration_weight: float = 1,
        gradient_mode="dbi",
        gradient_update_frequency: float = 1 / 4,
        learning_rate: float = 1e-2,
    ):

        self.node = node
        # self.meta_data = meta_data
        self.exploration_weight = exploration_weight
        self.gradient_update_frequency = gradient_update_frequency
        self.learning_rate = learning_rate
        self.tree = defaultdict(set)
        self.tree[node] = set()

        self.values = defaultdict(float)
        self.visit_counts = defaultdict(int)

        self.agent = agent  # the realizer of the game

        # Use the gradient-based agent implementing Differential Backward Propagation
        self.gradient_mode = gradient_mode
        self.grad_agent = DiffBP(
            game=QuantumSimpleBoard(),
            learning_rate=learning_rate,
            max_iter=1,
        )

    def tree_traversal(self, node: AbstractGameNode):

        path = []

        while True:
            path.append(node)

            # whether the node is a leaf node in the current tree
            is_leaf = not self.tree[node]

            # whether the node has been explored
            is_explored = self.visit_counts[node] > 0

            # whether all of its children has been visited at least once
            is_fully_explored = len(self.tree[node]) == len(
                node.find_children(self.agent.meta_data)
            )

            if is_leaf:

                if node.is_terminal(self.agent.meta_data):
                    # nothing more to do if we are already at the terminal node
                    return path

                if not is_explored:
                    # if the node is not at all explored, we stop tree traversal and initiate the rollout from here
                    return path

                # now the node is a leaf and has been explored for at least once, now enter the expansion phase and add its children to the tree

                random_child = node.find_random_child(self.agent.meta_data)
                path.append(random_child)
                self.tree[node].add(random_child)

                # added a random child to path. This node should now be a leaf node and completely unexplored, we start rollout phase from here
                return path

            # the node is not a leaf

            if not is_fully_explored:
                # not all of its children has been explored. Add a random unexplored child to the tree, and start rollout from there
                unexplored_children = (
                    node.find_children(self.agent.meta_data) - self.tree[node]
                )

                child = random.choice(list(unexplored_children))
                path.append(child)
                self.tree[node].add(child)

                return path

            # The node is not a leaf, and all of its children has been explored at least once. Now we implement the progressive widening technique.

            widening = (
                math.floor(scaling(self.visit_counts[node]))
                >= self.agent.meta_data.branching_ratio[len(node.abstract_state)]
            )

            if widening:
                # add a new child to the tree
                # should also update the abstract game state and the action spaces of the game realization.

                new_child = self.node.add_child(
                    self.agent.meta_data, level=len(node.abstract_state) + 1
                )
                self.agent.update_metadata(len(node.abstract_state))

                path.append(new_child)
                self.tree[node].add(new_child)

                return path

            # the node is not a leaf, and all of its children has been explored at least once. Now we implement the selection phase

            node = self.ucb1_select(node)

    def ucb1_select(self, node: AbstractGameNode):
        children = copy.deepcopy(self.tree[node])

        def ucb1(child_node: AbstractGameNode):
            if self.visit_counts[child_node] == 0:
                return float(
                    "inf"
                )  # this should not happen though, as we have explicitly guaranteed that the node is not a leaf

            return self.values[
                child_node
            ] * node.player + self.exploration_weight * math.sqrt(
                math.log(self.visit_counts[node]) / self.visit_counts[child_node]
            )

        best_score = -float("inf")
        best_children = []

        for child in children:
            score = ucb1(child)
            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)

        return random.choice(best_children)

    def rollout(self, node: AbstractGameNode):
        node_ = copy.deepcopy(node)

        while True:
            if node_.is_terminal(self.agent.meta_data):
                return self.agent.reward(node_.abstract_state), node_

            node_ = node_.find_random_child(self.agent.meta_data)

    def backpropagate(self, path: list[AbstractGameNode], value: float):
        for node in path:
            self.visit_counts[node] += 1
            self.values[node] += value

    def gradient_update(
        self,
        end_node: AbstractGameNode,
        value: float,
        eps: float = 1e-7,
    ) -> Union[list[np.ndarray], list[np.ndarray]]:

        realized_action_history = []
        for i in range(len(end_node.abstract_state)):
            realized_action = self.agent.action_space.actions[i][
                end_node.abstract_state[i]
            ]
            realized_action_history.append(realized_action)

        gradients = []
        for i in range(len(realized_action_history)):
            gradient = np.zeros_like(realized_action_history[i])

            for j in range(len(realized_action_history[i])):
                # evaluate the one-sided gradient
                realized_action_history[i][j] += eps
                forward_value = self.agent.reward_with_action(realized_action_history)
                realized_action_history[i][j] -= eps

                gradient[j] = (forward_value - value) / eps

            gradients.append(gradient)

        updated_realized_action_history = [None] * len(realized_action_history)
        # update the realized action history

        for i in range(len(realized_action_history)):
            updated_realized_action_history[i] = (
                realized_action_history[i]
                + gradients[i] * (-1) ** i * self.learning_rate
            )
            updated_realized_action_history[i] /= np.linalg.norm(
                updated_realized_action_history[i]
            )

            self.agent.action_space.actions[i][end_node.abstract_state[i]] = (
                updated_realized_action_history[i]
            )

        updated_value = self.agent.reward_with_action(updated_realized_action_history)

        return (gradients, updated_value)

    def grad_update_dbi(
        self,
        end_node: AbstractGameNode,
    ):

        realized_action_history = []
        for i in range(len(end_node.abstract_state)):
            realized_action = self.agent.action_space.actions[i][
                end_node.abstract_state[i]
            ]
            realized_action_history.append(realized_action)

        actions = T.vstack(realized_action_history)
        new_action, _ = self.grad_agent.train(
            actions,
            minibatch_size=1,
            print_every=float("inf"),
        )
        new_action = new_action / new_action.norm(dim=1, keepdim=True)

        # need to update the actions of the AbstractGameNode

        for i in range(len(realized_action_history)):
            self.agent.action_space.actions[i][end_node.abstract_state[i]] = new_action[
                i
            ].detach()

        value = self.agent.reward_with_action(new_action)

        return new_action, value

    def step(self, node: AbstractGameNode):
        # tree traversal and expansion
        path = self.tree_traversal(node)
        leaf = path[-1]

        if self.gradient_update_frequency == 0:
            # no gradient updates
            # rollout with uniform random prior policy
            value, _ = self.rollout(leaf)
            self.backpropagate(path, value)

            return self.tree

        else:
            value, end_node = self.rollout(leaf)
            if self.gradient_update_frequency >= 1:
                self.gradient_update_frequency = math.ceil(
                    self.gradient_update_frequency
                )
                for _ in range(self.gradient_update_frequency):
                    # gradients, value = self.gradient_update(end_node, value)
                    # using the new dbi solver

                    if self.gradient_mode == "dbi":
                        new_action, value = self.grad_update_dbi(end_node)
                    else:
                        gradients, value = self.gradient_update(end_node, value)

                self.backpropagate(path, value)

            else:  # frequency < 1
                if random.random() < self.gradient_update_frequency:
                    # gradients, value = self.gradient_update(end_node, value)
                    # using the new dbi solver

                    if self.gradient_mode == "dbi":
                        new_action, value = self.grad_update_dbi(end_node)
                    else:
                        gradients, value = self.gradient_update(end_node, value)

                    self.backpropagate(path, value)
                else:
                    # for rejected gradient updates, we still need to backpropagate the value
                    self.backpropagate(path, value)

            return self.tree

    def choose(
        self,
        node: AbstractGameNode,
        num_simulations: int = 100,
    ) -> AbstractGameNode:

        for _ in range(num_simulations):
            self.step(node)

        children = self.tree[node]

        if not children:
            return None

        best_child = max(
            children,
            key=lambda child: (
                -float("inf")
                if not self.visit_counts[child]
                else self.visit_counts[child]
            ),
            # the minus 1 here is because there's a change of player going from the parent to the child node
        )

        return best_child


if __name__ == "__main__":
    # initialize test data
    random.seed(42)
    np.random.seed(42)
    T.manual_seed(42)

    meta_data = GameMetaData([4] * 3, 3)
    action_space = SphereActionSpace(dim=5, meta_data=meta_data)
    game = GameRealization(meta_data, action_space, StateSpace)
    node = AbstractGameNode.reset()
    gradient_update_frequency = 0.4
    agent = cMCTS(node, game)

    agents = []
    for stage in range(meta_data.depth):
        print(f"Stage {stage+1}")
        agent = cMCTS(
            node, game, gradient_mode="vanilla", gradient_update_frequency=0.4
        )
        agents.append(agent)
        node = agent.choose(node, num_simulations=100)
        print(node)

    # print the realized actions
    realized_actions = agent.agent.get_realized_actions(node.abstract_state)
    value = game.reward_with_action(realized_actions)
    print(realized_actions, value)
