import numpy as np
import torch as T
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
import math
import random
import copy
import numpy as np


####################################################################################################

# The abstract sequential game model

####################################################################################################


class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True


class GameMetaData:
    """Define the game abstractly:
    - branching_ratio: the number of possible moves at each node
    - depth: the number of moves in the game

    The game is assumed to be a two-player zero-sum game, assuming the first player to be the maximizer.
    The branching ratio is a depth-dimensional list of integers
    """

    def __init__(self, branching_ratio: list[int], depth: int):
        self.branching_ratio = branching_ratio
        self.depth = depth

    def set_branching_ratio(self, level, value):
        if level < 0 or level >= self.depth:
            raise ValueError("Invalid level")

        self.branching_ratio[level] = value

    def add_child(self, level):
        if level < 0 or level >= self.depth:
            raise ValueError("Invalid level")

        self.branching_ratio[level] += 1


####################################################################################################

# Simple sequential games

####################################################################################################


class ExtensiveFormGame(ABC):
    """A sequential game with abstract methods"""

    @abstractmethod
    def is_legal_action(self):
        return True

    @abstractmethod
    def payoff(self):
        """Assign payoff given"""
        return 0


class RandGame(ExtensiveFormGame):
    def __init__(self):
        """A three-stage, three player game as defined in the demonstration"""
        self.players = list(range(3))
        self.stages = list(range(3))

    def is_legal_action(self, actions: T.Tensor):
        assert actions.shape == (
            3,
            1,
        ), f"Invalid action shape, expected (3, 1), got {actions.shape}"
        assert isinstance(
            actions, T.Tensor
        ), f"Invalid action type, expected torch.Tensor, got {type(actions)}"
        return True

    def payoff(self, actions: T.Tensor):
        # self.is_legal_action(actions)
        x = actions
        u1 = -7 * x[0, 0] ** 2 + 9 * x[0, 0] * x[2, 0] + x[0, 0] - x[2, 0]
        u2 = (
            -2 * x[1, 0] ** 2
            - 4 * x[1, 0] * x[2, 0]
            - 10 * x[0, 0] ** 2
            + 2 * x[0, 0] * x[2, 0]
            - 3 * x[2, 0] ** 2
            + 4 * x[1, 0]
            + 7 * x[0, 0]
            - 8 * x[2, 0]
            - 8 * x[0, 0] * x[1, 0] * x[2, 0]
        )
        u3 = (
            -10 * x[2, 0] ** 2
            - 9 * x[1, 0] * x[2, 0]
            + 9 * x[1, 0] ** 2
            - 5 * x[2, 0]
            - 2 * x[1, 0]
        )

        return T.stack([u1, u2, u3])


class Rand3DGame(ExtensiveFormGame):
    def __init__(self):
        self.players = list(range(3))
        self.stages = list(range(3))
        self.dim = 3  # Dimension of the action space

    def is_legal_action(self, actions):
        assert actions.shape == (
            len(self.stages),
            self.dim,
        ), f"Invalid action shape, expected (3, 3), got {actions.shape}"
        # The first index labels the stage, the second index is the dimension of action
        return True

    def payoff(self, actions: T.Tensor):
        x = actions[0]
        y = actions[1]
        z = actions[2]

        u1 = -7 * T.sum(x**2) + 9 * T.sum(x) * T.sum(z) + T.sum(x) - T.sum(z)

        u2 = (
            -2 * T.sum(y**2)
            - 4 * T.sum(y) * T.sum(z)
            - 10 * T.sum(x) * T.sum(z)
            + 2 * T.sum(x**2)
            - 3 * T.sum(z**2)
            + 4 * T.sum(x) * T.sum(y) * T.sum(z)
            + 7 * T.sum(x)
            - 8 * T.sum(y)
            - 8 * T.sum(z)
        )

        u3 = (
            -10 * T.sum(z**2)
            - 9 * T.sum(y) * T.sum(z)
            + 9 * T.sum(y**2)
            - 5 * T.sum(y)
            - 2 * T.sum(z)
        )

        return T.stack([u1, u2, u3])


class Rand3DGamev1(ExtensiveFormGame):
    def __init__(self):
        self.players = list(range(3))
        self.stages = list(range(3))
        self.dim = 3  # Dimension of the action space

    def is_legal_action(self, actions):
        assert actions.shape == (
            len(self.stages),
            self.dim,
        ), f"Invalid action shape, expected (3, 3), got {actions.shape}"
        # The first index labels the stage, the second index is the dimension of action
        return True

    def payoff(self, actions: T.Tensor):
        x = actions[0]
        y = actions[1]
        z = actions[2]

        u1 = -7 * T.sum(x) ** 2 + 9 * T.sum(x) * T.sum(z) + T.sum(x) - T.sum(z)

        u2 = (
            -2 * T.sum(y) ** 2
            - 4 * T.sum(y) * T.sum(z)
            - 10 * T.sum(x) ** 2
            + 2 * T.sum(x) * T.sum(z)
            - 3 * T.sum(z) ** 2
            + 4 * T.sum(y)
            + 7 * T.sum(x)
            - 8 * T.sum(z)
            - 8 * T.sum(x) * T.sum(y) * T.sum(z)
        )

        u3 = (
            -10 * T.sum(z) ** 2
            - 9 * T.sum(y) * T.sum(z)
            + 9 * T.sum(y) ** 2
            - 5 * T.sum(z)
            - 2 * T.sum(y)
        )

        return T.stack([u1, u2, u3])


class Stackelberg(ExtensiveFormGame):
    def __init__(self, n):
        self.players = list(range(n))
        self.stages = list(range(n))

    def is_legal_action(self, actions: T.Tensor):
        assert (
            actions.dtype == T.float
        ), f"Invalid action dtype, expected torch.float32, got {actions.dtype}"

        assert actions.shape == (
            len(self.players),
            1,
        ), f"Invalid action shape, expected ({len(self.players)}, 1), got {actions.shape}"
        return True

    def payoff(self, actions: T.Tensor):
        price = 1 - T.sum(actions)
        revenue = price * actions
        return revenue


####################################################################################################

# more complex sequential games

####################################################################################################

BoardConfig = namedtuple("BoardConfig", ["config", "player", "terminal", "value"])


class SimpleBoard(BoardConfig, Node):
    def __new__(cls, *args, **kwargs):
        if not args:
            return cls.reset(cls)

        return super(SimpleBoard, cls).__new__(cls, *args, **kwargs)

    def reset(self):
        config = (0,) * 5
        player = 1
        terminal = False
        payoff = None
        return SimpleBoard(config, player, terminal, payoff)

    def move(self, site: int):
        if self.config[site] != 0:
            raise ValueError("Double Occupancy")

        if self.terminal:
            raise ValueError("Game Over")

        # the markers of both players are indistinguishable
        config = self.config[:site] + (1,) + self.config[site + 1 :]
        player = -self.player

        payoff = self.get_payoff(config)
        terminal = True if payoff is not None else False

        return SimpleBoard(config, player, terminal, payoff)

    @staticmethod
    def winning_combos():
        """winning combinations for the simpler (1D) version of the game"""
        yield (0, 1, 2)
        yield (1, 2, 3)
        yield (2, 3, 4)

    def get_payoff(self, config):
        for combo in self.winning_combos():
            if all(config[i] == 1 for i in combo):
                return 1

        if sum(config[i] for i in range(len(config))) == 3:
            return -1

        return None

    ################################################################

    # Implementing the node class

    ################################################################

    def find_children(self):
        if self.terminal:
            return set()

        return {self.move(i) for i, value in enumerate(self.config) if value == 0}

    def find_random_child(self):
        if self.terminal:
            return None

        return random.choice(list(self.find_children()))

    # for completeness.. the following functions can also be accessed as properties of the Board class from the named tuple
    def is_terminal(self):
        return self.terminal

    def reward(self):
        return self.value


####################################################################################################

# Sampling action spaces for real numbers as well

####################################################################################################


class DiscretizedActionSpace(ABC):
    def __init__(self, dim: int, meta_data: GameMetaData):
        self.dim = dim
        self.meta_data = meta_data
        self.actions = self.sample()

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def add_action(self, level, action=None):
        pass


class CubeGrid(DiscretizedActionSpace):
    def __init__(self, dim: int, meta_data: GameMetaData, lows=[-1], highs=[1]):
        self.lows = lows
        self.highs = highs
        super().__init__(dim, meta_data)
        # note that the number of samples has been encoded in the meta_data.branching_ratio

    def sample(self):
        actions = []
        # edge_len = math.ceil(self.meta_data.branching_ratio[0] ** (1 / self.dim))

        for i in range(self.meta_data.depth):
            edge_len = math.ceil(self.meta_data.branching_ratio[i] ** (1 / self.dim))
            grid = [
                T.linspace(self.lows[i], self.highs[i], edge_len)
                for _ in range(self.dim)
            ]
            mesh = T.meshgrid(*grid)
            action = T.stack(mesh, dim=-1).reshape(-1, self.dim)
            actions.append(action)

        return actions

    def add_action(self, level, action=None):
        if action is not None:
            new_action = action
        else:
            new_action = T.random.uniform(
                self.low, self.high, self.meta_data.branching_ratio[level]
            )

        self.actions[level] = T.vstack([self.actions[level], new_action])
        pass


class BoxActionSpace(DiscretizedActionSpace):
    def __init__(self, dim: int, meta_data: GameMetaData, low=-1, high=1):
        self.low = low
        self.high = high
        super().__init__(dim, meta_data)

    def sample(self):
        actions = []
        for i in range(self.meta_data.depth):
            X = T.linspace(self.low, self.high, self.meta_data.branching_ratio[i])

            actions.append(X)

        return actions

    def add_action(self, level, action=None):
        if action is not None:
            new_action = action
        else:
            new_action = T.random.uniform(
                self.low, self.high, self.meta_data.branching_ratio[level]
            )

        self.actions[level] = T.vstack([self.actions[level], new_action])
        pass


class SphereActionSpace(DiscretizedActionSpace):

    def sample(self):
        actions = []
        for i in range(self.meta_data.depth):
            X = self.sampling_ndsphere_gaussian(self.meta_data.branching_ratio[i])
            actions.append(X)

        return actions

    def sampling_ndsphere_gaussian(self, n_samples):
        # Generate samples from a multivariate normal distribution
        samples = T.randn(n_samples, self.dim, dtype=T.float64)
        # Normalize the samples to lie on the surface of the sphere
        samples = samples / samples.norm(dim=1, keepdim=True)

        return samples

    def add_action(self, level, action=None):
        """This is to be used together with the add_action in the metadata class"""
        if action is not None:
            new_action = action
        else:
            new_action = self.sampling_ndsphere_gaussian(1)

        self.actions[level] = T.vstack([self.actions[level], new_action])
        pass


MixedState = namedtuple("MixedState", ["state", "player", "terminal", "value"])


class LossGameStateSpace(MixedState, Node):

    def __new__(cls, *args, **kwargs):
        if not args:
            return cls.reset(cls)

        return super(LossGameStateSpace, cls).__new__(cls, *args, **kwargs)

    def reset(self):
        start_state = ()
        start_player = 1
        start_terminal = False
        start_value = None

        return LossGameStateSpace(
            start_state, start_player, start_terminal, start_value
        )

    def move(self, realized_action: T.Tensor):
        if self.terminal:
            raise ValueError("Game Over")

        new_state = self.state + (realized_action,)
        new_player = self.player + 1  # Generalize from two-player games
        new_terminal = len(new_state) == 3
        new_value = self.reward(new_state)

        return LossGameStateSpace(new_state, new_player, new_terminal, new_value)

    def payoff(self, actions: T.Tensor):
        # self.is_legal_action(actions)
        x = actions
        u1 = -7 * x[0, 0] ** 2 + 9 * x[0, 0] * x[2, 0] + x[0, 0] - x[2, 0]
        u2 = (
            -2 * x[1, 0] ** 2
            - 4 * x[1, 0] * x[2, 0]
            - 10 * x[0, 0] ** 2
            + 2 * x[0, 0] * x[2, 0]
            - 3 * x[2, 0] ** 2
            + 4 * x[1, 0]
            + 7 * x[0, 0]
            - 8 * x[2, 0]
            - 8 * x[0, 0] * x[1, 0] * x[2, 0]
        )
        u3 = (
            -10 * x[2, 0] ** 2
            - 9 * x[1, 0] * x[2, 0]
            + 9 * x[1, 0] ** 2
            - 5 * x[2, 0]
            - 2 * x[1, 0]
        )

        return T.stack([u1, u2, u3])

    def reward(self, state=None):
        # In this game state space the value is a tensor of shape (3, 1)
        if state is not None:
            actions = T.vstack(state)
            if len(state) != 3:
                return None

            return self.payoff(actions)

        if len(self.state) != 3:
            return None

        actions = T.vstack(self.state)
        return self.payoff(actions)


class StateSpace(MixedState, Node):
    """Should assign a probability measure to each of the abstract state when queried"""

    def __new__(cls, *args, **kwargs):
        if not args:
            return cls.reset(cls)

        return super(StateSpace, cls).__new__(cls, *args, **kwargs)

    def reset(self):
        start_config = (0,) * 5
        start_player = 1
        start_terminal = False
        start_value = None
        start_board = SimpleBoard(
            start_config, start_player, start_terminal, start_value
        )

        start_amp = 1.0
        state = defaultdict(float)

        state[start_board] = start_amp

        player = 1
        terminal = False
        payoff = None
        return StateSpace(state, player, terminal, payoff)

    def move(self, action: np.ndarray):
        if self.terminal:
            raise ValueError("Game Over")

        new_state = defaultdict(float)

        for board, amp in self.state.items():
            if board.terminal:
                new_state[board] = amp
            else:
                for i in range(len(action)):
                    try:
                        board_cache = copy.deepcopy(board)
                        new_board = SimpleBoard.move(board_cache, i)
                        new_state[new_board] += amp * action[i]
                    except ValueError:  # Double Occupancy
                        # print(f'Double Occupancy found at site {i}')
                        pass

        if all([abs(amp) <= 1e-7 for amp in new_state.values()]):
            raise ValueError("Zero Amplitude, state completely annihilated")

        player = -self.player
        terminal = all([new_board.terminal for new_board in new_state.keys()])
        value = self.reward(new_state)

        return StateSpace(new_state, player, terminal, value)

    def reward(self, state=None):

        if state is not None:
            # requested a value for some separate state
            # the type should be dict[SimpleBoard, float]
            if not all(board.terminal for board in state.keys()):
                return None

            value = 0
            for board, amp in state.items():
                value += board.reward() * abs(amp) ** 2

            value /= sum([abs(amp) ** 2 for amp in state.values()])

            return value

        # requesting a value for an instance of the class
        if not self.terminal:
            return None

        value = 0
        for board, amp in self.state.items():
            if not board.terminal:
                raise ValueError("requesting value on a non-terminal state")

            value += board.reward() * abs(amp) ** 2

        value /= sum([abs(amp) ** 2 for amp in self.state.values()])

        return value


class clStateSpace(MixedState, Node):
    """
    Classical Simple Board Game
    -------
    __new__(cls, *args, **kwargs)
        Creates a new instance of the class. If no arguments are provided, it resets the state space.
    reset(self)
        Resets the state space to the initial configuration.
    move(self, action: np.ndarray)
        Applies a move to the current state space based on the given action. Raises a ValueError if the game is over or if the state is completely annihilated.
    reward(self, state=None)
        Calculates the reward for a given state or the current instance of the class. Returns None if the state is not terminal.
    """

    def __new__(cls, *args, **kwargs):
        if not args:
            return cls.reset(cls)

        return super(clStateSpace, cls).__new__(cls, *args, **kwargs)

    def reset(self):
        start_config = (0,) * 5
        start_player = 1
        start_terminal = False
        start_value = None
        start_board = SimpleBoard(
            start_config, start_player, start_terminal, start_value
        )

        start_amp = 1.0
        state = defaultdict(float)

        state[start_board] = start_amp

        player = 1
        terminal = False
        payoff = None
        return clStateSpace(state, player, terminal, payoff)

    def move(self, action: np.ndarray):
        if self.terminal:
            raise ValueError("Game Over")

        new_state = defaultdict(float)

        for board, amp in self.state.items():
            if board.terminal:
                new_state[board] = amp
            else:
                for i in range(len(action)):
                    try:
                        board_cache = copy.deepcopy(board)
                        new_board = SimpleBoard.move(board_cache, i)
                        new_state[new_board] += amp * action[i]
                    except ValueError:  # Double Occupancy
                        # print(f'Double Occupancy found at site {i}')
                        pass

        if all([abs(amp) <= 1e-7 for amp in new_state.values()]):
            raise ValueError("Zero Amplitude, state completely annihilated")

        player = -self.player
        terminal = all([new_board.terminal for new_board in new_state.keys()])
        value = self.reward(new_state)

        return clStateSpace(new_state, player, terminal, value)

    def reward(self, state=None):
        """
        This reward is for classical stochastic version of the boards.
        """
        if state is not None:
            # requested a value for some separate state
            # the type should be dict[SimpleBoard, float]
            if not all(board.terminal for board in state.keys()):
                return None

            value = 0
            for board, amp in state.items():
                value += board.reward() * amp

            value /= sum([amp for amp in state.values()])

            return value

        # requesting a value for an instance of the class
        if not self.terminal:
            return None

        value = 0
        for board, amp in self.state.items():
            if not board.terminal:
                raise ValueError("requesting value on a non-terminal state")

            value += board.reward() * amp

        value /= sum([amp for amp in self.state.values()])

        return value


class TicTacToeBoard(BoardConfig, Node):

    def __new__(cls, *args, **kwargs):
        if not args:
            return cls.reset(cls)

        return super(TicTacToeBoard, cls).__new__(cls, *args, **kwargs)

    def reset(self):
        config = (0,) * 9
        player = 1
        terminal = False
        payoff = None
        return TicTacToeBoard(config, player, terminal, payoff)

    def move(self, site: int):
        if self.config[site] != 0:
            raise ValueError("Double Occupancy")

        if self.terminal:
            raise ValueError("Game Over")

        # the markers of both players are now distinguishable
        config = self.config[:site] + (self.player,) + self.config[site + 1 :]
        player = -self.player

        payoff = self.get_payoff(config)
        terminal = True if payoff is not None else False

        return TicTacToeBoard(config, player, terminal, payoff)

    @staticmethod
    def winning_combos():
        """winning combinations for the simpler (1D) version of the game"""
        yield (0, 1, 2)
        yield (3, 4, 5)
        yield (6, 7, 8)
        yield (0, 3, 6)
        yield (1, 4, 7)
        yield (2, 5, 8)
        yield (0, 4, 8)
        yield (2, 4, 6)

    def get_payoff(self, config):
        for combo in self.winning_combos():
            if all(config[i] == 1 for i in combo):
                return 1

            if all(config[i] == -1 for i in combo):
                return -1

        if all(value != 0 for value in config):
            return 0

        return None

    ################################################################

    # Implementing the node class

    ################################################################

    def find_children(self):
        if self.terminal:
            return set()

        return {self.move(i) for i, value in enumerate(self.config) if value == 0}

    def find_random_child(self):
        if self.terminal:
            return None

        return random.choice(list(self.find_children()))

    # for completeness.. the following functions can also be accessed as properties of the Board class from the named tuple
    def is_terminal(self):
        return self.terminal

    def reward(self):
        return self.value


class QTTTStateSpace(MixedState, Node):
    def __new__(cls, *args, **kwargs):
        if not args:
            return cls.reset(cls)

        return super(QTTTStateSpace, cls).__new__(cls, *args, **kwargs)

    def reset(self):
        start_config = (0,) * 9
        start_player = 1
        start_terminal = False
        start_value = None
        empty_board = TicTacToeBoard(
            start_config, start_player, start_terminal, start_value
        )

        start_state = {empty_board: 1.0 + 0.0j}
        player = 1
        terminal = False
        payoff = None

        return QTTTStateSpace(start_state, player, terminal, payoff)

    def move(self, action: np.ndarray):
        # for this version the internal game logic is done using numpy instead of torch

        if self.terminal:
            raise ValueError("Game Over")

        new_state = defaultdict(complex)

        for board, amp in self.state.items():
            if board.terminal:
                new_state[board] = amp
            else:
                for i in range(len(action)):
                    try:
                        board_cache = copy.deepcopy(board)
                        new_board = TicTacToeBoard.move(board_cache, i)
                        new_state[new_board] += amp * action[i]
                    except ValueError:
                        pass

        if all([abs(amp) <= 1e-7 for amp in new_state.values()]):
            raise ValueError("Zero Amplitude, state completely annihilated")

        player = -self.player
        terminal = all([new_board.terminal for new_board in new_state.keys()])
        value = self.reward(new_state)

        return QTTTStateSpace(new_state, player, terminal, value)

    def reward(self, state=None):

        if state is not None:
            if not all(board.terminal for board in state.keys()):
                return None

            value = 0
            for board, amp in state.items():
                value += board.reward() * abs(amp) ** 2

            value /= sum([abs(amp) ** 2 for amp in state.values()])

            return value

        if not self.terminal:
            return None

        value = 0
        for board, amp in self.state.items():
            if not board.terminal:
                return None

            value += board.reward() * abs(amp) ** 2

        value /= sum([abs(amp) ** 2 for amp in self.state.values()])

        return value


class GameRealization:
    """
    A class to represent the realization of a game
    --------------------------------

    Given the state space and action space that constitute the rules of the game, i.e. the translation from abstract state labels to actural representation of states and actions.

    Attributes
    ----------
    meta_data : GameMetaData
        Metadata associated with the game.
    action_space : DiscretizedActionSpace
        The action space of the game.
    game_state : MixedState
        The initial state of the game.


    Methods
    -------
    reward(abstract_state: tuple[int]) -> float:
        Calculates the reward for a given abstract state.
    update_metadata(level: int, action=None) -> None:
        Updates the metadata and action space with a new level and optional action. This is designed to implement progressive widening
    reward_with_action(actions: list[T.Tensor]) -> T.Tensor:
        Calculates the reward for a given list of realized actions, useful for gradient calculation.
    get_realized_actions(abstract_state: tuple[int]) -> list[np.ndarray]:
        Returns the realized action history corresponding to a given abstract state.
    """

    def __init__(
        self,
        meta_data: GameMetaData,
        action_space: DiscretizedActionSpace,
        game_state: MixedState,
    ):
        # NOTE: the inputs should be rules instead of states. They should represent the high level rules for moving pieces and calculating payoffs, so the inputs should be passed as classes themselves instead of instances of the classes
        self.meta_data = meta_data
        self.action_space = action_space  # initialize the action space
        self.game_state = game_state

    def reward(self, abstract_state: tuple[int]):
        # double-check whether the state has been terminated

        # initialize state
        state = self.game_state()

        for level, action_index in enumerate(abstract_state):
            # print(level, action_index)
            state = state.move(self.action_space.actions[level][action_index])

        if not state.terminal:
            raise ValueError("The state has not been terminated")

        rwd = state.reward()
        return rwd

    def update_metadata(self, level, action=None):
        self.meta_data.add_child(level)
        self.action_space.add_action(level, action=action)

    def reward_with_action(self, actions: list[T.Tensor]) -> T.Tensor:
        """For gradient calculation"""
        state = self.game_state()

        # t0 = time.time()
        for level, realized_action in enumerate(actions):
            state = state.move(realized_action)
        # t1 = time.time()
        # print(t1 - t0)
        return state.reward()

    def get_realized_actions(self, abstract_state: tuple[int]) -> list[np.ndarray]:
        realized_actions = []

        for level, action_index in enumerate(abstract_state):
            realized_actions.append(self.action_space.actions[level][action_index])

        return realized_actions


AbstractGameState = namedtuple("AbstractGameState", ["abstract_state", "player"])


class AbstractGameNode(AbstractGameState, Node):
    def __new__(cls, *args, **kwargs):
        if not args:
            return cls.reset()

        return super(AbstractGameNode, cls).__new__(cls, *args, **kwargs)

    def reset():
        abstract_state = ()  # The empty tuple denote the root state
        player = 1

        return AbstractGameNode(abstract_state, player)

    def move(self, action: int, meta_data: GameMetaData):

        if self.is_terminal(meta_data):
            raise ValueError("Can't move from a terminal state")

        if action < 0 or action >= meta_data.branching_ratio[len(self.abstract_state)]:
            raise ValueError("Invalid action")

        new_abstract_state = self.abstract_state + (action,)
        new_player = -self.player

        return AbstractGameNode(new_abstract_state, new_player)

    def find_children(self, meta_data: GameMetaData):
        if len(self.abstract_state) == meta_data.depth:
            return set()

        return {
            self.move(action, meta_data)
            for action in range(meta_data.branching_ratio[len(self.abstract_state)])
        }

    def find_random_child(self, meta_data: GameMetaData):

        children = list(self.find_children(meta_data))
        if not children:
            raise ValueError("Can't choose a child from a terminal state")

        return random.choice(children)

    def is_terminal(self, meta_data: GameMetaData):
        return len(self.abstract_state) == meta_data.depth

    def is_legal(self, meta_data: GameMetaData):
        return all(
            [
                action_index < meta_data.branching_ratio[i]
                for i, action_index in enumerate(self.abstract_state)
            ]
        )

    # this method should be depracated as it requires a predefined GamerRealization instance
    def reward(self, meta_data: GameMetaData):
        if not self.is_terminal(meta_data):
            raise ValueError("Can't get value from a non-terminal state")

        return GameRealization.reward(self.abstract_state)

    def add_child(self, meta_data, level):
        if level <= 0 or level > meta_data.depth:
            print(level)
            raise ValueError("level out of range")

        abstract_state = self.abstract_state[: level - 1] + (
            meta_data.branching_ratio[level - 1],
        )
        # a workaround for player. The add_child function is not supposed to be used in the middle of the game, but nevertheless it is implemented as a intermediate-step workaround where we insert a child node in the middle of the game
        player = 1 if level % 2 == 0 else -1
        return AbstractGameNode(abstract_state, player)


# write a class that fit the Simple Board (1D Tic-Tac-Toe game) to the ExtensiveFormGame class so that we can use Differential Backward Induction on it.


class ClassicalSimpleBoard(ExtensiveFormGame):
    def __init__(self):
        self.players = list(range(2))
        self.stages = list(range(3))
        self.dim = 5
        self.runner = GameRealization(
            GameMetaData([1, 1, 1], 3), BoxActionSpace, clStateSpace
        )

    def is_legal_action(self, action: T.Tensor):
        assert action.shape == (
            len(self.stages),
            self.dim,
        ), f"Invalid action shape, expected (3, 1), got {action.shape}"

        assert T.all(action >= 0), "All elements of the action must be non-negative"
        return True

    def payoff(self, actions: T.Tensor):
        actions = actions / actions.sum(dim=1, keepdim=True)
        first_player_payoff = self.runner.reward_with_action(actions)

        return T.stack(
            [(-1) ** i * first_player_payoff for i in range(len(self.stages))]
        )


class QuantumSimpleBoard(ExtensiveFormGame):
    def __init__(self):
        self.players = list(range(2))
        self.stages = list(range(3))
        self.dim = 5
        self.runner = GameRealization(
            GameMetaData([1, 1, 1], 3), SphereActionSpace, StateSpace
        )  # this necessity of specifying the metadata is a bit of a hack, to be optimized later

    def is_legal_action(self, actions: T.Tensor):
        assert actions.shape == (
            len(self.stages),
            self.dim,
        ), f"Invalid action shape, expected (3, 5), got {actions.shape}"
        return True

    def payoff(self, actions: T.Tensor):

        actions = actions / actions.norm(dim=1, keepdim=True)
        first_player_payoff = self.runner.reward_with_action(actions)

        return T.stack(
            [(-1) ** i * first_player_payoff for i in range(len(self.stages))]
        )


class QuantumTicTacToe(ExtensiveFormGame):
    def __init__(self):
        self.players = list(range(2))
        self.stages = list(range(9))
        self.dim = 9
        self.runner = GameRealization(
            GameMetaData([3] * 9, 9), SphereActionSpace, QTTTStateSpace
        )

    def is_legal_action(self, actions: T.Tensor):
        assert actions.shape == (
            len(self.stages),
            self.dim,
        ), f"Invalid action shape, expected (9, 9), got {actions.shape}"
        return True

    def payoff(self, actions: T.Tensor):
        actions = actions / actions.norm(dim=1, keepdim=True)
        first_player_payoff = self.runner.reward_with_action(actions)

        return T.stack(
            [(-1) ** i * first_player_payoff for i in range(len(self.stages))]
        )


if __name__ == "__main__":
    # Test the game
    # game = RandGame()
    # actions = T.tensor([[0.0, 0.0, -1.09]]).transpose(0, 1).requires_grad_(True)
    # print(game.payoff(actions))
    # print(game.is_legal_action(actions))
    import time

    T.manual_seed(42)

    t0 = time.time()

    game = QuantumTicTacToe()
    t1 = time.time()
    actions = T.tensor(  # the optimal play where both get 0 payoff
        [
            [0, 1 / math.sqrt(2), 0, 1 / math.sqrt(2), 0],
            [1 / math.sqrt(2), 0, 0, 0, 1 / math.sqrt(2)],
            [0, 0, 1, 0, 0],
        ],
        dtype=T.float64,
        requires_grad=True,
    )
    actions = T.randn(9, 9, dtype=T.float64, requires_grad=True)
    actions = actions / actions.norm(dim=1, keepdim=True)
    # actions = actions + T.randn_like(actions) * 0.1
    # print(actions)
    M = game.payoff(actions)[0]
    print(M)

    from player import *

    agent = QAgent(cpol=actions)
    print(agent.exp_util)
    # print(agent.amps)
