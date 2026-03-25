"""
Contains environments of (extensive-form) games with continuous action spaces. Including
- Loss game for multiple players
- The simple board game (1D tic-tac-toe)
- The 2D tic-tac-toe

Each environment includes two core aspects
- the transition of environment's *state* upon certain actions by the environment
- the payoff assignment when a terminal state is reached
"""

from abc import ABC, abstractmethod
from sympy import Q
import torch as t
from torch import Tensor
from typing import Dict
from jaxtyping import Complex

class GameEnv(ABC):

    @abstractmethod
    def _is_legal_action(self, action):
        return True
    
    @abstractmethod
    def _calculate_payoff(self):
        return None
    
    @abstractmethod
    def move(self, action):
        return None

    @abstractmethod
    def _is_terminal(self):
        return False
    
    @abstractmethod
    def create_initial_state(self):
        return None


class LossGameEnv(GameEnv):
    def __init__(self, num_stages, num_players, d_action, state = None, player = None, preserve_grad = False):

        # Some metadata the class keeps track of
        self.num_stages = num_stages
        self.num_players = num_players
        self.d_action = d_action

        # The dynamic variables modified in a gameplay
        # this is to be treated as a Node object in the MCTS algorithm
        self.state = state if state is not None else [] # initialize if not given
        self.player = player if player is not None else self._get_player()
        self.terminal = self._is_terminal()
        self.payoff = self._calculate_payoff() if self.terminal else None

        self.preserve_grad = preserve_grad
    
    def _is_legal_action(self, action):
        # The action should be a tensor
        assert isinstance(action, t.Tensor), "Action should be a tensor"
        assert action.shape == (self.d_action,), f"Invalid action shape, get {action.shape}."
        assert action.dtype == t.float32, f"Expected dtype float32, get {action.dtype}."
        return True
    
    def _get_player(self):
        return len(self.state)
    
    def _is_terminal(self):
        return len(self.state) == self.num_stages

    def move(self, action):
        assert self._is_legal_action(action), "Illegal action"
        new_state = self.state + [action]
        new_player = self.player + 1
        return self.__class__(
            num_stages=self.num_stages,
            num_players=self.num_players,
            d_action=self.d_action,
            state=new_state,
            player=new_player,
            preserve_grad=self.preserve_grad
        )
        # self.state = self.state + [action]
        # self.player = self.player + 1
        # self.terminal = self._is_terminal()
        # self.payoff = self._calculate_payoff() if self.terminal else None
    
    def _calculate_payoff(self):
        pass

    def create_initial_state(self):
        return self.__class__(
            num_stages=self.num_stages,
            num_players=self.num_players,
            d_action=self.d_action,
            preserve_grad=self.preserve_grad,
        )

class LossGame1DEnv(LossGameEnv):
    def __init__(self, num_stages, num_players, d_action, state = None, player = None, preserve_grad = False):
        super().__init__(num_stages, num_players, d_action, state, player, preserve_grad)
    
    def _calculate_payoff(self):
        x = self.state
        u1 = -7 * x[0] ** 2 + 9 * x[0] * x[2] + x[0] - x[2]
        u2 = (
            -2 * x[1] ** 2
            - 4 * x[1] * x[2]
            - 10 * x[0] ** 2
            + 2 * x[0] * x[2]
            - 3 * x[2] ** 2
            + 4 * x[1]
            + 7 * x[0]
            - 8 * x[2]
            - 8 * x[0] * x[1] * x[2]
        )
        u3 = (
            -10 * x[2] ** 2
            - 9 * x[1] * x[2]
            + 9 * x[1] ** 2
            - 5 * x[2]
            - 2 * x[1]
        )

        return t.stack([u1, u2, u3]).reshape(self.num_players)

class LossGame3DEnv(LossGameEnv):
    def __init__(self, num_stages, num_players, d_action, state = None, player = None, preserve_grad = False):
        super().__init__(num_stages, num_players, d_action, state, player, preserve_grad)
    
    def _calculate_payoff(self):
        x = self.state[0]
        y = self.state[1]
        z = self.state[2]

        u1 = -7 * t.sum(x) ** 2 + 9 * t.sum(x) * t.sum(z) + t.sum(x) - t.sum(z)

        u2 = (
            -2 * t.sum(y) ** 2
            - 4 * t.sum(y) * t.sum(z)
            - 10 * t.sum(x) ** 2
            + 2 * t.sum(x) * t.sum(z)
            - 3 * t.sum(z) ** 2
            + 4 * t.sum(y)
            + 7 * t.sum(x)
            - 8 * t.sum(z)
            - 8 * t.sum(x) * t.sum(y) * t.sum(z)
        )

        u3 = (
            -10 * t.sum(z) ** 2
            - 9 * t.sum(y) * t.sum(z)
            + 9 * t.sum(y) ** 2
            - 5 * t.sum(z)
            - 2 * t.sum(y)
        )

        return t.stack([u1, u2, u3]).reshape(self.num_players)


# Board games

class Board(ABC):
    @property
    @abstractmethod
    def num_sites(self): ...

    # @property
    # @abstractmethod
    # def board(self): ...

    # @property
    # @abstractmethod
    # def player(self): ...

    # @property
    # @abstractmethod
    # def terminal(self): ...

    # @property
    # @abstractmethod
    # def payoff(self): ...

    @classmethod
    @abstractmethod
    def create_initial_state(cls): ...

    @abstractmethod
    def _is_legal_action(self, action): ...

    @abstractmethod
    def move(self, action): ...

    @abstractmethod
    def _is_terminal(self): ...

    @abstractmethod
    def _calculate_payoff(self): ...

    @staticmethod
    @abstractmethod
    def _winning_combos(): ...

    @abstractmethod
    def __hash__(self): ...

    @abstractmethod
    def __eq__(self, other): ...

class SimpleBoard(Board):
    """
    The 1D 5-site Tic-Tac-Toe board with the same marker for both players.
    """

    BoardType = tuple[int]
    num_sites = 5
    num_stages = 3

    def __init__(self, board = None, player = 1):
        self.board: SimpleBoard.BoardType = board if board is not None else (0,) * SimpleBoard.num_sites
        self.player = player
        self.terminal = self._is_terminal()
        self.payoff = self._calculate_payoff() if self.terminal else None
    
    def _is_legal_action(self, action):
        return self.board[action] == 0 and not self.terminal
    
    def _is_terminal(self):
        return self.board.count(1) == SimpleBoard.num_stages
    
    @classmethod
    def create_initial_state(cls):
        board = (0,) * SimpleBoard.num_sites
        player = 1
        return cls(board, player)
    
    @staticmethod
    def _winning_combos():
        """winning combinations for the simpler (1D) version of the game"""
        yield (0, 1, 2)
        yield (1, 2, 3)
        yield (2, 3, 4)
    
    def move(self, action):
        assert self._is_legal_action(action), "Illegal action or terminated board"
        new_board = self.board[:action] + (1,) + self.board[action + 1:]
        new_player = -self.player

        return self.__class__(
            board=new_board,
            player=new_player,
        )
    
    def _calculate_payoff(self):
        for combo in self._winning_combos():
            if all(self.board[i] == 1 for i in combo):
                return t.tensor(1.0)
        
        return t.tensor(-1.0)

    def __hash__(self):
        return hash(self.board)
    
    def __eq__(self, other):
        if not isinstance(other, SimpleBoard):
            return False
        return self.board == other.board


class TicTacToeBoard(Board):
    """Classic 3x3 Tic-Tac-Toe board with alternating players."""

    BoardType = t.Tensor
    num_sites = 9
    num_stages = 9

    def __init__(self, board: BoardType | None = None, player: int = 1):
        if board is None:
            self.board = t.zeros(TicTacToeBoard.num_sites, dtype=t.int8)
        elif isinstance(board, t.Tensor):
            self.board = board.to(dtype=t.int8).clone()
        else:
            self.board = t.as_tensor(board, dtype=t.int8).clone()

        self.player = player
        self.terminal = self._is_terminal()
        self.payoff = self._calculate_payoff() if self.terminal else None

    @classmethod
    def create_initial_state(cls):
        return cls()

    @staticmethod
    def _winning_combos():
        yield (0, 1, 2)
        yield (3, 4, 5)
        yield (6, 7, 8)
        yield (0, 3, 6)
        yield (1, 4, 7)
        yield (2, 5, 8)
        yield (0, 4, 8)
        yield (2, 4, 6)

    def _is_legal_action(self, action: int):
        if not (0 <= action < self.num_sites):
            return False
        if self.terminal:
            return False
        return int(self.board[action]) == 0

    def move(self, action: int):
        assert self._is_legal_action(action), "Illegal move or game already finished"
        new_board = self.board.clone()
        new_board[action] = int(self.player)
        return self.__class__(
            board=new_board,
            player=-self.player,
        )

    def _is_terminal(self):
        if self._calculate_winner() is not None:
            return True
        return not (self.board == 0).any()

    def _calculate_payoff(self):
        winner = self._calculate_winner()
        if winner == 1:
            return t.tensor(1.0, dtype=t.float32)
        if winner == -1:
            return t.tensor(-1.0, dtype=t.float32)
        if not (self.board == 0).any():
            return t.tensor(0.0, dtype=t.float32)
        return None

    def _calculate_winner(self):
        for combo in self._winning_combos():
            line = [int(self.board[i]) for i in combo]
            if all(val == 1 for val in line):
                return 1
            if all(val == -1 for val in line):
                return -1
        return None

    def __hash__(self):
        return hash((tuple(int(x) for x in self.board.tolist()), int(self.player)))

    def __eq__(self, other):
        if not isinstance(other, TicTacToeBoard):
            return False
        return t.equal(self.board, other.board) and self.player == other.player

class QuantumBoardGameEnv(GameEnv):

    # In this class, we use the chain of actions to represent the game state (since the game is perfect information this choice integrates better with the existing optimizer code)
    StateType = tuple[Tensor]
    StateDictType = Dict[Board, Tensor]

    def __init__(
        self,
        board_cls: type[Board],
        state: StateType | None = None,
        state_dict: StateDictType | None = None,
    ):
        self.board_cls = board_cls
        self.num_stages = self.board_cls.num_stages
        self.num_players = 2

        self.state: QuantumBoardGameEnv.StateType = () if state is None else state
        # Compute the state dictionary in addition to the chain of actions recorded.
        self._state_dict: QuantumBoardGameEnv.StateDictType = self._get_state_dict() if state_dict is None else state_dict
        self.player = self._get_player()
        self.terminal = self._is_terminal()
        self.payoff = self._calculate_payoff() if self.terminal else None
 
    def _get_player(self):
        return len(self.state) % 2

    def create_initial_state(self):
        return self.__class__(
            board_cls=self.board_cls,
            state=(),
            state_dict={self.board_cls.create_initial_state(): t.tensor(1.0 + 0.0j, dtype=t.complex64)},
        )
    
    def _is_terminal(self):
        return len(self.state) == self.board_cls.num_stages
    
    def _is_legal_action(self, action: Complex[Tensor, "site"]):
        if action.shape != (self.board_cls.num_sites,):
            return False
        if action.dtype != t.complex64:
            return False
        if not t.isclose(t.norm(action), t.tensor(1.0, dtype=t.float32), rtol=1e-6, atol=1e-6):
            return False

        return True
    
    def move(self, action):
        assert self._is_legal_action(action), "Illegal action"
        new_state = self.state + (action,)

        new_state_dict: QuantumBoardGameEnv.StateDictType = {}
        for board, amp in self._state_dict.items():
            if board.terminal:
                new_state_dict[board] = new_state_dict.get(board, t.tensor(0.0 + 0.0j, dtype=t.complex64)) + amp
                continue
            for i in range(len(action)):
                if board._is_legal_action(i):
                    child_board = board.move(i)
                    contribution = amp * action[i]
                    new_state_dict[child_board] = new_state_dict.get(child_board, t.tensor(0.0 + 0.0j, dtype=t.complex64)) + contribution

        if not new_state_dict:
            raise ValueError("Zero Amplitude, state completely annihilated")

        total_amplitude = t.sum(t.stack([amp for amp in new_state_dict.values()]).abs() ** 2).real
        if total_amplitude <= 1e-14:
            raise ValueError("Zero Amplitude, state completely annihilated")
        
        return self.__class__(
            board_cls=self.board_cls,
            state=new_state,
            state_dict=new_state_dict,
        )
    
    def _get_state_dict(self):
        # For some chain of actions, compute the dictionary of states
        state_dict: QuantumBoardGameEnv.StateDictType = {self.board_cls.create_initial_state(): t.tensor(1.0 + 0.0j, dtype=t.complex64)}

        if len(self.state) > self.board_cls.num_stages:
            raise ValueError("Chain of actions exceed the number of stages")


        for action in self.state:
            new_state_dict = {}
            assert self._is_legal_action(action), "Illegal action"
            for board, amp in state_dict.items():
                if board.terminal:
                    new_state_dict[board] = new_state_dict.get(board, t.tensor(0.0 + 0.0j, dtype=t.complex64)) + amp
                    continue
                for i in range(len(action)):
                    if board._is_legal_action(i):
                        child_board = board.move(i)
                        contribution = amp * action[i]
                        new_state_dict[child_board] = new_state_dict.get(child_board, t.tensor(0.0 + 0.0j, dtype=t.complex64)) + contribution
            state_dict = new_state_dict

        return state_dict

    def _calculate_payoff(self):
        payoffs = t.tensor([float(board.payoff) for board in self._state_dict.keys()], dtype=t.float32)
        amplitudes = t.stack([amp for amp in self._state_dict.values()])
        probs = amplitudes.abs() ** 2
        normalizer = probs.sum().real + 1e-10
        payoff_tensor = (payoffs @ probs.real) / normalizer
        # Gradient-preserving fix: keep as tensor, don't convert to float
        p0_payoff = payoff_tensor.real if t.is_complex(payoff_tensor) else payoff_tensor
        return t.stack([p0_payoff, -p0_payoff])
    


class QuantumBoardGameEnvOld(GameEnv):
    def __init__(
        self,
        board_cls: type[Board],
        state: Dict[Board, Tensor] | None = None,
    ):
        self.board_cls = board_cls
        self.state = self.create_initial_state() if state is None else state
        assert isinstance(self.state, dict), "State must be a mapping from boards to amplitudes"
        if state is not None:
            boards = list(self.state.keys())
            assert all(isinstance(board, self.board_cls) for board in boards), "All boards should be of the same type"
            first_player = boards[0].player if boards else 1
            assert all(board.player == first_player for board in boards), "All boards should have the same player"
        self.player = self._get_player()
        self.terminal = self._is_terminal()
        self.payoff = self._calculate_payoff() if self.terminal else None

        # other static variables
        # fixing the environment to be a zero-sum game
        self.num_stages = self.board_cls.num_stages
        self.num_players = 2
        
    def create_initial_state(self):
        initial_board = self.board_cls.create_initial_state()
        return {initial_board: t.tensor(1.0 + 0.0j, dtype=t.complex64)}

    def _get_player(self):
        # All boards should have the same player
        try:
            first_board = next(iter(self.state.keys()))
        except StopIteration:
            return 1
        return first_board.player

    def _is_terminal(self):
        return all(board.terminal for board in self.state.keys())

    def _calculate_payoff(self):
        payoffs = t.tensor([float(board.payoff) for board in self.state.keys()], dtype=t.float32)
        amplitudes = t.stack([amp for amp in self.state.values()])
        probs = amplitudes.abs() ** 2
        normalizer = probs.sum().real + 1e-10
        payoff_tensor = (payoffs @ probs.real) / normalizer
        return payoff_tensor.real

    def _is_legal_action(self, action: Complex[Tensor, "site"]):
        try:
            first_board = next(iter(self.state.keys()))
        except StopIteration:
            return False
        if action.shape != (first_board.num_sites,):
            return False
        if action.dtype != t.complex64:
            return False
        if not t.isclose(t.norm(action), t.tensor(1.0, dtype=t.float32), rtol=1e-6, atol=1e-6):
            return False

        return True

    def move(self, action: Complex[Tensor, "site"]):
        assert self._is_legal_action(action), "Illegal action"

        new_state: dict[Board, Tensor] = {}
        for board, amp in self.state.items():
            if board.terminal:
                new_state[board] = new_state.get(board, t.tensor(0.0 + 0.0j, dtype=t.complex64)) + amp
                continue
            for i in range(len(action)):
                if board._is_legal_action(i):
                    child_board = board.move(i)
                    contribution = amp * action[i]
                    new_state[child_board] = new_state.get(child_board, t.tensor(0.0 + 0.0j, dtype=t.complex64)) + contribution

        if not new_state:
            raise ValueError("Zero Amplitude, state completely annihilated")

        total_amplitude = t.sum(t.stack([amp for amp in new_state.values()]).abs() ** 2).real
        if total_amplitude <= 1e-14:
            raise ValueError("Zero Amplitude, state completely annihilated")
        
        return self.__class__(
            board_cls=self.board_cls,
            state=new_state,
        )


            
