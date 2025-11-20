import torch as t
from jaxtyping import Complex
from torch import Tensor
from collections import defaultdict, namedtuple
from typing import Dict

t.set_grad_enabled(False)

# Define namedtuple for hashing
BoardHashData = namedtuple("BoardHashData", ["board", "terminal", "payoff"])


class Board:
    """Classical tic-tac-toe board represented with torch tensors."""

    def __init__(self, board=None, player: int = 1, num_sites: int = 9):
        self.num_sites = num_sites
        if board is None:
            self.board = t.zeros(self.num_sites, dtype=t.int8)
        elif isinstance(board, Tensor):
            self.board = board.to(dtype=t.int8).clone()
        else:
            self.board = t.tensor(board, dtype=t.int8)
        self.player = player
        self.terminal = self._is_terminal()
        self.payoff = self._calculate_payoff() if self.terminal else None

    @classmethod
    def create_initial_board(cls, num_sites: int = 9):
        return cls(num_sites=num_sites)

    def winning_combos(self):
        yield (0, 1, 2)
        yield (3, 4, 5)
        yield (6, 7, 8)
        yield (0, 3, 6)
        yield (1, 4, 7)
        yield (2, 5, 8)
        yield (0, 4, 8)
        yield (2, 4, 6)

    def move(self, action: int) -> "Board":
        if int(self.board[action]) != 0:
            raise ValueError("Double Occupancy")
        if self.terminal:
            raise ValueError("Game Over")

        new_board = self.board.clone()
        new_board[action] = int(self.player)
        return Board(board=new_board, player=-self.player, num_sites=self.num_sites)

    def _calculate_payoff(self):
        for combo in self.winning_combos():
            if all(int(self.board[i]) == 1 for i in combo):
                return 1
            if all(int(self.board[i]) == -1 for i in combo):
                return -1
        if all(int(self.board[i]) != 0 for i in range(self.num_sites)):
            return 0
        return None

    def _is_terminal(self):
        return self._calculate_payoff() is not None

    def is_game_over(self):
        return self.terminal

    def get_legal_actions(self):
        return [i for i in range(self.num_sites) if int(self.board[i]) == 0]

    def __hash__(self):
        return hash(BoardHashData(tuple(int(x) for x in self.board.tolist()), self.terminal, self.payoff))

    def __eq__(self, other):
        if not isinstance(other, Board):
            return False
        return (
            t.equal(self.board, other.board)
            and self.terminal == other.terminal
            and self.payoff == other.payoff
        )

    def __repr__(self):
        board_str = ''.join(['X' if int(x) == 1 else 'O' if int(x) == -1 else '.' for x in self.board])
        return f"Board('{board_str[:3]}\\n{board_str[3:6]}\\n{board_str[6:9]}', player={self.player})"


class QuantumGameEnv:
    """Quantum superposition over classical boards using torch for amplitudes."""

    def __init__(
        self,
        state: Dict[Board, Tensor] | None = None,
        current_player: int = 1,
        preserve_grad: bool = False,
    ):
        if state is None or len(state) == 0:
            initial_board = Board.create_initial_board()
            self.state: Dict[Board, Tensor] = {initial_board: t.tensor(1.0 + 0.0j, dtype=t.complex64)}
            self.current_player = initial_board.player
        else:
            converted: Dict[Board, Tensor] = {}
            for board, amplitude in state.items():
                if isinstance(amplitude, Tensor):
                    converted[board] = amplitude.to(dtype=t.complex64)
                else:
                    converted[board] = t.tensor(complex(amplitude), dtype=t.complex64)
            self.state = converted
            self.current_player = current_player

        self.preserve_grad = preserve_grad

        self.terminal = all(board.terminal for board in self.state.keys())
        self.payoff = self._calculate_payoff() if self.terminal else None

    @classmethod
    def create_initial_state(cls, preserve_grad: bool = False):
        return cls(
            state={Board.create_initial_board(): t.tensor(1.0 + 0.0j, dtype=t.complex64)},
            current_player=1,
            preserve_grad=preserve_grad,
        )

    def _calculate_payoff(self):
        payoffs = t.tensor([float(board.payoff) for board in self.state.keys()], dtype=t.float32)
        amplitudes = t.stack([amp for amp in self.state.values()])
        probs = amplitudes.abs() ** 2
        normalizer = probs.sum().real + 1e-10
        payoff_tensor = (payoffs @ probs.real) / normalizer
        # explicitly setting payoff to real to make sure we have Wirtinger gradients
        return payoff_tensor.real if self.preserve_grad else float(payoff_tensor)

    def move(
        self,
        action: Complex[Tensor, "site"],
        *,
        preserve_grad: bool | None = None,
    ) -> "QuantumGameEnv":
        if not isinstance(action, Tensor):
            action_tensor = t.tensor(action, dtype=t.complex64)
        else:
            action_tensor = action.to(dtype=t.complex64)
        action_tensor = action_tensor.reshape(-1)

        action_norm = t.sum(action_tensor.abs() ** 2).real
        if not t.isclose(action_norm, t.tensor(1.0, dtype=t.float32), rtol=1e-6, atol=1e-6):
            raise ValueError(f"Action is not normalized: norm = {float(action_norm)}, expected 1.0")

        new_state: defaultdict[Board, Tensor] = defaultdict(lambda: t.zeros((), dtype=t.complex64))
        for board, prob_amp in self.state.items():
            if board.terminal:
                new_state[board] = new_state[board] + prob_amp
            else:
                for i in range(len(action_tensor)):
                    try:
                        new_board = board.move(i)
                        new_state[new_board] = new_state[new_board] + prob_amp * action_tensor[i]
                    except ValueError:
                        continue

        if new_state:
            stacked = t.stack(list(new_state.values()))
            total_amplitude = t.sum(stacked.abs() ** 2).real
        else:
            total_amplitude = t.tensor(0.0)
        if float(total_amplitude) <= 1e-14:
            raise ValueError("Zero Amplitude, state completely annihilated")

        preserve = self.preserve_grad if preserve_grad is None else preserve_grad
        return QuantumGameEnv(
            dict(new_state),
            current_player=-self.current_player,
            preserve_grad=preserve,
        )

class LossGameEnv:
    def __init__(
        self,
        state: tuple[float] | None = None,
        current_player = 0,
        num_stages = 3,
        preserve_grad = False,
    ):
        if state is None:
            self.state = ()
            self.current_player = 0 # overwriting the input current_player
        else:
            self.state = state
            self.current_player = current_player
            assert len(state) == self.current_player + 1
        
        self.preserve_grad = preserve_grad

        assert len(state) <= num_stages
        self.terminal = len(state) == num_stages
        self.payoff = self._calculate_payoff() if self.terminal else None
    
    @classmethod
    def create_initial_state(cls, preserve_grad: bool = False):
        return cls(
            state=(),
            current_player=0,
            preserve_grad=preserve_grad
        )
    
    def _calculate_payoff():
        pass

    def move(
        self,
        action: Tensor,
        *,
        preserve_grad: bool | None = None,
    ) -> "LossGameEnv":
        pass


class LossGameInstance(LossGameEnv):
    def __init__(
        self,
        state: tuple[float],
        current_player: int,
        preserve_grad: bool,
    ):
        super().__init__(state, current_player, preserve_grad)
    
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

        return t.stack([u1, u2, u3])
    
    def move(self, action: Tensor, *, preserve_grad: bool | None = None) -> "LossGameInstance":
        pass



