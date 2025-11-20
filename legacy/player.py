import torch
import torch as T
import itertools
import numpy as np
import copy
import time
import pdb
import torch.nn.functional as F
import pandas as pd


class ExactAgent:
    def __init__(self):
        self.policies = torch.rand(
            9, 9
        )  # initialize the policies (strategies) to be uniformly random, The last step is completely determined by the previous moves
        self.policies = self.policies / self.policies.sum(
            axis=1, keepdims=True
        )  # and normalize

        # static knowledge of the agent
        self.actions = list(itertools.product(range(3), range(3)))
        self.all_boards = self.get_legal_boards()
        self.board_to_idx, self.idx_to_board = self.get_board_index_dicts()

    def get_legal_boards(self):
        """
        Get all board configurations that could appear in a gameplay. The output is a 10-element list, consisting of legal boards at each stage, sorted in lexicographic order
        """
        # all board configurations 3^9 = 19683
        states = [0, 1, -1]
        all_boards = list(itertools.product(states, repeat=9))
        all_boards = [list(config) for config in all_boards]

        # check that the board is legal:
        legal_boards = [
            board for board in all_boards if np.sum(board) == 0 or np.sum(board) == 1
        ]
        legal_boards = [
            board
            for board in legal_boards
            if self.is_legal_board(
                board,
                line_win_configs=[
                    (0, 1, 2),
                    (3, 4, 5),
                    (6, 7, 8),
                    (0, 3, 6),
                    (1, 4, 7),
                    (2, 5, 8),
                ],
            )
        ]

        board_layer = [[] for _ in range(10)]
        for board in legal_boards:
            board = np.array(board)
            board_layer[sum(abs(board))].append(board.reshape([3, 3]))

        # return lexicographically ordered boards
        board_layer = [
            sorted(boards, key=lambda x: tuple(x.flatten())) for boards in board_layer
        ]

        return board_layer

    def is_legal_board(
        self,
        board,
        line_win_configs=[
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
        ],
    ):
        """
        Checking whether a board is legal by excluding both-win scenarios
        """
        is_legal = True
        if sum([1 if board[i] != 0 else 0 for i in range(len(board))]) >= 5:
            # exclude simultaneous wins
            x_win = False
            o_win = False
            for i in range(len(line_win_configs)):
                line_check = [board[line_win_configs[i][j]] for j in range(3)]
                if sum(line_check) == 3:
                    x_win = True
                elif sum(line_check) == -3:
                    o_win = True

            is_legal = not (x_win and o_win)

        return is_legal

    def is_terminal_board(
        self,
        board,
        line_win_configs=[
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
            (0, 4, 8),  # NOTE: fixed the bug by updating the new winning configurations
            (2, 4, 6),
        ],
        output_result=False,
    ):
        """
        Checking whether a board configuration is a terminal state, with an option of outputing game outcome (payoff to player X) if the game is terminal
        """
        is_terminal = False
        if sum([1 if board[i] != 0 else 0 for i in range(len(board))]) >= 5:
            # exclude simultaneous wins
            x_win = False
            o_win = False
            for i in range(len(line_win_configs)):
                line_check = [board[line_win_configs[i][j]] for j in range(3)]
                if sum(line_check) == 3:
                    x_win = True
                elif sum(line_check) == -3:
                    o_win = True

            if x_win and o_win:
                raise ValueError("The board is illegal")
            elif x_win or o_win:
                is_terminal = True
            else:
                if sum([1 if board[i] != 0 else 0 for i in range(len(board))]) == 9:
                    is_terminal = True

        if output_result and is_terminal:
            if x_win:
                output = 1
            elif o_win:
                output = -1
            else:
                output = 0

        elif output_result and not is_terminal:
            # warnings.warn("Requested output at non-terminal states, output only is_terminal")
            output = None

        if output_result:
            return is_terminal, output
        else:
            return is_terminal

    def get_board_index_dicts(self):
        """
        Index boards at each stage. Return two lists dicts mapping boards to indices and vice-versa.
        """
        board_to_idx = [
            {
                tuple(board.flatten()): idx
                for idx, board in enumerate(self.all_boards[i])
            }
            for i in range(len(self.all_boards))
        ]  # for faster searches

        idx_to_board = [
            {value: key for key, value in board_to_idx[i].items()}
            for i in range(len(self.all_boards))
        ]

        return board_to_idx, idx_to_board

    def get_next_boards(self, board):
        """
        Given a board configuration, if it is not terminal, simulate next action with the renormalized probability distribution. Return all possible next boards and corresponding probabilities.
        """
        if not self.is_terminal_board(board):
            vacancies = [i for i in range(len(board)) if board[i] == 0]
            stage = 9 - len(vacancies)
            player = 1 if sum(board) == 0 else -1
            next_boards = []
            probs = []
            for i in vacancies:
                next_board = list(copy.deepcopy(board))
                next_board[i] = player
                next_boards.append(tuple(next_board))

                prob = self.policies[stage][i] / self.policies[stage][vacancies].sum()
                probs.append(prob)

            return next_boards, probs
        else:
            return [], []

    def policy_to_mat(self):
        """
        From given policy, generate matrices that 'forward the game', i.e. mapping a probability distribution of board configurations at stage i to stage i+1
        """
        # board_to_idx, idx_to_board = self.get_board_index_dicts()
        sizes = [len(self.board_to_idx[i]) for i in range(len(self.board_to_idx))]
        policy_mats = []
        for stage in range(9):
            # initialize
            n_row = sizes[stage + 1]
            n_col = sizes[stage]
            M = torch.zeros(n_row, n_col)

            for i in range(n_col):
                # for each of the previous state, get next board and assign the corresponding renormalized probabilities
                board = self.idx_to_board[stage][i]
                next_boards, probs = self.get_next_boards(board)
                for next_board, prob in zip(next_boards, probs):
                    M[self.board_to_idx[stage + 1][next_board]][i] = prob

            policy_mats.append(M)

        # regarding early-terminated states: these will be zero rows carrying forward...
        return policy_mats

    def assign_probs(self):
        """
        Given policy (from both players) generate the probabilities for every intermediate and terminal board configruations by forwarding the game
        """
        # all_boards = self.get_legal_boards()
        policy_mats = self.policy_to_mat()
        probs = []

        # initialize for the 0th stage
        prob = torch.tensor([1.0])
        probs.append(prob)
        for stage in range(9):
            # initialize the probabilities
            prob = torch.matmul(policy_mats[stage], prob)
            probs.append(prob)

        return probs

    def assign_utilities(self):
        """
        Given probabilities (computed from assign_probs) generate the utilities for every intermediate and the starting mode. The utility for each mode is defined by the expected utility player X get by starting the game at this node and assume that both players follow the current policy. The calculation is propagating backwards. The utility at the empty board at start would be the payoff (rsep. cost) of the player X (resp. player O).
        """
        board_to_idx, idx_to_board = self.get_board_index_dicts()
        probs = self.assign_probs()
        policy_mats = self.policy_to_mat()
        # initialize the values on the end layer
        utils = []
        util = torch.zeros(len(probs[-1]))
        for i in range(len(board_to_idx[-1])):
            _, output = self.is_terminal_board(idx_to_board[-1][i], output_result=True)

            util[i] = output  # all of the states are terminal states...

        utils.append(util)
        # pdb.set_trace()
        for reversed_stage in range(2, 11):

            util = torch.matmul(util, policy_mats[-reversed_stage + 1])
            # need to consider early-terminated boards

            for i in range(len(idx_to_board[-reversed_stage])):
                is_terminated, output = self.is_terminal_board(
                    idx_to_board[-reversed_stage][i], output_result=True
                )
                if is_terminated:
                    # test for util[i] = 0?
                    util[i] = output

            utils.append(util)

        utils.reverse()

        # pdb.set_trace()
        return utils

    ################

    # the following method is not used in the actual gradient descent.

    ################

    def get_gradient(self, policy_mat, probs, utils, lr=6e-4):
        # the data for policy_mat, probs and utils should be computed in advance
        moves = (
            1
            / torch.sqrt(torch.tensor(8 / 9))
            * (torch.eye(9) - 1 / 9 * torch.ones((9, 9)))
        )
        expected_utility = utils[0][0]
        grad_vecs = torch.zeros((9, 9))
        for stage in range(9):
            current_probs = probs[stage]
            next_probs = probs[stage + 1]
            next_utils = utils[stage + 1]
            current_policy = self.policies[stage]

            for i in range(9):
                right_policy = current_policy + moves[i] * lr
                new_next_probs = self.get_next_probs(current_probs, stage, right_policy)
                grad_vecs[stage][i] = torch.dot(
                    (new_next_probs - next_probs),
                    (next_utils - expected_utility * torch.ones(next_utils.shape)),
                )

        return grad_vecs

    def get_next_probs(self, current_probs, stage, new_policy):
        # board_to_idx, idx_to_board = self.get_board_index_dicts()
        n_row = len(self.board_to_idx[stage + 1])
        n_col = len(self.board_to_idx[stage + 1])
        M = torch.zeros(n_row, n_col)

        cache_policy = copy.deepcopy(self.policies[stage])
        self.policies[stage] = new_policy

        for i in range(n_col):
            board = self.idx_to_board[stage][i]
            next_boards, probs = self.get_next_boards(board)
            for next_board, prob in zip(next_boards, probs):
                M[self.board_to_idx[stage + 1][next_board]][i] = prob

        # restore changes
        self.policies[stage] = cache_policy
        next_probs = torch.matmul(M, current_probs)

        return next_probs


class QAgent(ExactAgent):
    def __init__(
        self,
        cpol=T.complex(
            T.rand(9, 9, requires_grad=True), T.rand(9, 9, requires_grad=True)
        ),
    ):
        super().__init__()
        # re = T.rand(9, 9, requires_grad=True)
        # im = T.rand(9, 9, requires_grad=True)
        self.cpol = cpol

        # normalize using L2 norms
        norm = T.sqrt(T.sum(T.abs(self.cpol) ** 2, axis=1, keepdim=True))
        self.cpol = self.cpol / norm
        # t0 = time.time()
        self.amps = self.assign_amps()

        self.exp_util, _, _ = self.get_exp_util()
        # t1 = time.time()
        # print(f"Time to compute expected utility: {t1-t0}")

    def get_next_boards(self, board):
        if not self.is_terminal_board(board):
            vacancies = [i for i in range(len(board)) if board[i] == 0]
            stage = 9 - len(vacancies)
            player = 1 if sum(board) == 0 else -1

            next_boards = []
            amps = []

            for i in vacancies:
                next_board = list(copy.deepcopy(board))
                next_board[i] = player
                next_boards.append(tuple(next_board))

                amp = self.cpol[stage][i]
                amps.append(amp)

            return next_boards, amps
        else:
            return [], []

    def policy_to_mat(self):
        sizes = [len(self.board_to_idx[i]) for i in range(len(self.board_to_idx))]
        cpol_mats = []

        for stage in range(9):
            n_row = sizes[stage + 1]
            n_col = sizes[stage]
            M = T.zeros(n_row, n_col, dtype=T.cfloat)
            for i in range(n_col):
                board = self.idx_to_board[stage][i]
                next_boards, amps = self.get_next_boards(board)
                for next_board, amp in zip(next_boards, amps):
                    M[self.board_to_idx[stage + 1][next_board]][i] = (
                        M[self.board_to_idx[stage + 1][next_board]][i] + amp
                    )
            cpol_mats.append(M)
        return cpol_mats

    def assign_amps(self, normalize=False):
        # If we do not normalize the state, the final state would be
        cpol_mats = self.policy_to_mat()
        amps = []
        amp = T.tensor([1.0 + 0.0j])
        amps.append(amp)

        for stage in range(9):
            amp = T.matmul(cpol_mats[stage], amp)

            if normalize:
                norm = T.sqrt(T.sum(T.abs(amp) ** 2))
                amp = amp / norm

            amps.append(amp)

        return amps

    def get_exp_util(self):
        # collect the amplitudes on terminal states
        term_amp = []
        term_output = []
        term_board = []
        for stage in range(5, 10):  # when termination is possible
            for i in range(len(self.idx_to_board[stage])):
                is_terminated, output = self.is_terminal_board(
                    self.idx_to_board[stage][i], output_result=True
                )

                if is_terminated:
                    term_amp.append(self.amps[stage][i])
                    term_output.append(output)
                    term_board.append(self.idx_to_board[stage][i])

        # term_amp = T.tensor(term_amp, dtype=T.cfloat)
        # norm2 = T.sum(T.abs(term_amp)**2)
        # term_prob = T.abs(term_amp)**2 / norm2
        # term_output = T.tensor(term_output, dtype=T.float)

        ####################################################
        # NOTE: USE STACK INSTEAD OF REPLICATING TENSORS TO PRESERVE GRADIENTS
        term_amp = T.stack(term_amp)
        ####################################################
        norm2 = T.sum(T.abs(term_amp) ** 2)
        term_prob = T.abs(term_amp) ** 2 / norm2
        term_output = T.tensor(term_output, dtype=T.float)

        exp_util = T.dot(term_prob, term_output)

        return exp_util, term_prob, term_output


if __name__ == "__main__":
    T.manual_seed(42)
    # agent = QAgent()

    actions = T.randn(9, 9, dtype=T.float64, requires_grad=True)
    actions = actions / actions.norm(dim=1, keepdim=True)

    agent = QAgent(cpol=actions)
    # print(agent.cpol)
    print(agent.exp_util)
