from environment import LossGame1DEnv, QuantumBoardGameEnv, SimpleBoard
from solver import VanillaGradientUpdate, LocalMinimaxUpdate, LocalStochasticMinimaxUpdate, RandomMinimaxUpdate
import torch as t
from tqdm import tqdm
from mcts import MCTS
from spaces import ActionSpace, BoxActionSpace, ComplexProjectiveActionSpace
import math
import time


def test_vanilla_gradient_update():
    env = LossGame1DEnv(num_stages=3, num_players=3, d_action=1)
    env.move(t.tensor([0.0], requires_grad=True))
    env.move(t.tensor([0.0], requires_grad=True))
    env.move(t.tensor([0.0], requires_grad=True))

    print(env.payoff)

    for _ in range(10):
        update = VanillaGradientUpdate(env, learning_rate=0.1, clip_grad_norm=1.0)
        new_env = update.update()
        print(new_env.state)
        env = new_env

def test_local_minimax_update():
    env = LossGame1DEnv(num_stages=3, num_players=3, d_action=1)
    # for stage in range(env.num_stages):
    #     env.move(t.tensor([0.3], requires_grad=True))
    env = env.move(t.tensor([0.85], requires_grad=False))
    env = env.move(t.tensor([0.5], requires_grad=False))
    env = env.move(t.tensor([-1.09], requires_grad=False))
    print(env.terminal)
    print(f"Initial actions: {env.state}")
    print(f"Initial payoff: {env.payoff}")


    action_space = BoxActionSpace(d_action=1, lows=[-5.0], highs=[5.0])
    updater = LocalStochasticMinimaxUpdate(env, action_space, learning_rate=0.01, max_branching_factor=10)
    # updater = RandomMinimaxUpdate(env, action_space, max_branching_factor=10)
    new_env = updater.update()
    print(f"Updated actions: {new_env.state}")
    print(f"Updated payoff: {new_env.payoff}")

def test_mcts():
    env = LossGame1DEnv(num_stages=3, num_players=3, d_action=1)
    mcts = MCTS(env, action_space=BoxActionSpace(d_action=1, lows=[-5.0], highs=[5.0]), num_simulations=5000)
    actions = []
    root = env

    for _ in range(env.num_stages):
        mcts.run()
        abstract_action, action = mcts.select_action()
        root = root.move(action)
        actions.append(action)
        mcts.reroot((abstract_action,), root)
    
    print(actions)
    print(mcts.env.payoff)


def test_quantum_board():
    env = QuantumBoardGameEnv(board_cls=SimpleBoard)
    
    actions = [
        t.tensor([0, 0, 1, 0, 0], dtype=t.complex64),
        t.tensor([1/math.sqrt(2), 0, 0, 0, 1/math.sqrt(2)], dtype=t.complex64),
        t.tensor([1/2, 0, 1/2, 1/math.sqrt(2), 0], dtype=t.complex64),
    ]    
    for action in actions:
        env = env.move(action)
    
    print(env.terminal)
    print(env.payoff)

def test_action_sapces():
    cp = ComplexProjectiveActionSpace(d_action=5)
    action = cp.sample(num_samples=1)
    print(action)

    new_actions = cp.sample_local(num_samples=2, action=action, step_size=0.1)
    print(new_actions)

def test_local_stochastic_minimax_update():
    env =  QuantumBoardGameEnv(board_cls=SimpleBoard)
    action_space = ComplexProjectiveActionSpace(d_action=5)
    actions = [
        t.tensor([0, 0, 1, 0, 0], dtype=t.complex64),
        t.tensor([1/math.sqrt(2), 0, 0, 0, 1/math.sqrt(2)*1j], dtype=t.complex64),
        t.tensor([0, 1/math.sqrt(2), 0, 1/math.sqrt(2), 0], dtype=t.complex64),
    ]
    for action in actions:
        env = env.move(action)
    
    print(f"Initial actions: {env.state}")
    print(f"Initial payoff: {env.payoff}")
    
    t0 = time.time()
    updater = LocalStochasticMinimaxUpdate(env, action_space, learning_rate=0.01, max_branching_factor=10)
    new_env = updater.update()
    print(f"Updated actions: {new_env.state}")
    # print(new_env._state_dict)
    print(f"Updated payoff: {new_env.payoff}")
    t1 = time.time()
    print(f"Time taken: {t1 - t0} seconds")


if __name__ == "__main__":
    test_local_stochastic_minimax_update()
    # test_local_minimax_update()


