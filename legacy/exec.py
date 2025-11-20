import random
import numpy as np
import torch as T

# import matplotlib.pyplot as plt
import joblib, os
import pandas as pd

from games import *
from DBI import *
from cmcts import *
from IBR import *


def format_learning_rate(lr):
    """Convert learning rate to string format suitable for filenames."""
    return f"{lr:.0e}".replace("-", "m")


def format_filename(value):
    return "{:.2f}".format(value).replace(".", "_").replace("-", "n")


################################################################################
############# SETUP ############################################################
################################################################################

game_mode = "quantum_board"
solver_mode = "dbi"

################################################################################
############# GAME CONFIGURATIONS ##############################################
################################################################################


################################################################################

# Random Loss Game with 1D action spaces

if game_mode == "rand_loss":
    game = RandGame()
    actions = T.tensor([[-1, 1, -1.09]], dtype=T.float64, requires_grad=True)
    init_actions = actions.transpose(0, 1)
    LEARNING_RATE = 0.001
    MAX_ITER = 6000
    PRINT_EVERY = 50
    path_to_dir = "Data/rand_loss/"


################################################################################

# Stackelberg Game

if game_mode == "stackelberg":
    game = Stackelberg(4)
    actions = T.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=T.float64)
    init_actions = actions.transpose(0, 1)
    LEARNING_RATE = 0.001
    MAX_ITER = 800
    PRINT_EVERY = 25
    path_to_dir = "Data/stackelberg_game/"

################################################################################

# Quantum Board Game

if game_mode == "quantum_board":
    game = QuantumSimpleBoard()
    statespace = StateSpace
    actions = T.randn((3, 5), dtype=T.float64)
    init_actions = actions / actions.norm(dim=1, keepdim=True)
    LEARNING_RATE = 0.001
    MAX_ITER = 100
    PRINT_EVERY = 5
    path_to_dir = "Data/quantum_board/"

################################################################################

# Classical Board

if game_mode == "classical_board":
    game = ClassicalSimpleBoard()
    statespace = clStateSpace
    actions = T.rand((3, 5), dtype=T.float64)
    init_actions = actions / actions.sum(dim=1, keepdim=True)
    LEARNING_RATE = 0.001
    MAX_ITER = 1000
    path_to_dir = "Data/classical_board/"


################################################################################
############# SOLVER CONFIGURATIONS ############################################
################################################################################


################################################################################

# Iterative Best Response

if solver_mode == "ibr":
    solver = IterBR(game, learning_rate=LEARNING_RATE, max_iter=MAX_ITER).loop
    fstr = f"IBR_lr{format_learning_rate(LEARNING_RATE)}_maxiter{MAX_ITER}_time{datetime.now().strftime('%H%M%S')}"

    def collect(solver, init_actions, path_to_dir, fstr):
        action_history = []
        payoff_history = []

        new_actions, action_history, payoff_history = solver(init_actions)

        data = {"iter": [], "action": [], "payoff": []}
        for i, (action, payoff) in enumerate(zip(action_history, payoff_history)):
            data["iter"].append(i)
            data["action"].append(action)
            data["payoff"].append(payoff)

        df = pd.DataFrame(data)

        os.makedirs(path_to_dir, exist_ok=True)
        joblib.dump(
            df,
            f"{path_to_dir}" + fstr + ".pkl",
        )
        pass

    collect(solver, init_actions, path_to_dir, fstr)


################################################################################

# Differential Backward Induction

if solver_mode == "dbi":
    MINIBATCH_SIZE = 5
    solver = lambda actions: DiffBP(
        game, learning_rate=LEARNING_RATE, max_iter=MAX_ITER
    ).train(minibatch_size=MINIBATCH_SIZE, init_action=actions, print_every=PRINT_EVERY)
    fstr = f"DBI_lr{format_learning_rate(LEARNING_RATE)}_maxiter{MAX_ITER}_minibatch{MINIBATCH_SIZE}_time{datetime.now().strftime('%H%M%S')}"

    def collect(solver, init_actions, path_to_dir, fstr):
        new_action, data = solver(init_actions)
        os.makedirs(path_to_dir, exist_ok=True)
        joblib.dump(
            data,
            f"{path_to_dir}" + fstr + ".pkl",
        )
        pass

    collect(solver, init_actions, path_to_dir, fstr)

################################################################################

# Monte Carlo Tree Search

if solver_mode == "mcts":
    EXPLORATION_WEIGHT = 1.0
    GRADIENT_MODE = "dbi"
    gradient_update_frequency = 1

    # some tunable params
    # currently for loss game only
    num_init_branch = 10
    num_simulations = np.logspace(2, 4, 50, dtype=int)

    fstr = (
        f"MCTS_ew{format_filename(EXPLORATION_WEIGHT)}_initbranch_{num_init_branch}_gf_{format_filename(gradient_update_frequency)}"
        + GRADIENT_MODE
        + f"_time{datetime.now().strftime('%H%M%S')}"
    )
    T.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    meta_data = GameMetaData(
        branching_ratio=[num_init_branch] * len(game.stages), depth=len(game.stages)
    )
    action_space = SphereActionSpace(dim=5, meta_data=meta_data)
    realized_game = GameRealization(meta_data, action_space, statespace)
    # node = AbstractGameNode.reset()
    # agent = cMCTS(node, game)

    action_history = []
    value_history = []
    for n_sim in num_simulations:
        print(f"Running {n_sim} simulations")
        node = AbstractGameNode.reset()
        for stage in range(meta_data.depth):
            agent = cMCTS(
                node=node,
                agent=realized_game,
                exploration_weight=EXPLORATION_WEIGHT,
                gradient_mode=GRADIENT_MODE,
            )
            node = agent.choose(node, num_simulations=n_sim)

        action_history.append(agent.agent.get_realized_actions(node.abstract_state))
        value_history.append(realized_game.reward(node.abstract_state))

    data = {
        "num_simulations": num_simulations,
        "action": action_history,
        "value": value_history,
    }
    df = pd.DataFrame(data)
    os.makedirs(path_to_dir, exist_ok=True)
    joblib.dump(
        df,
        f"{path_to_dir}" + fstr + ".pkl",
    )

if solver_mode == "mcts_vg":
    pass
