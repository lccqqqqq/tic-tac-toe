# Description: Benchmarking MCTS, MCTS-VG and MCTS-DBI on the quantum board game to evaluate performance

from games import *
from DBI import *
from cmcts import *
from IBR import *

import joblib, os
import pandas as pd
import random
import numpy as np
import torch as T
import datetime
from tqdm import tqdm

def mcts_setup(game: ExtensiveFormGame, states, num_init_branches: int = 4):
    meta_data = GameMetaData(
        branching_ratio=[num_init_branches] * len(game.stages), depth=len(game.stages)
    )
    action_space = SphereActionSpace(dim=game.dim, meta_data=meta_data)
    realized_game = GameRealization(meta_data, action_space, states)

    return meta_data, realized_game


def mcts_collect(
    meta_data: GameMetaData,
    realized_game: GameRealization,
    exploration_weight: float = 1.0,
    num_mcts_sims: int = 100,
    learning_rate: float = 1e-2,
    gradient_update_mode: str = "none",
    gradient_update_frequency: float = 0.0,
) -> pd.DataFrame:
    """
    Collect data for vanilla MCTS algorithm.
    """
    # NOTE: functionality for zero-sum games only

    node = AbstractGameNode.reset()
    agents = []
    for stage in range(meta_data.depth):
        agent = cMCTS(
            node=node,
            agent=realized_game,
            exploration_weight=exploration_weight,
            gradient_mode=gradient_update_mode,
            gradient_update_frequency=gradient_update_frequency,
            learning_rate=learning_rate,
        )
        node = agent.choose(node, num_simulations=num_mcts_sims)
        agents.append(agent)
        

    realized_actions = agent.agent.get_realized_actions(node.abstract_state)
    value = realized_game.reward_with_action(realized_actions)

    return realized_actions, value, agents


def format_filename(
    learning_rate: float, gradient_update_mode: str, gradient_update_frequency: float
):
    suffix = "MCTS-"

    if gradient_update_mode == "none":
        mode = "no_grad"
        lr = ""
    elif gradient_update_mode == "dbi":
        mode = f'DBI-freq_{"{:.2f}".format(gradient_update_frequency).replace(".", "_").replace("-", "n")}'
        lr = f'lr_{f"{learning_rate:.0e}".replace("-", "m")}'
    elif gradient_update_mode == "vanilla":
        mode = f'vgd-freq_{"{:.2f}".format(gradient_update_frequency).replace(".", "_").replace("-", "n")}'
        lr = f'lr_{f"{learning_rate:.0e}".replace("-", "m")}'
    else:
        raise ValueError("Invalid gradient update mode")

    timestr = datetime.datetime.now().strftime("%H%M%S")
    extention = ".pkl"

    return suffix + mode + lr + timestr + extention


if __name__ == "__main__":
    T.manual_seed(42)

    # Input arguments and the path to save data

    game = QuantumSimpleBoard()
    States = StateSpace
    path_to_dir = "Data/quantum_board/"
    os.makedirs(path_to_dir, exist_ok=True)

    meta_data, realized_game = mcts_setup(game, States)

    # num_simulations = np.linspace(300, 304, 1, dtype=int)
    num_simulations = [1000]
    learning_rate = 1e-3
    gradient_update_frequency = 2
    gradient_update_mode = "vanilla"

    data = []
    for n_sim in tqdm(num_simulations):
        print(f"Running {n_sim} simulations")
        actions, value, agent = mcts_collect(
            meta_data,
            realized_game,
            num_mcts_sims=n_sim,
            learning_rate=learning_rate,
            gradient_update_mode=gradient_update_mode,
            gradient_update_frequency=gradient_update_frequency,
        )
        data.append(
            {
                "num_simulations": n_sim,
                "action": actions,
                "value": value,
            }
        )

    data = pd.DataFrame(data)
    joblib.dump(
        data,
        f"{path_to_dir}"
        + format_filename(
            learning_rate, gradient_update_mode, gradient_update_frequency
        ),
    )

    # also keeps track of the tree, the agent scores, etc in the cMCTS object
    # joblib.dump(
    #     agent,
    #     f"{path_to_dir}"
    #     + f"agent_{format_filename(learning_rate, gradient_update_mode, gradient_update_frequency)}.pkl",
    # )
