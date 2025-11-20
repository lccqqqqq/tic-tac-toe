from games import *
from cmcts import *
from exec_mcts_benchmark import mcts_setup, mcts_collect, format_filename

import joblib
import os
import pandas as pd
import random
import numpy as np
import torch as T
import datetime
# from mpi4py import MPI
import time
import argparse


def chunkify(iter_start, iter_end, num_datapoints, num_chunk):
    if num_datapoints != num_chunk:
        chunk_size = num_datapoints // num_chunk
        num_simulations = np.linspace(
            iter_start, iter_end, num_datapoints, dtype=int
        ).reshape(2, num_datapoints // 2)
        num_simulations[1, :] = num_simulations[1, ::-1]

        chunk_simus = []
        for rk in range(num_chunk):
            chunk_simus.append(
                num_simulations[
                    :, chunk_size // 2 * rk : chunk_size // 2 * (rk + 1)
                ].flatten()
            )

        return chunk_simus

    if num_datapoints == num_chunk:
        num_simulations = np.linspace(iter_start, iter_end, num_datapoints, dtype=int)
        return [[num_simulations[i]] for i in range(num_chunk)]
    

# def main():
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     size = comm.Get_size()

#     print(f"Initializing Game Data")
#     T.manual_seed(42)
#     game = QuantumTicTacToe()
#     States = QTTTStateSpace
#     path_to_dir = "Data/quantum_tic_tac_toe/"
#     os.makedirs(path_to_dir, exist_ok=True)
#     meta_data, realized_game = mcts_setup(game, States)

#     ##############################

#     # Define the parameters for the MCTS

#     learning_rate = 2e-4
#     gradient_update_frequency = 0
#     gradient_update_mode = "none"

#     ##############################

#     chunk_simus = chunkify(iter_start=20, iter_end=600, num_datapoints=40, num_chunk=20)

#     local_result = []

#     for n_sim in chunk_simus[rank]:
#         # print(f"Rank {rank} running {n_sim} simulations")
#         t0 = time.time()
#         actions, value, agent = mcts_collect(
#             meta_data,
#             realized_game,
#             num_mcts_sims=n_sim,
#             learning_rate=learning_rate,
#             gradient_update_mode=gradient_update_mode,
#             gradient_update_frequency=gradient_update_frequency,
#         )
#         local_result.append(
#             {
#                 "num_simulations": n_sim,
#                 "action": actions,
#                 "value": value,
#             }
#         )
#         t1 = time.time()
#         print(f"Worker {rank} finished {n_sim} simulations, time taken: {t1-t0:.2g}s.")

#     all_results = comm.gather(local_result, root=0)

#     if rank == 0:
#         all_results = [item for sublist in all_results for item in sublist]

#         data = pd.DataFrame(all_results)

#         joblib.dump(
#             data,
#             f"{path_to_dir}"
#             + format_filename(
#                 learning_rate, gradient_update_mode, gradient_update_frequency
#             ),
#         )
        


def create_config_file(
    ns: int = 4,
    num_simu_low = 10, # The range of n_sim
    num_simu_high = 15,
):
    lr = 2e-4
    grad_freq = [0, 0.15, 0.2]
    grad_mode = [0, 1, 1]
    # ns = 4
    num_simu = np.linspace(num_simu_low, num_simu_high, ns, dtype=int)
    seed = np.random.randint(0, 1000, ns)
    
    headerstr = "/usr/bin/python3 exec_cmcts_qt.py "
    
    grad_info = zip(grad_freq, grad_mode)
    params_str = []
    
    for (i, (gf, gm)) in enumerate(grad_info):
        for j in range(ns):
            str_param = headerstr
            str_param += f"--lr {lr} "
            str_param += f"--grad_freq {gf} "
            str_param += f"--grad_mode {gm} "
            str_param += f"--num_simu {num_simu[j]} "
            str_param += f"--seed {seed[j]}"
            str_param += "\n"
            
            params_str.append(str_param)
    
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config_file.txt")
    
    with open(config_path, "w") as config_file:
        for param in params_str:
            config_file.write(param)
    
    config_file.close()

    

if __name__ == "__main__":
       # using the config file and multirun functionality in Hydra
    # To NOT abandon the entire job if one oom_kill event is identified
    # This uses a configuration file to run the job
    # create_config_file(
    #     ns=40,
    #     num_simu_low=20,
    #     num_simu_high=400,
    # )
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--grad_freq", type=float, default=0, help="Gradient update frequency")
    parser.add_argument("--grad_mode", type=int, default=0, help="Gradient update mode")
    parser.add_argument("--num_simu", type=int, default=10, help="Number of simulations")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed")
    
    args = parser.parse_args()
    if args.grad_mode == 0:
        GRAD_MODE = "none"
    elif args.grad_mode == 1:
        GRAD_MODE = "vanilla"
    else:
        raise ValueError(f"Invalid gradient mode, get {args.grad_mode}")
    
    T.manual_seed(args.seed)
    game = QuantumTicTacToe()
    States = QTTTStateSpace
    path_to_dir = "Data/quantum_tic_tac_toe/"
    os.makedirs(path_to_dir, exist_ok=True)
    meta_data, realized_game = mcts_setup(game, States)
    
    print(f"Running {args.num_simu} simulations")
    
    
    t0 = time.time()
    actions, value, agent = mcts_collect(
        meta_data,
        realized_game,
        num_mcts_sims=args.num_simu,
        learning_rate=args.lr,
        gradient_update_mode=GRAD_MODE,
        gradient_update_frequency=args.grad_freq,
    )
    t1 = time.time()
    
    print(f"Finished {args.num_simu} simulations, time taken: {t1-t0:.2g}s.")
    
    result = {
        "num_simulations": args.num_simu,
        "grad_freq": args.grad_freq,
        "grad_mode": GRAD_MODE,
        "lr": args.lr,
        "action": actions,
        "value": value,
        "agent": agent,
    }
    
    joblib.dump(
        result,
        f"{path_to_dir}"
        + f"nsimu{args.num_simu}_"
        + format_filename(
            args.lr, GRAD_MODE, args.grad_freq
        )
    )
