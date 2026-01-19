#%% Imports
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import pandas as pd
#%% Load data
data = pd.read_pickle("Data/loss_game_comparison/LossGame1D_20260119_231627.pkl")
print(data.keys())
#%% Plot payoff curves
data['results']
# %%
