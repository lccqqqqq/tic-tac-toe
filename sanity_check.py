"""Quick sanity check for loss game solver implementation."""

import sys
import os
# Add tests directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests'))

from test_loss_game_solvers import GAME_CONFIGS, SOLVER_CONFIGS
from environment import LossGame1DEnv
from spaces import BoxActionSpace
from mcts import MCTS
import torch as t

print('='*60)
print('SANITY CHECK: Loss Game Solver Implementation')
print('='*60)

# Test imports
print('\n1. Testing imports... OK')

# Test game config
game_config = GAME_CONFIGS['LossGame1D']
print(f'2. Game config loaded: {game_config["env_cls"].__name__}')

# Test environment creation
env = game_config['env_cls'](**game_config['env_kwargs'])
print(f'3. Environment created: {env.__class__.__name__}')
print(f'   - num_stages: {env.num_stages}')
print(f'   - num_players: {env.num_players}')
print(f'   - d_action: {env.d_action}')

# Test action space
action_space = BoxActionSpace(**game_config['action_space_kwargs'], seed=42)
print(f'4. Action space created')
print(f'   - d_action: {action_space.d_action}')
print(f'   - bounds: {action_space.lows} to {action_space.highs}')

# Test MCTS with minimal simulations
print('\n5. Testing MCTS with 10 simulations...')
mcts = MCTS(env, action_space, num_simulations=10, seed=42)
mcts.run()
print('   MCTS run completed successfully!')

# Get best action
best_child, best_action = mcts.select_action()
print(f'   Best child (abstract node): {best_child}')
print(f'   Best action shape: {best_action.shape if hasattr(best_action, "shape") else "scalar"}')

print('\n' + '='*60)
print('SANITY CHECK PASSED!')
print('='*60)
