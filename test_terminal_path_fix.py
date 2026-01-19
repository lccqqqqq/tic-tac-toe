#!/usr/bin/env python3
"""Test that the terminal path fix works correctly."""

import sys
import os
sys.path.insert(0, 'tests')

from test_loss_game_solvers import GAME_CONFIGS, _get_best_terminal_path, _reconstruct_terminal_from_tree_path
from environment import LossGame1DEnv
from spaces import BoxActionSpace
from mcts import MCTS
import torch as t

print("="*70)
print("Testing Terminal Path Fix")
print("="*70)

# Setup
game_config = GAME_CONFIGS['LossGame1D']
env = game_config['env_cls'](**game_config['env_kwargs'])
action_space = BoxActionSpace(**game_config['action_space_kwargs'], seed=42)

print(f"\nGame: {env.__class__.__name__}")
print(f"  num_stages: {env.num_stages}")
print(f"  num_players: {env.num_players}")

# Run MCTS with minimal simulations
print(f"\nRunning MCTS with 50 simulations...")
mcts = MCTS(
    env, action_space,
    exploration_weight=1.0,
    widening_factor=2.0,
    num_simulations=50,
    seed=42
)
mcts.run()

# Test the fix
print(f"\n[OLD WAY] Using select_action() - returns immediate best child:")
best_child_old, best_action_old = mcts.select_action()
print(f"  best_child: {best_child_old}")
print(f"  depth: {len(best_child_old)}")

print(f"\n[NEW WAY] Using _get_best_terminal_path() - returns complete path:")
try:
    best_terminal_path = _get_best_terminal_path(mcts, env.num_stages)
    print(f"  best_terminal_path: {best_terminal_path}")
    print(f"  depth: {len(best_terminal_path)}")

    # Reconstruct environment
    terminal_env = _reconstruct_terminal_from_tree_path(env, mcts, best_terminal_path)
    print(f"\n[VERIFICATION]")
    print(f"  terminal_env.terminal: {terminal_env.terminal}")
    print(f"  terminal_env.payoff: {terminal_env.payoff}")
    print(f"  len(terminal_env.state): {len(terminal_env.state)}")

    if terminal_env.terminal and terminal_env.payoff is not None:
        print(f"\n✓ SUCCESS! The fix works correctly.")
        print(f"  - Found complete path to terminal node")
        print(f"  - Terminal state has payoff: {terminal_env.payoff.tolist()}")
    else:
        print(f"\n✗ FAILED! Terminal state not reached properly.")
        sys.exit(1)

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("Test completed successfully!")
print("="*70)
