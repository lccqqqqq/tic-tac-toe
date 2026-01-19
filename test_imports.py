#!/usr/bin/env python3
"""Test if imports work correctly."""

import sys
import os

print("Testing imports from root directory...")

try:
    from environment import LossGame1DEnv, LossGame3DEnv
    print("✓ environment imports work")
except Exception as e:
    print(f"✗ environment import failed: {e}")
    sys.exit(1)

try:
    from spaces import BoxActionSpace
    print("✓ spaces imports work")
except Exception as e:
    print(f"✗ spaces import failed: {e}")
    sys.exit(1)

try:
    from mcts import MCTS, GradientMCTS, MCTSwithLocalUpdate
    print("✓ mcts imports work")
except Exception as e:
    print(f"✗ mcts import failed: {e}")
    sys.exit(1)

try:
    from solver import LocalStochasticMinimaxUpdate
    print("✓ solver imports work")
except Exception as e:
    print(f"✗ solver import failed: {e}")
    sys.exit(1)

try:
    from evaluator import Evaluator
    print("✓ evaluator imports work")
except Exception as e:
    print(f"✗ evaluator import failed: {e}")
    sys.exit(1)

print("\nAll root imports successful!")

print("\nTesting imports from tests/ directory...")
sys.path.insert(0, 'tests')

try:
    from test_loss_game_solvers import GAME_CONFIGS, SOLVER_CONFIGS
    print("✓ test_loss_game_solvers imports work")
    print(f"  - Found {len(GAME_CONFIGS)} game configs")
    print(f"  - Found {len(SOLVER_CONFIGS)} solver configs")
except Exception as e:
    print(f"✗ test_loss_game_solvers import failed: {e}")
    sys.exit(1)

print("\n✓✓✓ All imports successful! ✓✓✓")
