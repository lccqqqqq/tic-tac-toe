"""
Test VG-MCTS with multiple gradient updates per rollout (freq >= 2).
"""

import torch as t
from environment import QuantumBoardGameEnv, SimpleBoard
from spaces import ComplexProjectiveActionSpace
from mcts import GradientMCTS

# Configuration
num_simulations = 100
frequencies = [0.0, 1.0, 2.0, 3.0]  # Test multiple updates
learning_rate = 1e-3

print("="*70)
print("VG-MCTS Multiple Gradient Updates Test")
print("="*70)
print(f"Simulations: {num_simulations}")
print(f"Frequencies: {frequencies}")
print(f"Learning Rate: {learning_rate}")
print()

action_space = ComplexProjectiveActionSpace(d_action=5, seed=42)

for freq in frequencies:
    print(f"Testing freq={freq}... ", end='', flush=True)

    env = QuantumBoardGameEnv(board_cls=SimpleBoard)

    mcts = GradientMCTS(
        env=env,
        action_space=action_space,
        gradient_update_frequency=freq,
        learning_rate=learning_rate,
        clip_grad_norm=1.0,
        exploration_weight=1.0,
        widening_factor=2.0,
        num_simulations=num_simulations,
        seed=42
    )

    mcts.run()
    best_child, _ = mcts.select_action()

    visits = mcts.visits[best_child]
    avg = mcts.values[best_child] / visits

    print(f"P0={avg[0].item():+.4f}, visits={visits}")

print("\nTest completed successfully!")
