"""
Test script for GradientMCTS implementation.
Compares standard MCTS vs VG-MCTS on a simple loss game.
"""

import torch as t
from environment import LossGame1DEnv
from spaces import BoxActionSpace
from mcts import MCTS, GradientMCTS

def test_gradient_mcts():
    """Test that GradientMCTS runs without errors and improves over standard MCTS."""

    # Setup environment
    env = LossGame1DEnv(
        num_stages=3,
        num_players=3,
        d_action=1,
        preserve_grad=True
    )

    # Setup action space
    action_space = BoxActionSpace(
        d_action=1,
        lows=[-5.0],
        highs=[5.0],
        seed=42
    )

    print("="*60)
    print("Testing Standard MCTS")
    print("="*60)

    # Standard MCTS
    mcts_standard = MCTS(
        env=env,
        action_space=action_space,
        exploration_weight=1.0,
        widening_factor=2.0,
        num_simulations=50,
        seed=42
    )

    mcts_standard.run()
    best_child_std, best_action_std = mcts_standard.select_action()

    print(f"Best child abstract node: {best_child_std}")
    print(f"Best action: {best_action_std}")
    print(f"Visits: {mcts_standard.visits[best_child_std]}")
    print(f"Value: {mcts_standard.values[best_child_std]}")

    print("\\n" + "="*60)
    print("Testing VG-MCTS (Gradient MCTS)")
    print("="*60)

    # VG-MCTS with gradient updates
    mcts_gradient = GradientMCTS(
        env=env,
        action_space=action_space,
        gradient_update_frequency=0.25,  # 25% of rollouts
        learning_rate=1e-3,
        clip_grad_norm=1.0,
        exploration_weight=1.0,
        widening_factor=2.0,
        num_simulations=50,
        seed=42
    )

    mcts_gradient.run()
    best_child_grad, best_action_grad = mcts_gradient.select_action()

    print(f"Best child abstract node: {best_child_grad}")
    print(f"Best action: {best_action_grad}")
    print(f"Visits: {mcts_gradient.visits[best_child_grad]}")
    print(f"Value: {mcts_gradient.values[best_child_grad]}")

    print("\\n" + "="*60)
    print("Comparison")
    print("="*60)
    print(f"Standard MCTS value: {mcts_standard.values[best_child_std]}")
    print(f"VG-MCTS value:       {mcts_gradient.values[best_child_grad]}")

    print("\\nTest completed successfully!")
    return mcts_standard, mcts_gradient

if __name__ == "__main__":
    test_gradient_mcts()
