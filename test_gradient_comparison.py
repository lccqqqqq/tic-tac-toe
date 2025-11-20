"""
Comprehensive test comparing Standard MCTS vs VG-MCTS.
Tests different gradient update frequencies.
"""

import torch as t
from environment import LossGame1DEnv, LossGame3DEnv
from spaces import BoxActionSpace
from mcts import MCTS, GradientMCTS

def compare_mcts_variants():
    """Compare standard MCTS with different VG-MCTS configurations."""

    print("="*70)
    print("MCTS vs VG-MCTS Comparison on Loss Game")
    print("="*70)

    # Setup environment
    env = LossGame3DEnv(
        num_stages=3,
        num_players=3,
        d_action=3,
        preserve_grad=True
    )

    # Setup action space
    action_space = BoxActionSpace(
        d_action=3,
        lows=[-5.0, -5.0, -5.0],
        highs=[5.0, 5.0, 5.0],
        seed=42
    )

    num_simulations = 1000
    results = []

    # Test 1: Standard MCTS (no gradients)
    print(f"\\n[1/4] Running Standard MCTS ({num_simulations} simulations)...")
    mcts_std = MCTS(
        env=env,
        action_space=action_space,
        exploration_weight=1.0,
        widening_factor=2.0,
        num_simulations=num_simulations,
        seed=42
    )
    mcts_std.run()
    best_child_std, best_action_std = mcts_std.select_action()
    results.append(("Standard MCTS", mcts_std, best_child_std))
    print(f"   Completed. Best value: {mcts_std.values[best_child_std]}")

    # Test 2: VG-MCTS with low frequency (0.1)
    print(f"\\n[2/4] Running VG-MCTS (freq=0.1, {num_simulations} simulations)...")
    mcts_grad_low = GradientMCTS(
        env=env,
        action_space=action_space,
        gradient_update_frequency=0.1,
        learning_rate=1e-3,
        clip_grad_norm=1.0,
        exploration_weight=1.0,
        widening_factor=2.0,
        num_simulations=num_simulations,
        seed=42
    )
    mcts_grad_low.run()
    best_child_grad_low, _ = mcts_grad_low.select_action()
    results.append(("VG-MCTS (freq=0.1)", mcts_grad_low, best_child_grad_low))
    print(f"   Completed. Best value: {mcts_grad_low.values[best_child_grad_low]}")

    # Test 3: VG-MCTS with medium frequency (0.25)
    print(f"\\n[3/4] Running VG-MCTS (freq=0.25, {num_simulations} simulations)...")
    mcts_grad_med = GradientMCTS(
        env=env,
        action_space=action_space,
        gradient_update_frequency=0.25,
        learning_rate=1e-3,
        clip_grad_norm=1.0,
        exploration_weight=1.0,
        widening_factor=2.0,
        num_simulations=num_simulations,
        seed=42
    )
    mcts_grad_med.run()
    best_child_grad_med, _ = mcts_grad_med.select_action()
    results.append(("VG-MCTS (freq=0.25)", mcts_grad_med, best_child_grad_med))
    print(f"   Completed. Best value: {mcts_grad_med.values[best_child_grad_med]}")

    # Test 4: VG-MCTS with high frequency (1.0 - every rollout)
    print(f"\\n[4/4] Running VG-MCTS (freq=1.0, {num_simulations} simulations)...")
    mcts_grad_high = GradientMCTS(
        env=env,
        action_space=action_space,
        gradient_update_frequency=1.0,
        learning_rate=1e-3,
        clip_grad_norm=1.0,
        exploration_weight=1.0,
        widening_factor=2.0,
        num_simulations=num_simulations,
        seed=42
    )
    mcts_grad_high.run()
    best_child_grad_high, _ = mcts_grad_high.select_action()
    results.append(("VG-MCTS (freq=1.0)", mcts_grad_high, best_child_grad_high))
    print(f"   Completed. Best value: {mcts_grad_high.values[best_child_grad_high]}")

    # Print summary
    print("\\n" + "="*70)
    print("RESULTS SUMMARY (Average Payoffs)")
    print("="*70)
    print(f"{'Method':<25} {'Visits':<10} {'Player 0':<15} {'Player 1':<15} {'Player 2':<15}")
    print("-"*70)

    for name, mcts, best_child in results:
        values = mcts.values[best_child]
        visits = mcts.visits[best_child]
        # Calculate average payoffs by dividing cumulative values by visits
        avg_values = values / visits if visits > 0 else values
        print(f"{name:<25} {visits:<10} {avg_values[0].item():<15.4f} {avg_values[1].item():<15.4f} {avg_values[2].item():<15.4f}")

    print("="*70)
    print("\\nNotes:")
    print("- Player 0 (maximizer) should have higher values with gradient updates")
    print("- Gradient updates improve actions along explored paths")
    print("- Higher frequency = more gradient updates per simulation")

    return results

if __name__ == "__main__":
    compare_mcts_variants()
