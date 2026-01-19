"""
Minimal VG-MCTS test on Quantum Simple Board.
Runs MCTS from fresh at each decision node.
"""

import torch as t
from environment import QuantumBoardGameEnv, SimpleBoard
from spaces import ComplexProjectiveActionSpace
from mcts import GradientMCTS

# Configuration
num_simulations = 1000
frequencies = [2]
learning_rate = 1e-2

print("="*70)
print("VG-MCTS on Quantum Simple Board - Sequential MCTS Test")
print("="*70)
print(f"Simulations per move: {num_simulations}")
print(f"Frequencies: {frequencies}")
print(f"Learning Rate: {learning_rate}")
print()

action_space = ComplexProjectiveActionSpace(d_action=5, seed=42)

results = []

for freq in frequencies:
    print(f"Running freq={freq}...")

    # Start from initial state
    current_env = QuantumBoardGameEnv(board_cls=SimpleBoard)
    action_chain = []

    # Play out 3 moves (to terminal state)
    for move_idx in range(3):
        # Run fresh MCTS from current state
        mcts = GradientMCTS(
            env=current_env,
            action_space=action_space,
            gradient_update_frequency=freq,
            learning_rate=learning_rate,
            clip_grad_norm=1.0,
            exploration_weight=1.0,
            widening_factor=2.0,
            num_simulations=num_simulations,
            seed=42 + move_idx,  # Different seed for each move
            show_progress=False
        )

        mcts.run()

        # Select best action from root
        best_child, best_action = mcts.select_action()
        action_chain.append(best_action)

        # Apply action to environment for next iteration
        current_env = current_env.move(best_action)

    # Get final payoff
    final_payoff = current_env.payoff

    print(f"  Final payoff: P0={final_payoff[0].item():+.4f}, P1={final_payoff[1].item():+.4f}")

    results.append({
        'freq': freq,
        'p0': final_payoff[0].item(),
        'p1': final_payoff[1].item(),
        'actions': action_chain,
        'final_env': current_env
    })

print()
print("="*70)
print("Results Summary:")
print(f"{'Frequency':<12} {'Player 0':<12} {'Player 1':<12}")
print("-"*70)
for r in results:
    print(f"{r['freq']:<12} {r['p0']:<+12.4f} {r['p1']:<+12.4f}")
print("="*70)

best = max(results, key=lambda r: r['p0'])
print(f"\nBest for P0: freq={best['freq']}, payoff={best['p0']:+.4f}")

# Display action chains
print()
print("="*70)
print("Action Chains (Sequential MCTS Moves)")
print("="*70)
for r in results:
    print(f"\nFrequency {r['freq']}:")
    print(f"  Action chain (3 quantum moves):")
    for i, action in enumerate(r['actions']):
        probs = (action.abs() ** 2).real
        print(f"    Move {i}: sites {[f'{p.item():.3f}' for p in probs]}")
