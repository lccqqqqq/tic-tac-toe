"""
Test MCTS with Local Stochastic Minimax Update on Quantum Simple Board.
"""

import torch as t
from environment import QuantumBoardGameEnv, SimpleBoard
from spaces import ComplexProjectiveActionSpace
from mcts import MCTSwithLocalUpdate

# Configuration
num_simulations = 100
frequencies = [1.0]
learning_rate = 1e-3
max_branching_factor = 5

print("="*70)
print("MCTS with Local Stochastic Minimax Update Test")
print("="*70)
print(f"Simulations: {num_simulations}")
print(f"Frequencies: {frequencies}")
print(f"Learning Rate: {learning_rate}")
print(f"Max Branching Factor: {max_branching_factor}")
print()

action_space = ComplexProjectiveActionSpace(d_action=5, seed=42)

results = []

for freq in frequencies:
    print(f"Testing freq={freq}... ", end='', flush=True)

    env = QuantumBoardGameEnv(board_cls=SimpleBoard)

    mcts = MCTSwithLocalUpdate(
        env=env,
        action_space=action_space,
        update_frequency=freq,
        learning_rate=learning_rate,
        max_branching_factor=max_branching_factor,
        exploration_weight=1.0,
        widening_factor=2.0,
        num_simulations=num_simulations,
        seed=42
    )

    mcts.run()
    best_child, _ = mcts.select_action()

    visits = mcts.visits[best_child]
    avg = mcts.values[best_child] / visits

    print(f"P0={avg[0].item():+.4f}, P1={avg[1].item():+.4f}, visits={visits}")

    results.append({
        'freq': freq,
        'p0': avg[0].item(),
        'p1': avg[1].item(),
        'visits': visits
    })

print()
print("="*70)
print("Results Summary:")
print(f"{'Frequency':<12} {'Player 0':<12} {'Player 1':<12} {'Visits':<10}")
print("-"*70)
for r in results:
    print(f"{r['freq']:<12} {r['p0']:<+12.4f} {r['p1']:<+12.4f} {r['visits']:<10}")
print("="*70)

best = max(results, key=lambda r: r['p0'])
print(f"\nBest for P0: freq={best['freq']}, payoff={best['p0']:+.4f}")
print("\nTest completed successfully!")
