"""
Codex variant of the Quantum Simple Board VG-MCTS test.

We run GradientMCTS with rerooting across all stages of the quantum 1D board
game and confirm the discovered strategy concentrates the opening move on the
centre site and yields a positive payoff for Player 0.
"""

import torch as t

from environment import QuantumBoardGameEnv, SimpleBoard
from spaces import ComplexProjectiveActionSpace
from mcts import GradientMCTS


def _format_probs(action: t.Tensor) -> list[float]:
    probs = (action.abs() ** 2).real
    return [round(float(p), 3) for p in probs.tolist()]


def test_vg_mcts_quantum_codex():
    env = QuantumBoardGameEnv(board_cls=SimpleBoard)
    action_space = ComplexProjectiveActionSpace(
        d_action=SimpleBoard.num_sites,
        seed=5,
    )

    mcts = GradientMCTS(
        env=env,
        action_space=action_space,
        gradient_update_frequency=1.0,
        learning_rate=0.01,
        clip_grad_norm=5.0,
        exploration_weight=1.0,
        widening_factor=2.0,
        num_simulations=700,
        seed=11,
    )

    plan_actions: list[t.Tensor] = []
    current_env = env

    for stage in range(SimpleBoard.num_stages):
        mcts.run()
        best_child, best_action = mcts.select_action()

        probs = (best_action.abs() ** 2).real
        assert t.isclose(probs.sum(), t.tensor(1.0), atol=1e-4), "Action probabilities must sum to 1."
        plan_actions.append(best_action.detach())

        print(f"Stage {stage} action probabilities:", _format_probs(best_action))

        current_env = current_env.move(best_action)

        if stage < SimpleBoard.num_stages - 1:
            mcts.reroot(best_child, current_env)

    assert current_env.terminal

    first_move_probs = (plan_actions[0].abs() ** 2).real
    centre_prob = float(first_move_probs[2])
    print("Centre-site probability:", round(centre_prob, 3))
    assert t.argmax(first_move_probs).item() == 2
    assert centre_prob > 0.45

    payoff = current_env.payoff
    print("Terminal payoff:", [round(float(x), 3) for x in payoff.tolist()])
    assert payoff[0].item() > 0.15
    assert t.allclose(payoff[0], -payoff[1], atol=1e-6)


if __name__ == "__main__":
    test_vg_mcts_quantum_codex()
