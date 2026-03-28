from abc import ABC, abstractmethod
import torch as t
from torch import Tensor
from jaxtyping import Complex
import einops
import math

class ActionSpace(ABC):

    @abstractmethod
    def sample(self, num_samples: int = 1) -> list[t.Tensor]:
        return None

    @abstractmethod
    def sample_local(self, num_samples: int, action: Tensor, step_size: float) -> Tensor:
        return None
    
class BoxActionSpace(ActionSpace):
    def __init__(self, d_action: int, lows: list[float], highs: list[float], seed: int | None = None):

        assert len(lows) == d_action, "Length of lows must match d_action"
        assert len(highs) == d_action, "Length of highs must match d_action"
        assert all(low <= high for low, high in zip(lows, highs)), "Low must be less than high"

        self.d_action = d_action
        self.lows = t.tensor(lows, dtype=t.float32)
        self.highs = t.tensor(highs, dtype=t.float32)
        self.seed = seed
        self.rng = t.Generator().manual_seed(self.seed) if self.seed is not None else t.Generator()

    def sample(self, num_samples: int = 1) -> list[t.Tensor]:
        if num_samples == 1:
            return t.rand((self.d_action), generator=self.rng) * (self.highs - self.lows) + self.lows
        else:
            return t.rand((num_samples, self.d_action), generator=self.rng) * (self.highs - self.lows) + self.lows

    def sample_local(self, num_samples: int, action: Tensor, step_size: float) -> Tensor:
        base = action.reshape(-1).to(dtype=t.float32)
        assert base.shape == (self.d_action,), "Action dimensionality mismatch"
        base = base.clamp(self.lows, self.highs)

        directions = t.randn((num_samples, self.d_action), generator=self.rng, dtype=t.float32)
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        proposals = base.unsqueeze(0) + step_size * directions
        proposals = proposals.clamp(self.lows, self.highs)

        return proposals

class ComplexProjectiveActionSpace(ActionSpace):
    def __init__(self, d_action: int, seed: int | None = None):
        self.d_action = d_action # This is NOT the n in CP^n, instead this aligns with the number of sites on the board
        self.seed = seed
        self.rng = t.Generator().manual_seed(self.seed) if self.seed is not None else t.Generator()

    def _regularize(self, actions: Complex[Tensor, "sample_site"]):
        norms = t.linalg.vector_norm(actions, dim=-1, keepdim=True).clamp_min(1e-12)
        first_column = actions[:, 0]
        phases = first_column / first_column.abs().clamp_min(1e-12)
        actions = actions * phases.conj().unsqueeze(-1)
        actions = actions / norms
        return actions

    def sample(self, num_samples: int = 1) -> Tensor:
        two_pi = 2.0 * math.pi

        if num_samples == 1:
            indices = t.randperm(self.d_action, generator=self.rng)[:2]

            u = t.rand((), generator=self.rng, dtype=t.float32)
            theta = t.arcsin(t.sqrt(u))
            phi = two_pi * t.rand((), generator=self.rng, dtype=t.float32)

            a0 = t.cos(theta)
            a1 = t.sin(theta) * t.complex(t.cos(phi), t.sin(phi))

            action = t.zeros(self.d_action, dtype=t.complex64)
            action[indices[0]] = a0.to(t.complex64)
            action[indices[1]] = a1.to(t.complex64)
            return action

        actions = t.zeros((num_samples, self.d_action), dtype=t.complex64)

        for n in range(num_samples):
            indices = t.randperm(self.d_action, generator=self.rng)[:2]

            u = t.rand((), generator=self.rng, dtype=t.float32)
            theta = t.arcsin(t.sqrt(u))
            phi = two_pi * t.rand((), generator=self.rng, dtype=t.float32)

            a0 = t.cos(theta)
            a1 = t.sin(theta) * t.complex(t.cos(phi), t.sin(phi))

            actions[n, indices[0]] = a0.to(t.complex64)
            actions[n, indices[1]] = a1.to(t.complex64)

        return actions

    def sample_local(self, num_samples: int, action: Tensor, step_size: float) -> Tensor:
        base = action.reshape(-1).to(dtype=t.complex64)
        assert base.shape == (self.d_action,), "Action dimensionality mismatch"

        support = (base.abs() > 1e-8).nonzero(as_tuple=False).flatten()
        assert support.numel() == 2, "Expected exactly two active amplitudes"

        base_amps = base[support]

        global_phase = base_amps[0] / base_amps[0].abs().clamp_min(1e-12)
        canonical = base_amps * global_phase.conj()

        a0 = canonical[0].real.clamp(-1.0, 1.0)
        theta0 = t.arccos(a0)

        if canonical[1].abs() < 1e-12:
            phi0 = t.tensor(0.0, dtype=t.float32)
        else:
            phi0 = t.angle(canonical[1]).to(t.float32)

        proposals = t.zeros((num_samples, self.d_action), dtype=t.complex64)

        for n in range(num_samples):
            dtheta = step_size * t.randn((), generator=self.rng, dtype=t.float32)
            dphi = step_size * t.randn((), generator=self.rng, dtype=t.float32)

            theta = (theta0 + dtheta).clamp(0.0, math.pi / 2.0)
            phi = t.remainder(phi0 + dphi, 2.0 * math.pi)

            new_a0 = t.cos(theta)
            new_a1 = t.sin(theta) * t.complex(t.cos(phi), t.sin(phi))

            proposals[n, support[0]] = new_a0.to(t.complex64)
            proposals[n, support[1]] = new_a1.to(t.complex64)

        return proposals
    

