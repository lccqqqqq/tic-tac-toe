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

    def _regularize(self, actions: Complex[Tensor, "sample site"]):
        norms = t.linalg.vector_norm(actions, dim=-1, keepdim=True).clamp_min(1e-12)
        first_column = actions[:, 0]
        phases = first_column / first_column.abs().clamp_min(1e-12)
        actions = actions * phases.conj().unsqueeze(-1)
        actions = actions / norms
        return actions

    def sample(self, num_samples: int = 1) -> list[t.Tensor]:
        # The sampling method here returns shape (1, 5) instead of (5,)
        real = t.randn((num_samples, self.d_action), generator=self.rng, dtype=t.float32)
        imag = t.randn((num_samples, self.d_action), generator=self.rng, dtype=t.float32)
        actions = t.complex(real, imag)
        actions = self._regularize(actions)
        if num_samples == 1:
            return actions[0]
        return actions

    def sample_local(self, num_samples: int, action: Tensor, step_size: float):
        v_real = t.randn((num_samples, self.d_action), generator=self.rng, dtype=t.float32)
        v_imag = t.randn((num_samples, self.d_action), generator=self.rng, dtype=t.float32)
        v = t.complex(v_real, v_imag)
        
        projections = einops.einsum(
            v, action.squeeze(),
            "sample action, action -> sample action"
        )
        v_h = v - projections
        v_h = self._regularize(v_h)
        new_actions = action.squeeze() * math.cos(step_size) + v_h * math.sin(step_size)
        return self._regularize(new_actions) # shape (num_samples, 5)
    

