"""
Weighted sampling strategies for training sample selection.

Implements adaptive sampling based on historical performance feedback,
using lazy update mechanism for efficient weight computation.
"""

import os
import json
import numpy as np
import yaml
from typing import Optional, List, Dict, Any


class WeightedSampler:
    """Base class for weighted sampling strategies."""

    def __init__(self, n_samples: int, seed: int = 42, **kwargs):
        self.n_samples = n_samples
        self._used_in_epoch: set = set()
        self._rng = np.random.RandomState(seed)

    def sample_batch(self, batch_size: int, global_step: int) -> List[int]:
        raise NotImplementedError

    def update(self, indices: List[int], rewards: List[float], global_step: int):
        raise NotImplementedError

    def reset_epoch(self):
        self._used_in_epoch.clear()

    @property
    def remaining(self) -> int:
        return self.n_samples - len(self._used_in_epoch)

    def save_state(self, path: str):
        pass

    def load_state(self, path: str):
        pass

    def get_diagnostics(self) -> Dict[str, Any]:
        return {"algorithm": "base", "n_samples": self.n_samples, "remaining": self.remaining}


class UniformSampler(WeightedSampler):
    """Default uniform random sampling (no weighting)."""

    def sample_batch(self, batch_size: int, global_step: int) -> List[int]:
        available = np.array([i for i in range(self.n_samples) if i not in self._used_in_epoch])
        if len(available) == 0:
            self.reset_epoch()
            available = np.arange(self.n_samples)

        actual_size = min(batch_size, len(available))
        selected = self._rng.choice(available, size=actual_size, replace=False).tolist()
        self._used_in_epoch.update(selected)
        return selected

    def update(self, indices: List[int], rewards: List[float], global_step: int):
        pass

    def save_state(self, path: str):
        state = {
            "algorithm": "uniform",
            "n_samples": self.n_samples,
            "used_in_epoch": sorted(self._used_in_epoch),
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)

        if state.get("n_samples") != self.n_samples:
            print(
                f"  [WeightedSampler] Warning: n_samples mismatch "
                f"(saved={state['n_samples']}, current={self.n_samples}), skipping state load"
            )
            return

        self._used_in_epoch = set(state.get("used_in_epoch", []))

    def get_diagnostics(self) -> Dict[str, Any]:
        return {"algorithm": "uniform", "n_samples": self.n_samples, "remaining": self.remaining}


class SqrtBiasSampler(WeightedSampler):
    """
    Sqrt-bias weighted sampling with lazy update mechanism.

    At each step t, for every sample i the effective weight is:

        delta_T = t - T_last[i]
        I_alpha_now = I_alpha[i] * rho^delta_T
        I_beta_now  = I_beta[i]  * rho^delta_T
        V = (1 + I_alpha_now) / (2 + I_alpha_now + I_beta_now)
        S = sqrt(V * (1 - V)) + gamma * (1 - V) + epsilon

    Only the sampled sample's state is written back after feedback;
    all other samples stay untouched (lazy / deferred decay).
    """

    def __init__(
        self,
        n_samples: int,
        rho: float = 0.9,
        gamma: float = 0.2,
        epsilon: float = 0.1,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(n_samples, seed=seed)
        self.rho = rho
        self.gamma = gamma
        self.epsilon = epsilon

        self.I_alpha = np.zeros(n_samples, dtype=np.float64)
        self.I_beta = np.zeros(n_samples, dtype=np.float64)
        self.T_last = np.zeros(n_samples, dtype=np.int64)

    # ------------------------------------------------------------------
    # Core vectorised weight computation
    # ------------------------------------------------------------------

    def _compute_weights(self, global_step: int) -> np.ndarray:
        """Vectorised weight computation with lazy decay."""
        delta_T = np.maximum(global_step - self.T_last, 0)
        decay = np.power(self.rho, delta_T)

        cur_alpha = self.I_alpha * decay
        cur_beta = self.I_beta * decay

        V = (1.0 + cur_alpha) / (2.0 + cur_alpha + cur_beta)
        S = np.sqrt(V * (1.0 - V)) + self.gamma * (1.0 - V) + self.epsilon
        return S

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_batch(self, batch_size: int, global_step: int) -> List[int]:
        actual_size = min(batch_size, self.n_samples)

        weights = self._compute_weights(global_step)
        w_sum = weights.sum()
        probs = weights / w_sum if w_sum > 0 else np.ones(self.n_samples) / self.n_samples

        selected = self._rng.choice(
            self.n_samples, size=actual_size, replace=False, p=probs
        ).tolist()
        return selected

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def update(self, indices: List[int], rewards: List[float], global_step: int):
        """Update state only for the sampled indices (lazy write-back)."""
        for idx, r in zip(indices, rewards):
            delta_T = max(global_step - int(self.T_last[idx]), 0)
            decay = self.rho ** delta_T

            cur_alpha = self.I_alpha[idx] * decay
            cur_beta = self.I_beta[idx] * decay

            self.I_alpha[idx] = cur_alpha + r
            self.I_beta[idx] = cur_beta + (1.0 - r)
            self.T_last[idx] = global_step

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: str):
        state = {
            "algorithm": "sqrt_bias",
            "n_samples": self.n_samples,
            "rho": self.rho,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "I_alpha": self.I_alpha.tolist(),
            "I_beta": self.I_beta.tolist(),
            "T_last": self.T_last.tolist(),
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)

        if state.get("n_samples") != self.n_samples:
            print(
                f"  [WeightedSampler] Warning: n_samples mismatch "
                f"(saved={state['n_samples']}, current={self.n_samples}), skipping state load"
            )
            return

        self.I_alpha = np.array(state["I_alpha"], dtype=np.float64)
        self.I_beta = np.array(state["I_beta"], dtype=np.float64)
        self.T_last = np.array(state["T_last"], dtype=np.int64)

    def get_diagnostics(self) -> Dict[str, Any]:
        weights = self._compute_weights(int(self.T_last.max()) + 1 if self.T_last.max() > 0 else 0)
        return {
            "algorithm": "sqrt_bias",
            "n_samples": self.n_samples,
            "remaining": self.remaining,
            "rho": self.rho,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "weight_mean": float(weights.mean()),
            "weight_std": float(weights.std()),
            "weight_min": float(weights.min()),
            "weight_max": float(weights.max()),
        }


# ======================================================================
# Factory
# ======================================================================

_ALGORITHM_REGISTRY = {
    "uniform": UniformSampler,
    "sqrt_bias": SqrtBiasSampler,
}


def create_sampler(
    n_samples: int,
    config_path: Optional[str] = None,
    state_path: Optional[str] = None,
    seed: int = 42,
) -> WeightedSampler:
    """
    Create a WeightedSampler from a YAML config file.

    If *config_path* is ``None``, returns a ``UniformSampler`` (equivalent to
    no weighting – the original sequential behaviour).

    Config YAML example::

        algorithm: sqrt_bias
        gamma: 0.2
        rho: 0.9
        epsilon: 0.1
    """
    if config_path is None:
        return UniformSampler(n_samples, seed=seed)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    algorithm = config.pop("algorithm", "uniform")
    cls = _ALGORITHM_REGISTRY.get(algorithm)
    if cls is None:
        print(
            f"  [WeightedSampler] Unknown algorithm '{algorithm}', "
            f"falling back to uniform"
        )
        cls = UniformSampler

    sampler = cls(n_samples=n_samples, seed=seed, **config)

    if state_path and os.path.exists(state_path):
        try:
            sampler.load_state(state_path)
            print(f"  [WeightedSampler] Resumed state from {state_path}")
        except Exception as e:
            print(f"  [WeightedSampler] Failed to load state: {e}")

    print(f"  [WeightedSampler] Created '{algorithm}' sampler for {n_samples} samples")
    return sampler


def compute_sample_reward(sample_info: dict) -> float:
    """
    Extract average accuracy from rollout results as the reward signal r_t.

    Returns 0.5 (uninformative prior) when no results are available.
    """
    results = sample_info.get("sample_rollout_results", [])
    if not results:
        return 0.5
    accuracies = [
        r.get("accuracy_score", 0.0)
        for r in results
        if isinstance(r, dict)
    ]
    return sum(accuracies) / len(accuracies) if accuracies else 0.5
