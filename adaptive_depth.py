"""Adaptive depth utilities: progress monitor and difficulty estimator."""

import torch


class ProgressMonitor:
    """Tracks convergence dynamics of hidden states during recurrence."""

    def __init__(self, exit_threshold=5e-4, extend_threshold=0.0, pool="mean"):
        self.exit_threshold = exit_threshold    # δ below this → converged
        self.extend_threshold = extend_threshold # Δ above this → struggling
        self.pool = pool
        self.reset()

    def reset(self):
        self.deltas = []  # velocity history
        self.prev_state = None

    def step(self, state: torch.Tensor) -> dict:
        """Call after each recurrence step with the current hidden state.

        Args:
            state: shape (batch, seq_len, hidden_dim)

        Returns:
            dict with keys:
                - delta: current velocity (float)
                - acceleration: current acceleration (float or None)
                - signal: one of "continue", "exit", "extend"
        """
        if self.prev_state is not None:
            diff = (state - self.prev_state).float()
            if self.pool == "mean":
                delta = diff.norm(dim=-1).mean().item()
            else:
                delta = diff.norm(dim=-1).max().item()
            self.deltas.append(delta)
        else:
            delta = float("inf")
            self.deltas.append(delta)

        self.prev_state = state.detach().clone()

        # Need at least 2 deltas to compute acceleration
        if len(self.deltas) >= 2:
            acceleration = self.deltas[-1] - self.deltas[-2]
        else:
            acceleration = None

        # Decision logic
        if delta < self.exit_threshold:
            signal = "exit"
        elif acceleration is not None and acceleration > self.extend_threshold:
            signal = "extend"
        else:
            signal = "continue"

        return {"delta": delta, "acceleration": acceleration, "signal": signal}


class DifficultyEstimator(torch.nn.Module):
    """Predicts required recurrence depth from an early hidden state."""

    def __init__(self, hidden_dim, max_recurrence=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid(),  # output in [0, 1]
        )
        self.max_recurrence = max_recurrence

    def forward(self, pooled_state):
        """Returns predicted recurrence budget as a float."""
        return self.net(pooled_state).squeeze(-1) * self.max_recurrence
