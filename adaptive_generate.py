"""Adaptive-depth inference for retrofitted recurrent models.

Replaces fixed-depth forward passes with a dynamic system that:
  1. Exits early when the hidden state has converged (ProgressMonitor).
  2. Optionally estimates the required depth from an early probe state (DifficultyEstimator).
  3. Extends the budget when the model is still making rapid progress.

Usage example:
    from adaptive_generate import adaptive_forward, load_estimator
    from adaptive_depth import ProgressMonitor

    monitor = ProgressMonitor(exit_threshold=5e-4)
    estimator = load_estimator("checkpoints/difficulty_estimator.pt", device)

    logits, steps_used = adaptive_forward(
        model, input_ids, monitor, estimator,
        probe_step=4, min_r=4, max_r=64, extend_margin=8,
    )
"""

import torch
from typing import Optional
from adaptive_depth import ProgressMonitor, DifficultyEstimator


def load_estimator(path: str, device: str = "cuda") -> DifficultyEstimator:
    """Load a saved DifficultyEstimator checkpoint."""
    ckpt = torch.load(path, map_location=device)
    estimator = DifficultyEstimator(
        hidden_dim=ckpt["hidden_dim"],
        max_recurrence=ckpt["max_recurrence"],
    ).to(device)
    estimator.load_state_dict(ckpt["state_dict"])
    estimator.eval()
    return estimator


@torch.no_grad()
def adaptive_forward(
    model,
    input_ids: torch.Tensor,
    monitor: ProgressMonitor,
    estimator: Optional[DifficultyEstimator] = None,
    probe_step: int = 4,
    min_r: int = 4,
    max_r: int = 64,
    extend_margin: int = 8,
):
    """Run the model with adaptive recurrence depth.

    Args:
        model: RavenForCausalLM instance.
        input_ids: (batch, seq_len) token ids.
        monitor: ProgressMonitor instance (will be reset on each call).
        estimator: Optional DifficultyEstimator for budget prediction.
        probe_step: Recurrence step at which to query the estimator.
        min_r: Never exit before this many steps.
        max_r: Hard ceiling on recurrences.
        extend_margin: Extra steps to grant when extending budget.

    Returns:
        (logits, steps_taken): logits of shape (batch, seq, vocab), int steps used.
    """
    monitor.reset()

    device = input_ids.device
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)

    # --- Embed tokens ---
    input_embeds = model.transformer.wte(input_ids)
    if model.emb_scale != 1:
        input_embeds = input_embeds * model.emb_scale

    freqs_cis = model.rotary_emb(input_embeds, position_ids)

    # --- Prelude ---
    block_idx = torch.tensor(-1, device=torch.device("cpu"), dtype=torch.long)
    x = input_embeds
    for block in model.transformer.prelude:
        block_idx += 1
        x = block(x, freqs_cis, block_idx, None, None)

    input_embeds = x  # embedded sequence after prelude

    # --- Initialize recurrent state ---
    state = model.initialize_state(input_embeds)

    budget: Optional[int] = None
    steps_taken = 0

    for i in range(1, max_r + 1):
        # --- One recurrence step ---
        state, block_idx = model.core_block_forward(
            state, input_embeds, freqs_cis, None, None,
            torch.tensor(model.config.n_layers_in_prelude - 1, device=torch.device("cpu"), dtype=torch.long),
            current_step=i - 1,
        )
        steps_taken = i

        # --- At probe step, get initial budget estimate ---
        if i == probe_step and estimator is not None:
            pooled = state.detach().mean(dim=1)  # (batch, hidden)
            raw_budget = estimator(pooled).mean().item()
            budget = int(max(min_r, min(raw_budget, max_r)))

        # --- From step 2 onward, check progress monitor ---
        if i >= 2:
            result = monitor.step(state)

            if i >= min_r and result["signal"] == "exit":
                break

            if result["signal"] == "extend" and budget is not None:
                budget = min(budget + extend_margin, max_r)

            if budget is not None and i >= budget and result["signal"] == "continue":
                # At or past budget and still moving but not accelerating
                if i >= budget + extend_margin:
                    break
        else:
            monitor.step(state)

    # --- Coda ---
    block_idx = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
    for block in model.transformer.coda:
        block_idx -= 1
        state = block(state, freqs_cis, block_idx, None, None)
    state = model.transformer.ln_f(state)

    logits = model.lm_head(state)
    return logits, steps_taken
