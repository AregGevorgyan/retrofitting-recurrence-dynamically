# Adaptive depth for retrofitted recurrent LLMs

## Implementation plan for `mcleish7/retrofitting-recurrence`

---

## The problem

The retrofitting-recurrence codebase trains depth-recurrent models by sampling a fixed recurrence count `r` per batch from a Poisson-Lognormal distribution (via `num_steps_sampler` in `train.py`). At inference time, you pick a fixed `r` (e.g. 32) for all inputs. There is no mechanism to vary depth per-token or per-sequence based on difficulty, and the paper identifies this as an open problem.

## Goal

Enable the model to dynamically allocate more recurrences to harder inputs and fewer to easy ones, both increasing efficiency and improving accuracy on problems that turn out harder than expected mid-inference.

---

## Architecture of the existing codebase

The key files and their roles:

| File | Role |
|------|------|
| `train.py` | Training loop. Samples `(n, k)` via `num_steps_sampler()` where `n` = steps without grad, `k` = steps with grad. Calls `model(input_ids, labels=labels, num_steps=num_steps)`. |
| `raven_modeling_minimal.py` | The model forward pass (derived from Huginn-0125). Takes `num_steps` as input, runs the prelude, loops the recurrent block `n+k` times, runs the coda. Returns `loss`, `log_ppl`, and `stats` including step counts. |
| `convert_pretrained_model/` | Scripts to convert Llama/TinyLlama/OLMo into the prelude-recurrent-coda architecture. |
| `multi_recurrence_eval.py` | Evaluates at multiple fixed recurrence depths. |

The forward pass structure (from the Huginn architecture):

```
e = Prelude(x)                     # embed tokens
s₀ ~ N(0, σ²)                      # random init state
for i in 1..r:
    s_i = RecurrentBlock(e, s_{i-1}) # concat + adapter + transformer layers
p = Coda(s_r)                       # decode to vocab
```

The recurrent block uses a linear adapter `A: R^{2h} -> R^h` that concatenates `s_{i-1}` and `e`, then passes through the core transformer layers, followed by an RMSNorm.

---

## Phase 1: Instrument the forward pass to expose hidden states

**What to change:** Modify `raven_modeling_minimal.py` to optionally return intermediate hidden states `{s_1, s_2, ..., s_r}` alongside the final output.

**Where in the code:** The forward pass already tracks `num_steps_no_grad` and `num_steps_with_grad` in the `stats` dict. Add a new `output_details` flag `return_trajectory: bool = False`. When enabled, collect `s_i.detach()` at each recurrence step into a list.

```python
# Inside the recurrence loop in the forward method:
trajectory = []
for i in range(total_steps):
    state = self.core_block(embedded, state)
    state = self.core_norm(state)
    if output_details.get("return_trajectory", False):
        trajectory.append(state.detach().clone())
```

Return `trajectory` in the output dict under key `"trajectory"`.

**Why this is needed:** Both the progress monitor and the difficulty estimator need access to intermediate states. This change is backward-compatible — existing training code doesn't set this flag.

---

## Phase 2: Build the progress monitor (no training required)

**What it does:** At each recurrence step `i ≥ 2`, compute:

- δ_i = ‖s_i − s_{i-1}‖₂  (pooled across sequence dim, mean or max over tokens)
- Δ_i = δ_i − δ_{i-1}  (acceleration)

**Implementation:** Create a new file `adaptive_depth.py`:

```python
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
```

**Key design choices:**
- `pool="mean"` averages across tokens; `pool="max"` is more conservative (one hard token keeps the model running). Start with mean, but max may be better for generation where a single token matters.
- `exit_threshold=5e-4` matches the threshold Huginn uses for KL-based exits. This will need tuning for the retrofitted models.
- `extend_threshold=0.0` means any positive acceleration triggers extension. This is conservative; in practice you'd want a small positive margin.

---

## Phase 3: Build the difficulty estimator (requires data collection + training)

### Step 3a: Collect training data

Run the retrofitted model on a diverse evaluation set (e.g., a mix of GSM8K, ARC, HellaSwag, general text) at a high fixed recurrence (e.g., r=64). At each step, record:

- The hidden state `s_k` at an early step (k=4 recommended, since the curriculum starts at low recurrence)
- The actual step `r*` at which δ fell below the exit threshold for the first time

Create a script `collect_difficulty_data.py`:

```python
def collect_convergence_data(model, dataloader, device, probe_step=4, max_r=64,
                              exit_threshold=5e-4):
    """Run model at max_r recurrences, record (s_probe, r*) pairs."""
    records = []
    monitor = ProgressMonitor(exit_threshold=exit_threshold)

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        # Run forward with trajectory collection
        outputs = model(input_ids, num_steps=torch.tensor([0, max_r]),
                       output_details={"return_trajectory": True})

        trajectory = outputs["trajectory"]  # list of (batch, seq, hidden)

        # Find convergence step per batch element
        for b in range(input_ids.shape[0]):
            monitor.reset()
            convergence_step = max_r  # default if never converges
            for i, state in enumerate(trajectory):
                result = monitor.step(state[b:b+1])
                if result["signal"] == "exit" and i > probe_step:
                    convergence_step = i
                    break

            # Record the probe state and target
            probe_state = trajectory[probe_step][b].mean(dim=0)  # pool over seq
            records.append({
                "probe_state": probe_state.cpu(),
                "convergence_step": convergence_step,
            })

    return records
```

### Step 3b: Train the estimator MLP

A tiny 2-layer MLP that maps pooled hidden state at step k to predicted total recurrences needed:

```python
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
```

Train with MSE loss on the convergence_step targets. This should train in minutes on a single GPU since the dataset is just (hidden_dim,) → scalar.

**Where to save:** Save alongside the model checkpoint. The estimator is ~260K parameters for hidden_dim=2048, negligible compared to the main model.

---

## Phase 4: Adaptive inference loop

Create `adaptive_generate.py` that replaces the fixed-depth inference with the adaptive system. This modifies how `model.generate()` works:

```python
def adaptive_forward(model, input_ids, monitor, estimator,
                     probe_step=4, min_r=4, max_r=64, extend_margin=8):
    """Run the model with adaptive recurrence depth.

    Returns the model output and actual recurrence count used.
    """
    # --- Prelude ---
    embedded = model.prelude(input_ids)
    state = model.initialize_state(embedded)

    budget = None
    steps_taken = 0

    for i in range(1, max_r + 1):
        # --- Core recurrence step ---
        state = model.core_block(embedded, state)
        state = model.core_norm(state)
        steps_taken = i

        # --- At probe step, get initial budget estimate ---
        if i == probe_step and estimator is not None:
            pooled = state.detach().mean(dim=1)  # (batch, hidden)
            budget = int(estimator(pooled).mean().item())
            budget = max(min_r, min(budget, max_r))

        # --- From step 3 onward, check progress ---
        if i >= 3:
            result = monitor.step(state)

            if result["signal"] == "exit":
                break

            if result["signal"] == "extend" and budget is not None:
                # Problem is harder than expected — extend budget
                budget = min(budget + extend_margin, max_r)

            if budget is not None and i >= budget and result["signal"] == "continue":
                # Reached budget and state is still moving but not accelerating
                # Give a few more steps before forcing exit
                if i >= budget + extend_margin:
                    break
        else:
            monitor.step(state)

    # --- Coda ---
    output = model.coda(state)
    return output, steps_taken
```

### Integration with the existing eval pipeline

The existing `multi_recurrence_eval.py` evaluates at fixed depths by calling `model(input_ids, num_steps=torch.tensor([0, r]))`. Replace this with `adaptive_forward()` for adaptive evaluation. Add a new eval mode:

```bash
# Fixed depth (existing)
python multi_recurrence_eval.py --recurrences 1 4 8 16 32

# Adaptive depth (new)
python multi_recurrence_eval.py --adaptive \
    --estimator_path checkpoints/difficulty_estimator.pt \
    --exit_threshold 5e-4 \
    --max_r 64
```

---

## Phase 5: Adaptive-aware training (optional, for best results)

The phases above work with a frozen, already-trained retrofitted model. For even better adaptive behavior, modify `train.py` to occasionally train with early exits during the forward pass.

### Minimal change to `train.py`

Add a new training mode where, with some probability (e.g., 10% of batches), the model is trained with a random early exit instead of always using the full sampled `r`:

```python
# In the training loop, after sampling num_steps:
if cfg.adaptive_training and random.random() < 0.1:
    # Train with a random early exit point
    actual_r = random.randint(min_r, num_steps[0] + num_steps[1])
    num_steps = torch.tensor([max(0, actual_r - k), min(actual_r, k)])
```

This teaches the coda to produce reasonable outputs even when the recurrent block hasn't fully converged, which is critical for the adaptive system to work well.

### Adding the estimator loss

Optionally, train the difficulty estimator jointly by adding an auxiliary loss:

```python
# After the recurrence loop, if we have the probe state and know the
# actual convergence behavior from this batch:
if cfg.train_estimator and step % estimator_interval == 0:
    with torch.no_grad():
        # Run extra recurrences to find convergence point
        ...
    estimator_loss = F.mse_loss(estimator(probe_state), target_steps)
    (estimator_loss * estimator_weight).backward()
```

This is more complex and probably not worth it in the first iteration. Train the estimator offline first (Phase 3).

---

## Phase 6: Handling batched inference

The main engineering challenge: different examples in a batch may need different depths. Options:

**Option A: Pad to max in batch.** Simple. Run all examples to the depth of the hardest one. Examples that converge early still compute, but their states don't change (they're at a fixed point). Overhead depends on variance in difficulty within a batch.

**Option B: Bucketing.** Sort examples by estimated difficulty (from the estimator at step k), group into buckets of similar depth, process each bucket. More complex but much more efficient for heterogeneous workloads.

**Option C: Per-token masking.** Track convergence per-token and mask out converged tokens from the recurrence computation. This is the most efficient but requires modifying the attention mask at each recurrence step, which interacts badly with KV caching.

**Recommendation:** Start with Option A. It requires zero changes to the batch processing logic. Move to Option B only if profiling shows significant waste from difficulty variance.

---

## File-by-file change summary

| File | Change type | Description |
|------|-------------|-------------|
| `raven_modeling_minimal.py` | Modify | Add `return_trajectory` flag to forward pass; collect and return intermediate states |
| `adaptive_depth.py` | **New** | `ProgressMonitor` class, `DifficultyEstimator` MLP |
| `collect_difficulty_data.py` | **New** | Script to run model at high depth and record (s_k, r*) pairs |
| `train_estimator.py` | **New** | Train the difficulty estimator MLP on collected data |
| `adaptive_generate.py` | **New** | `adaptive_forward()` function replacing fixed-depth inference |
| `multi_recurrence_eval.py` | Modify | Add `--adaptive` mode calling `adaptive_forward()` |
| `train.py` | Modify (optional) | Add `adaptive_training` flag for early-exit training |

---

## Hyperparameters to tune

| Parameter | Default | What it controls |
|-----------|---------|------------------|
| `probe_step` | 4 | Which recurrence step the estimator reads from |
| `exit_threshold` | 5e-4 | δ below this → model has converged |
| `extend_threshold` | 0.0 | Δ above this → model is struggling, extend budget |
| `extend_margin` | 8 | How many extra steps to grant when extending |
| `max_r` | 64 | Absolute ceiling on recurrences |
| `min_r` | 4 | Never exit before this step |
| `pool` | "mean" | How to reduce δ across tokens ("mean" or "max") |

**Tuning order:** Start with `exit_threshold` (sweep 1e-3 to 1e-5 on GSM8K, measuring accuracy vs. average steps used). Then tune `probe_step` (try 2, 4, 8). The other parameters are secondary.

---

## Expected outcomes

1. **Easy queries** (HellaSwag, simple text completion): Model exits at r ≈ 4–8, saving 4–8× compute vs. fixed r=32.
2. **Medium queries** (ARC, OBQA): Model uses r ≈ 12–20, close to optimal.
3. **Hard queries** (GSM8K multi-step problems): Model uses r ≈ 24–48, sometimes extending beyond the initial budget when the progress monitor detects lack of convergence.
4. **Overall:** Same or better accuracy as fixed r=32, with significantly lower average compute per token.
