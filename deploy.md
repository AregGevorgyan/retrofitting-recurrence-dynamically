# Deployment & Testing Guide: Adaptive Depth for Retrofitted Recurrent LLMs

## Overview of what was added

| File | Type | Purpose |
|------|------|---------|
| `convert_pretrained_model/raven_modeling_minimal_llama.py` | Modified | `return_trajectory` flag; `iterate_forward` now returns intermediate states |
| `adaptive_depth.py` | New | `ProgressMonitor` + `DifficultyEstimator` |
| `collect_difficulty_data.py` | New | Runs model at high depth, records `(probe_state, convergence_step)` pairs |
| `train_estimator.py` | New | Trains the estimator MLP on collected data |
| `adaptive_generate.py` | New | `adaptive_forward()` — replaces fixed-depth inference |
| `multi_recurence_eval.py` | Modified | `--adaptive` flag for adaptive-depth validation loss eval |

---

## Critical prerequisite: propagate the modeling changes

The modifications to `raven_modeling_minimal_llama.py` only affect **newly converted** models.
If you are using an already-converted model checkpoint (e.g. downloaded from HuggingFace),
its own copy of `raven_modeling_minimal.py` inside the checkpoint directory is what gets loaded
by `AutoModelForCausalLM` — not the file in `convert_pretrained_model/`.

**You must manually sync the changes** into every converted model checkpoint you plan to use:

```bash
# For a Llama-based converted model:
cp convert_pretrained_model/raven_modeling_minimal_llama.py \
   PATH_TO_CONVERTED_MODEL/raven_modeling_minimal.py
```

The three changes to port are:
1. `trajectory: Optional[list] = None` field on `CausalLMOutputRecurrentLatents`
2. `return_trajectory` parameter on `iterate_forward` + trajectory collection inside the loop
3. `return_trajectory = output_details.get("return_trajectory", False)` in `forward` + `trajectory=trajectory` in the return

Verify the sync worked before running anything:
```bash
grep -n "return_trajectory" PATH_TO_CONVERTED_MODEL/raven_modeling_minimal.py
# Should print 4 lines
```

---

## Step 0: Smoke test (no GPU required)

Confirm the new modules import cleanly and the `ProgressMonitor` logic is correct:

```python
import torch
from adaptive_depth import ProgressMonitor, DifficultyEstimator

monitor = ProgressMonitor(exit_threshold=5e-4, pool="mean")

# Simulate a converging hidden state
state = torch.randn(1, 16, 2048)
for i in range(20):
    state = state * 0.95 + torch.randn_like(state) * 0.01  # shrinking noise → convergence
    result = monitor.step(state)
    print(f"step {i+1:2d}: delta={result['delta']:.6f}, signal={result['signal']}")
# Expected: signal should transition from "continue" to "exit" as delta falls below 5e-4

# Confirm DifficultyEstimator forward pass
est = DifficultyEstimator(hidden_dim=2048, max_recurrence=64)
pooled = torch.randn(4, 2048)
pred = est(pooled)
assert pred.shape == (4,)
assert (pred >= 0).all() and (pred <= 64).all()
print("Smoke test passed.")
```

---

## Step 1: Test trajectory collection

Load any converted model and confirm `return_trajectory` works end-to-end:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "PATH_TO_CONVERTED_MODEL",
    trust_remote_code=True,
    torch_dtype=torch.float32,
    device_map="cuda",
)
tokenizer = AutoTokenizer.from_pretrained("PATH_TO_CONVERTED_MODEL")

input_ids = tokenizer("The capital of France is", return_tensors="pt").input_ids.cuda()

with torch.no_grad():
    out = model(
        input_ids,
        num_steps=torch.tensor([8, 0], device="cuda"),
        output_details={
            "return_logits": False,
            "return_latents": False,
            "return_head": False,
            "return_stats": False,
            "return_trajectory": True,
        },
    )

assert out.trajectory is not None
assert len(out.trajectory) == 8          # one tensor per recurrence step
assert out.trajectory[0].shape == (1, input_ids.shape[1], model.config.n_embd)
print(f"Trajectory OK: {len(out.trajectory)} steps, shape {out.trajectory[0].shape}")
```

---

## Step 2: Test adaptive_forward

Confirm the adaptive loop runs and produces logits of the right shape:

```python
import torch
from adaptive_depth import ProgressMonitor
from adaptive_generate import adaptive_forward

monitor = ProgressMonitor(exit_threshold=5e-4)

with torch.no_grad():
    logits, steps = adaptive_forward(
        model, input_ids, monitor,
        estimator=None,   # no estimator yet
        probe_step=4, min_r=4, max_r=32, extend_margin=8,
    )

assert logits.shape == (1, input_ids.shape[1], model.config.padded_vocab_size)
print(f"adaptive_forward OK — steps used: {steps}")
```

**Known issue to watch for:** `adaptive_forward` resets `block_idx` to
`n_layers_in_prelude - 1` at every recurrence step. This is intentional for the
no-KV-cache path (recurrence is stateless in that direction), but if you add KV-cache
support later, block indexing will need to carry state across steps.

Compare the output quality against a fixed-depth call to make sure logits are in the
same ballpark:

```python
with torch.no_grad():
    fixed_out = model(input_ids, num_steps=torch.tensor([steps, 0], device="cuda"))

diff = (logits - fixed_out.logits).abs().max()
print(f"Max logit diff vs fixed-{steps}: {diff:.4f}")
# Will not be zero (different code paths), but should be < ~0.5 for a working model
```

---

## Step 3: Collect difficulty data

Run on your evaluation parquet (the same one used for validation loss). Start small
(512–1024 samples) to check everything runs, then scale to 4096+ for real training data.

```bash
python collect_difficulty_data.py \
    --model_name PATH_TO_CONVERTED_MODEL \
    --eval_file_path PATH_TO_EVAL_PARQUET \
    --output_path difficulty_data.pt \
    --max_r 64 \
    --probe_step 4 \
    --exit_threshold 5e-4 \
    --batch_size 8 \
    --max_samples 4096 \
    --device cuda
```

Check the output distribution — if nearly all examples converge at `max_r` (the default),
your `exit_threshold` is too tight. If nearly all converge at step 4, it's too loose.
Aim for a roughly spread distribution across the range.

```bash
python - <<'EOF'
import torch
records = torch.load("difficulty_data.pt")
steps = [r["convergence_step"] for r in records]
import collections
dist = collections.Counter(steps)
for k in sorted(dist):
    print(f"  r={k:3d}: {dist[k]:4d} samples ({100*dist[k]/len(steps):.1f}%)")
EOF
```

**Threshold tuning guide:**

| `exit_threshold` | Behaviour |
|-----------------|-----------|
| `1e-3` | Exits very early; most samples converge in <10 steps |
| `5e-4` | Recommended starting point (matches Huginn's KL threshold) |
| `1e-4` | Conservative; only declares convergence when state is nearly static |
| `1e-5` | Almost never exits early; useful as a sanity-check upper bound |

---

## Step 4: Train the difficulty estimator

```bash
mkdir -p checkpoints

python train_estimator.py \
    --data_path difficulty_data.pt \
    --output_path checkpoints/difficulty_estimator.pt \
    --max_recurrence 64 \
    --epochs 30 \
    --batch_size 256 \
    --lr 1e-3 \
    --device cuda
```

This should train in under 5 minutes on any GPU. Watch the val MSE — it will converge
to a floor determined by how predictable convergence depth actually is from step-4 states.
A val RMSE of 8–15 steps is reasonable for a first iteration.

If val loss doesn't decrease at all, the estimator can't learn from step-4 states; try
`--probe_step 8` in the data collection step and retrain.

---

## Step 5: Run adaptive eval

### Monitor-only (no estimator)

```bash
python multi_recurence_eval.py \
    PATH_TO_CONVERTED_MODEL [CKPT_NUMBER] \
    --eval_file_path PATH_TO_EVAL_PARQUET \
    --adaptive true \
    --exit_threshold 5e-4 \
    --max_r 64 \
    --min_r 4 \
    --device cuda
```

### With estimator

```bash
python multi_recurence_eval.py \
    PATH_TO_CONVERTED_MODEL [CKPT_NUMBER] \
    --eval_file_path PATH_TO_EVAL_PARQUET \
    --adaptive true \
    --estimator_path checkpoints/difficulty_estimator.pt \
    --exit_threshold 5e-4 \
    --max_r 64 \
    --min_r 4 \
    --device cuda
```

Results are saved to `loss_over_rec_eval/<model_name>/chkpt_<N>_adaptive.json`.

### Run fixed-depth baseline for comparison

```bash
python multi_recurence_eval.py \
    PATH_TO_CONVERTED_MODEL [CKPT_NUMBER] \
    --eval_file_path PATH_TO_EVAL_PARQUET \
    --device cuda
```

Results saved to `loss_over_rec_eval/<model_name>/chkpt_<N>.json`.

### Compare results

```python
import json, statistics

with open("loss_over_rec_eval/YOUR_MODEL/chkpt_N.json") as f:
    fixed = json.load(f)
with open("loss_over_rec_eval/YOUR_MODEL/chkpt_N_adaptive.json") as f:
    adaptive = json.load(f)

for r in [4, 8, 16, 32, 64]:
    losses = [l for batch in fixed[str(r)] for l in batch]
    print(f"Fixed  r={r:2d}: mean_loss={statistics.mean(losses):.4f}")

adaptive_losses = [l for batch in adaptive["adaptive"] for l in batch]
mean_steps = statistics.mean(adaptive["_steps_used"])
print(f"Adaptive:      mean_loss={statistics.mean(adaptive_losses):.4f}  "
      f"(avg {mean_steps:.1f} steps)")
```

**What to look for:** Adaptive loss should be close to fixed loss at `r=mean_steps`.
If it's significantly worse, `exit_threshold` is too aggressive (exiting before convergence).

---

## Hyperparameter tuning order

1. **`exit_threshold`** — most impactful. Sweep `[1e-3, 5e-4, 2e-4, 1e-4]` and plot
   accuracy vs. average steps. Pick the point where accuracy stops improving as threshold tightens.

2. **`probe_step`** — try `[2, 4, 8]`. Later steps have more signal but cost more compute.
   For very fast models (low mean recurrence), `probe_step=2` may be necessary.

3. **`min_r`** — keep at 4 unless you observe degenerate outputs at very low step counts,
   in which case raise to 6 or 8.

4. **`pool`** — `"mean"` is default. Switch to `"max"` if single-token difficulty matters
   (e.g. rare entity names, numbers mid-computation). Max is slower to exit.

5. **`extend_threshold`** — leave at `0.0` initially. Raise toward `0.1` if the model
   extends too aggressively on easy inputs.

---

## Known limitations and future work

### Per-sample depth in batched inference
`adaptive_forward` currently runs all batch elements to the depth of the longest one
(Option A from the plan). Examples that converge early continue computing but their
converged states don't change meaningfully. To measure actual waste:

```python
# After collecting per-sample convergence data:
import torch
records = torch.load("difficulty_data.pt")
steps = torch.tensor([r["convergence_step"] for r in records], dtype=torch.float)
# Simulate batch of size 64, compute overhead ratio
import random
overhead = []
for _ in range(1000):
    batch = random.sample(steps.tolist(), 64)
    overhead.append(max(batch) / (sum(batch) / len(batch)))
print(f"Mean compute overhead from padding to max: {sum(overhead)/len(overhead):.2f}x")
```

If overhead > 1.5×, implement difficulty bucketing (sort by estimated depth, group similar ones).

### KV-cache incompatibility
`adaptive_forward` bypasses the model's KV-cache mechanism. For autoregressive generation
(token-by-token), use the model's existing `generate()` with `iterate_one_step()` instead.
Adaptive depth for generation is a separate workstream.

### Estimator training data quality
The estimator is trained on convergence steps measured at `exit_threshold=5e-4`. If you
later change the threshold for inference, retrain the estimator with the new threshold.

### OLMo models
`adaptive_generate.py` uses `model.transformer.wte`, `model.rotary_emb`, etc., which match
the Llama-based architecture. For OLMo-based models (using `raven_modeling_minimal_olmo.py`),
verify that these attribute names are the same. If not, the `adaptive_forward` function will
need a small adaptation for the embedding and rotary embedding access patterns.
