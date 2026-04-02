"""Collect (probe_state, convergence_step) pairs for training the difficulty estimator.

Usage:
    python collect_difficulty_data.py \
        --model_name smcleish/huginn-0125 \
        --eval_file_path path/to/eval.parquet \
        --output_path difficulty_data.pt \
        --max_r 64 \
        --probe_step 4 \
        --exit_threshold 5e-4
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from jsonargparse import CLI
from typing import Optional

from adaptive_depth import ProgressMonitor


@torch.no_grad()
def collect_convergence_data(model, dataloader, device, probe_step=4, max_r=64, exit_threshold=5e-4):
    """Run model at max_r recurrences, record (s_probe, r*) pairs."""
    records = []
    monitor = ProgressMonitor(exit_threshold=exit_threshold)

    for batch in tqdm(dataloader, desc="Collecting convergence data"):
        input_ids = batch["input_ids"].to(device)

        # Run forward with trajectory collection
        outputs = model(
            input_ids,
            num_steps=torch.tensor([max_r, 0], device=device),
            output_details={
                "return_logits": False,
                "return_latents": False,
                "return_head": False,
                "return_stats": False,
                "return_trajectory": True,
            },
        )

        trajectory = outputs.trajectory  # list of (batch, seq, hidden)
        if trajectory is None or len(trajectory) == 0:
            continue

        # Find convergence step per batch element
        for b in range(input_ids.shape[0]):
            monitor.reset()
            convergence_step = max_r  # default if never converges
            for i, state in enumerate(trajectory):
                result = monitor.step(state[b : b + 1])
                if result["signal"] == "exit" and i > probe_step:
                    convergence_step = i
                    break

            # Guard: need enough steps to have a probe state
            if probe_step >= len(trajectory):
                continue

            # Record the probe state (pooled over sequence dim) and target
            probe_state = trajectory[probe_step][b].mean(dim=0)  # (hidden_dim,)
            records.append(
                {
                    "probe_state": probe_state.cpu(),
                    "convergence_step": convergence_step,
                }
            )

    return records


def main(
    model_name: str,
    eval_file_path: str,
    output_path: str = "difficulty_data.pt",
    batch_size: int = 16,
    max_samples: int = 4096,
    probe_step: int = 4,
    max_r: int = 64,
    exit_threshold: float = 5e-4,
    device: str = "cuda",
    ckpt: Optional[int] = None,
):
    load_name = f"{model_name}/model_only_chkpt_{ckpt}" if ckpt is not None else model_name

    model = AutoModelForCausalLM.from_pretrained(
        load_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=device,
        torch_dtype=torch.float32,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(load_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("parquet", data_files=eval_file_path)["train"]
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    dataset.set_format("pt")

    # Build batched dataloader manually
    def batch_iter(dataset, batch_size):
        batch = []
        for item in dataset:
            input_ids = item["input_ids"][:-1].to(dtype=torch.long)
            batch.append(input_ids)
            if len(batch) == batch_size:
                # Pad to same length
                max_len = max(t.shape[0] for t in batch)
                padded = torch.stack(
                    [torch.nn.functional.pad(t, (0, max_len - t.shape[0])) for t in batch]
                )
                yield {"input_ids": padded}
                batch = []
        if batch:
            max_len = max(t.shape[0] for t in batch)
            padded = torch.stack(
                [torch.nn.functional.pad(t, (0, max_len - t.shape[0])) for t in batch]
            )
            yield {"input_ids": padded}

    records = collect_convergence_data(
        model,
        batch_iter(dataset, batch_size),
        device=device,
        probe_step=probe_step,
        max_r=max_r,
        exit_threshold=exit_threshold,
    )

    torch.save(records, output_path)
    print(f"Saved {len(records)} records to {output_path}")

    # Print distribution summary
    steps = [r["convergence_step"] for r in records]
    steps_t = torch.tensor(steps, dtype=torch.float)
    print(f"Convergence step stats: mean={steps_t.mean():.1f}, "
          f"std={steps_t.std():.1f}, min={steps_t.min():.0f}, max={steps_t.max():.0f}")


if __name__ == "__main__":
    CLI(main)
