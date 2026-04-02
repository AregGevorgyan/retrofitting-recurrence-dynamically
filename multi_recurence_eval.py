from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Optional
import os
import json
from jsonargparse import CLI

from adaptive_depth import ProgressMonitor
from adaptive_generate import adaptive_forward, load_estimator

def get_model_and_tokenizer(model_name, device):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=device,
        torch_dtype=torch.float32,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

@torch.no_grad()
def main(
    model_name: str,
    ckpts: List[int],
    batch_size: int = 64,
    device: str = "cuda",
    eval_file_path: str = "path/eval_dataset/shard-00512.parquet",
    # Adaptive mode options
    adaptive: bool = False,
    estimator_path: Optional[str] = None,
    exit_threshold: float = 5e-4,
    extend_threshold: float = 0.0,
    probe_step: int = 4,
    min_r: int = 4,
    max_r: int = 64,
    extend_margin: int = 8,
    pool: str = "mean",
):
    dataset = load_dataset(
        "parquet",
        data_files=eval_file_path,
    )["train"].select(range(1024))
    dataset.set_format("pt")

    for ckpt in ckpts:
        output = {}
        model, tokenizer = get_model_and_tokenizer(f"{model_name}/model_only_chkpt_{ckpt}", device)

        # Set up adaptive components if requested
        estimator = None
        monitor = None
        if adaptive:
            monitor = ProgressMonitor(
                exit_threshold=exit_threshold,
                extend_threshold=extend_threshold,
                pool=pool,
            )
            if estimator_path is not None:
                estimator = load_estimator(estimator_path, device=device)
            print(f"Adaptive mode: exit_threshold={exit_threshold}, max_r={max_r}, "
                  f"estimator={'yes' if estimator else 'no'}")

        batch = {"input_ids": [], "labels": []}
        steps_used_all = []  # only used in adaptive mode

        for data_idx, inputs in tqdm(enumerate(dataset, start=1)):
            input_ids = inputs["input_ids"][:-1].to(dtype=torch.long, device=device, non_blocking=True)
            mask = ~inputs["attention_mask"].bool()
            labels = torch.where(mask[1:], -100, inputs["input_ids"][1:]).to(
                dtype=torch.long, device=device, non_blocking=True
            )
            batch["input_ids"].append(input_ids)
            batch["labels"].append(labels)

            if (data_idx % batch_size == 0) or (data_idx == len(dataset)):
                input_ids = torch.stack(batch["input_ids"], dim=0)
                labels = torch.stack(batch["labels"], dim=0)
                mask = (labels != -100).float()

                if adaptive:
                    logits, steps_taken = adaptive_forward(
                        model, input_ids, monitor, estimator,
                        probe_step=probe_step, min_r=min_r, max_r=max_r,
                        extend_margin=extend_margin,
                    )
                    steps_used_all.append(steps_taken)

                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100, reduction='none'
                    )
                    loss = loss.view(logits.size(0), logits.size(1))
                    loss_per_sample = (loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

                    this_list = output.get("adaptive", [])
                    this_list.append(loss_per_sample.tolist())
                    output["adaptive"] = this_list
                    del logits
                else:
                    for num_rec in [1, 2, 4, 8, 16, 32, 64]:
                        logits = model(input_ids, num_steps=torch.tensor([num_rec, 0], device=model.device)).logits

                        loss = torch.nn.functional.cross_entropy(
                            logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100, reduction='none'
                        )
                        loss = loss.view(logits.size(0), logits.size(1))
                        loss_per_sample = (loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

                        this_list = output.get(num_rec, [])
                        this_list.append(loss_per_sample.tolist())
                        output[num_rec] = this_list
                        del logits

                batch["input_ids"] = []
                batch["labels"] = []

        if adaptive and steps_used_all:
            import statistics
            print(f"Adaptive steps — mean: {statistics.mean(steps_used_all):.1f}, "
                  f"min: {min(steps_used_all)}, max: {max(steps_used_all)}")
            output["_steps_used"] = steps_used_all

        output_dir = f"{os.getcwd()}/loss_over_rec_eval/{model_name.split('/')[-1]}"
        os.makedirs(output_dir, exist_ok=True)
        suffix = "_adaptive" if adaptive else ""
        with open(f"{output_dir}/chkpt_{ckpt}{suffix}.json", "w") as f:
            json.dump(output, f)

if __name__ == "__main__":
    CLI(main)

# HIP_VISIBLE_DEVICES=0 python multi_recurence_eval.py huginn_llama/YOUR_MODEL [1000] --eval_file_path=YOUR_PATH