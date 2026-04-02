"""Train the DifficultyEstimator MLP on collected (probe_state, convergence_step) data.

Usage:
    python train_estimator.py \
        --data_path difficulty_data.pt \
        --output_path checkpoints/difficulty_estimator.pt \
        --hidden_dim 2048 \
        --max_recurrence 64 \
        --epochs 20 \
        --lr 1e-3
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from jsonargparse import CLI
from typing import Optional

from adaptive_depth import DifficultyEstimator


class ConvergenceDataset(Dataset):
    def __init__(self, records):
        self.probe_states = torch.stack([r["probe_state"] for r in records])
        self.targets = torch.tensor([r["convergence_step"] for r in records], dtype=torch.float)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.probe_states[idx], self.targets[idx]


def main(
    data_path: str,
    output_path: str = "checkpoints/difficulty_estimator.pt",
    hidden_dim: Optional[int] = None,
    max_recurrence: int = 64,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_fraction: float = 0.1,
    device: str = "cuda",
):
    records = torch.load(data_path)
    print(f"Loaded {len(records)} records from {data_path}")

    dataset = ConvergenceDataset(records)

    if hidden_dim is None:
        hidden_dim = dataset.probe_states.shape[-1]
    print(f"Hidden dim: {hidden_dim}, max_recurrence: {max_recurrence}")

    # Train/val split
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = DifficultyEstimator(hidden_dim=hidden_dim, max_recurrence=max_recurrence).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for states, targets in train_loader:
            states, targets = states.to(device), targets.to(device)
            pred = model(states)
            loss = torch.nn.functional.mse_loss(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(targets)
        train_loss /= train_size

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for states, targets in val_loader:
                states, targets = states.to(device), targets.to(device)
                pred = model(states)
                val_loss += torch.nn.functional.mse_loss(pred, targets).item() * len(targets)
        val_loss /= val_size
        scheduler.step()

        print(f"Epoch {epoch:3d}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            import os
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "hidden_dim": hidden_dim,
                    "max_recurrence": max_recurrence,
                },
                output_path,
            )
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")

    print(f"Training complete. Best val MSE: {best_val_loss:.4f}")
    print(f"Best val RMSE (steps): {best_val_loss ** 0.5:.2f}")


if __name__ == "__main__":
    CLI(main)
