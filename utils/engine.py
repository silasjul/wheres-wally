from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainConfig:
    """Container for training hyperparameters."""

    num_epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_interval: int = 10
    gradient_clip: Optional[float] = None  # e.g. 1.0 or None


def _move_to_device(
    images: Any,
    targets: Any,
    device: torch.device,
) -> tuple[Any, Any]:
    """Move images and targets to the given device."""
    if isinstance(images, Tensor):
        images = images.to(device)
    elif isinstance(images, (list, tuple)):
        images = [img.to(device) for img in images]

    if isinstance(targets, (list, tuple)):
        moved_targets: List[Dict[str, Tensor]] = []
        for t in targets:
            moved_targets.append(
                {k: v.to(device) if isinstance(v, Tensor) else v for k, v in t.items()}
            )
        targets = moved_targets

    return images, targets


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Optional[Callable[[Any, Any], Tensor]] = None,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    gradient_clip: Optional[float] = None,
) -> float:
    """
    Train the model for a single epoch.

    - If `loss_fn` is None, this assumes the model itself returns a loss
      when called as `model(images, targets)` (e.g. torchvision detection models).
    - Otherwise, it expects the model to return predictions and uses
      `loss_fn(preds, targets)`.
    """
    model.train()
    device = torch.device(device)
    model.to(device)

    epoch_loss = 0.0
    num_batches = 0

    progress = tqdm(dataloader, desc="Train", leave=False)
    for images, targets in progress:
        images, targets = _move_to_device(images, targets, device)

        optimizer.zero_grad(set_to_none=True)

        if loss_fn is None:
            # Expect model to return a dict of losses or a scalar loss
            outputs = model(images, targets)
            if isinstance(outputs, dict):
                loss = sum(v for v in outputs.values())
            else:
                loss = outputs
        else:
            preds = model(images)
            loss = loss_fn(preds, targets)

        loss.backward()

        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        batch_loss = float(loss.detach().item())
        epoch_loss += batch_loss
        num_batches += 1
        progress.set_postfix(loss=batch_loss)

    return epoch_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Optional[Callable[[Any, Any], Tensor]] = None,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
) -> float:
    """
    Simple evaluation loop that mirrors `train_one_epoch`.

    Returns average loss over the validation set.
    """
    model.eval()
    device = torch.device(device)
    model.to(device)

    epoch_loss = 0.0
    num_batches = 0

    progress = tqdm(dataloader, desc="Eval", leave=False)
    for images, targets in progress:
        images, targets = _move_to_device(images, targets, device)

        if loss_fn is None:
            outputs = model(images, targets)
            if isinstance(outputs, dict):
                loss = sum(v for v in outputs.values())
            else:
                loss = outputs
        else:
            preds = model(images)
            loss = loss_fn(preds, targets)

        batch_loss = float(loss.detach().item())
        epoch_loss += batch_loss
        num_batches += 1
        progress.set_postfix(loss=batch_loss)

    return epoch_loss / max(num_batches, 1)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    config: Optional[TrainConfig] = None,
    loss_fn: Optional[Callable[[Any, Any], Tensor]] = None,
    lr_scheduler: Optional[Any] = None,
) -> Dict[str, List[float]]:
    """
    High-level training loop.

    Parameters
    ----------
    model : nn.Module
        Your model (e.g. torchvision detection model).
    train_loader : DataLoader
        Dataloader for the training set.
    val_loader : DataLoader | None
        Optional validation dataloader.
    optimizer : torch.optim.Optimizer
    config : TrainConfig | None
        Training hyperparameters (epochs, device, etc.).
    loss_fn : callable | None
        Custom loss; if None, assumes model returns loss.
    lr_scheduler : any
        Optional LR scheduler with step() method.

    Returns
    -------
    history : dict
        Contains lists: `train_loss`, and optionally `val_loss`.
    """
    if config is None:
        config = TrainConfig()

    # Explicitly show whether we're using GPU or CPU
    print(f"Using device: {config.device}")

    history: Dict[str, List[float]] = {"train_loss": []}
    if val_loader is not None:
        history["val_loss"] = []

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=config.device,
            gradient_clip=config.gradient_clip,
        )
        history["train_loss"].append(train_loss)
        print(f"  train_loss: {train_loss:.4f}")

        if val_loader is not None:
            val_loss = evaluate(
                model=model,
                dataloader=val_loader,
                loss_fn=loss_fn,
                device=config.device,
            )
            history["val_loss"].append(val_loss)
            print(f"  val_loss:   {val_loss:.4f}")

        if lr_scheduler is not None:
            lr_scheduler.step()

    return history


__all__ = ["TrainConfig", "train_one_epoch", "evaluate", "fit"]


