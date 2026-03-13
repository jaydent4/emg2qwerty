# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim.lr_scheduler import LRScheduler


def instantiate_optimizer_and_scheduler(
    params: Iterator[nn.Parameter],
    optimizer_config: DictConfig,
    lr_scheduler_config: DictConfig,
) -> dict[str, Any]:
    optimizer = instantiate(optimizer_config, params)
    scheduler = instantiate(lr_scheduler_config.scheduler, optimizer)
    lr_scheduler = instantiate(lr_scheduler_config, scheduler=scheduler)
    return {
        "optimizer": optimizer,
        "lr_scheduler": OmegaConf.to_container(lr_scheduler),
    }


def get_last_checkpoint(checkpoint_dir: Path) -> Path | None:
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def cpus_per_task(gpus_per_node: int, tasks_per_node: int, num_workers: int) -> int:
    """Number of CPUs to request per task per node taking into account
    the number of GPUs and dataloading workers."""
    gpus_per_task = gpus_per_node // tasks_per_node
    if gpus_per_task <= 0:
        return num_workers + 1
    else:
        return (num_workers + 1) * gpus_per_task


class TransformerLRScheduler(LRScheduler):
    """Learning rate scheduler from "Attention Is All You Need".

    Schedule: ``lr = d_model^{-0.5} * min(step^{-0.5}, step * warmup_steps^{-1.5})``

    Args:
        optimizer: Torch optimizer.
        warmup_steps (int): Steps over which to linearly increase the learning rate.
        d_model (int): Model dimension used to scale the rate. (default: 512)
        last_epoch (int): Index of the last epoch. (default: -1)
    """

    def __init__(self, optimizer, warmup_steps: int, d_model: int = 512, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch + 1)
        scale = (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
        return [group["initial_lr"] * scale for group in self.optimizer.param_groups]


class Permute(nn.Module):
    """Permutes tensor dimensions, for use inside ``nn.Sequential``.

    Args:
        *dims (int): Desired ordering of dimensions, passed to ``torch.Tensor.permute``.
    """

    def __init__(self, *dims: int):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)
