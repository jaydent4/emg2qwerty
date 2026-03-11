# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import MetricCollection

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    CNNTransformerEncoder,
    ConformerEncoder,
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
)
from emg2qwerty.transforms import Transform


class WindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform: Transform[np.ndarray, torch.Tensor],
        val_transform: Transform[np.ndarray, torch.Tensor],
        test_transform: Transform[np.ndarray, torch.Tensor],
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.train_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                )
                for hdf5_path in self.train_sessions
            ]
        )
        self.val_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.val_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.val_sessions
            ]
        )
        self.test_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.test_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.test_sessions
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )


class TDSConvCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        # Model
        # inputs: (T, N, bands=2, electrode_channels=16, freq)
        self.model = nn.Sequential(
            # (T, N, bands=2, C=16, freq)
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            # (T, N, bands=2, mlp_features[-1])
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            # (T, N, num_features)
            nn.Flatten(start_dim=2),
            TDSConvEncoder(
                num_features=num_features,
                block_channels=block_channels,
                kernel_width=kernel_width,
            ),
            # (T, N, num_classes)
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


class CNNTransformerCTCModule(pl.LightningModule):
    """A CNN + Transformer model for EMG-to-text decoding with CTC loss.

    Architecture:
      1. ``SpectrogramNorm``   – per-channel 2D batch norm on log spectrograms.
      2. ``CNNTransformerEncoder`` – 2D CNN extracts local freq/electrode
         features; Transformer models long-range temporal context.
      3. Linear + LogSoftmax  – CTC output projection.

    Args:
        in_features (int): ``freq_bins * electrode_channels``
            (e.g. 33 * 16 = 528 for n_fft=64). Kept for documentation /
            config compatibility; the CNN infers dimensions at runtime.
        cnn_channels (list): Output channels per CNN layer in
            ``CNNFeatureExtractor``. (default: ``[64, 128, 256, 256]``)
        model_dim (int): Transformer model dimension. (default: 256)
        nhead (int): Number of attention heads. (default: 8)
        num_layers (int): Number of Transformer encoder layers. (default: 4)
        dim_feedforward (int): FFN inner dimension. (default: 1024)
        dropout (float): Dropout probability used throughout. (default: 0.1)
        attn_chunk_size (int): Max time steps per attention chunk during
            inference to bound memory on long sessions. (default: 1000)
    """

    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        cnn_channels: Sequence[int],
        model_dim: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        attn_chunk_size: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        left_context_frames: int = 0,
        right_context_frames: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Model
        # inputs: (T, N, bands=2, electrode_channels=16, freq)
        self.model = nn.Sequential(
            # (T, N, bands=2, C=16, freq)
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            # (T, N, model_dim)
            CNNTransformerEncoder(
                bands=self.NUM_BANDS,
                electrode_channels=self.ELECTRODE_CHANNELS,
                cnn_channels=cnn_channels,
                model_dim=model_dim,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                attn_chunk_size=attn_chunk_size,
            ),
            # (T, N, num_classes)
            nn.Linear(model_dim, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        emissions = self.forward(inputs)

        # CNN + Transformer preserves the time dimension (no temporal striding),
        # so emission_lengths == input_lengths.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,               # (T, N, num_classes)
            targets=targets.transpose(0, 1),   # (T, N) -> (N, T)
            input_lengths=emission_lengths,    # (N,)
            target_lengths=target_lengths,     # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def on_test_epoch_start(self) -> None:
        self._test_emission_chunks: list[np.ndarray] = []
        self._test_targets: list[int] = []

    def test_step(self, batch: dict[str, torch.Tensor], *args, **kwargs) -> None:
        inputs = batch["inputs"]
        targets = batch["targets"]
        target_lengths = batch["target_lengths"]
        N = len(target_lengths)
        emissions = self.forward(inputs)
        l = self.hparams.left_context_frames
        r = self.hparams.right_context_frames
        T = emissions.shape[0]
        l_trim = min(l, T)
        r_trim = min(r, T - l_trim)
        emissions_np = emissions.detach().cpu().numpy()
        for i in range(N):
            trimmed = emissions_np[l_trim : T - r_trim if r_trim > 0 else T, i, :]
            self._test_emission_chunks.append(trimmed)
            tgt_len = int(target_lengths[i])
            self._test_targets.extend(targets[:tgt_len, i].cpu().numpy().tolist())

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        if not self._test_emission_chunks:
            return
        full_emissions = np.concatenate(self._test_emission_chunks, axis=0)
        T_total = full_emissions.shape[0]
        predictions = self.decoder.decode_batch(
            emissions=full_emissions[:, np.newaxis, :],
            emission_lengths=np.array([T_total], dtype=np.int32),
        )
        target = LabelData.from_labels(self._test_targets)
        metrics = self.metrics["test_metrics"]
        metrics.update(prediction=predictions[0], target=target)
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


class ConformerCTCModule(pl.LightningModule):
    """Conformer (CNN + Conformer blocks) for EMG-to-text decoding with CTC loss.

    Architecture:
      1. ``SpectrogramNorm``  – per-channel 2D batch norm on log spectrograms.
      2. ``ConformerEncoder`` – 2D CNN reduces freq/electrode dims to a feature
         vector per time step; Conformer blocks model both local (depthwise conv)
         and global (attention) temporal context.
      3. Linear + LogSoftmax – CTC output projection.

    Test inference accumulates trimmed windowed emissions before a single
    decode, matching the windowed training distribution.

    Args:
        in_features (int): Kept for config compatibility; CNN infers dims.
        cnn_channels (list): Output channels per 2D CNN layer.
            (default: ``[64, 128, 256]``)
        model_dim (int): Conformer model dimension. (default: 256)
        nhead (int): Attention heads per block. (default: 4)
        num_layers (int): Number of Conformer blocks. (default: 6)
        ff_expansion_factor (int): FFN hidden dim multiplier. (default: 4)
        conv_kernel_size (int): Depthwise conv kernel size. (default: 31)
        dropout (float): Dropout probability. (default: 0.1)
        attn_chunk_size (int): Max time steps per attention chunk during
            inference. (default: 1000)
        left_context_frames (int): Emission frames to trim on the left per
            window (equals left_padding // hop_length). (default: 0)
        right_context_frames (int): Emission frames to trim on the right per
            window (equals right_padding // hop_length). (default: 0)
    """

    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        cnn_channels: Sequence[int],
        model_dim: int,
        nhead: int,
        num_layers: int,
        ff_expansion_factor: int,
        conv_kernel_size: int,
        dropout: float,
        attn_chunk_size: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        left_context_frames: int = 0,
        right_context_frames: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            ConformerEncoder(
                bands=self.NUM_BANDS,
                electrode_channels=self.ELECTRODE_CHANNELS,
                cnn_channels=cnn_channels,
                model_dim=model_dim,
                nhead=nhead,
                num_layers=num_layers,
                ff_expansion_factor=ff_expansion_factor,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
                attn_chunk_size=attn_chunk_size,
            ),
            nn.Linear(model_dim, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        emissions = self.forward(inputs)

        # Conformer preserves the time dimension (no temporal striding).
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0, 1),
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
        )

        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        metrics = self.metrics[f"{phase}_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def on_test_epoch_start(self) -> None:
        self._test_emission_chunks: list[np.ndarray] = []
        self._test_targets: list[int] = []

    def test_step(self, batch: dict[str, torch.Tensor], *args, **kwargs) -> None:
        inputs = batch["inputs"]
        targets = batch["targets"]
        target_lengths = batch["target_lengths"]
        N = len(target_lengths)
        emissions = self.forward(inputs)
        l = self.hparams.left_context_frames
        r = self.hparams.right_context_frames
        T = emissions.shape[0]
        l_trim = min(l, T)
        r_trim = min(r, T - l_trim)
        emissions_np = emissions.detach().cpu().numpy()
        for i in range(N):
            trimmed = emissions_np[l_trim : T - r_trim if r_trim > 0 else T, i, :]
            self._test_emission_chunks.append(trimmed)
            tgt_len = int(target_lengths[i])
            self._test_targets.extend(targets[:tgt_len, i].cpu().numpy().tolist())

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        if not self._test_emission_chunks:
            return
        full_emissions = np.concatenate(self._test_emission_chunks, axis=0)
        T_total = full_emissions.shape[0]
        predictions = self.decoder.decode_batch(
            emissions=full_emissions[:, np.newaxis, :],
            emission_lengths=np.array([T_total], dtype=np.int32),
        )
        target = LabelData.from_labels(self._test_targets)
        metrics = self.metrics["test_metrics"]
        metrics.update(prediction=predictions[0], target=target)
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )
