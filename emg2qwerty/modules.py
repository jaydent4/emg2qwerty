# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections.abc import Sequence

import torch
from torch import nn


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)


class ConformerFeedForward(nn.Module):
    """Conformer feed-forward module with pre-norm and half-step residual.

    Input/output shape: (T, N, d_model).
    """

    def __init__(self, d_model: int, expansion_factor: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion_factor, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.5 * self.ff(self.norm(x))


class ConformerConvolution(nn.Module):
    """Conformer convolution module: pointwise → GLU → depthwise → BN → SiLU → pointwise.

    Input/output shape: (T, N, d_model).

    Args:
        d_model (int): Model dimension.
        kernel_size (int): Depthwise conv kernel size (must be odd). (default: 31)
        dropout (float): Dropout applied after final pointwise conv. (default: 0.1)
    """

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_expand = nn.Linear(d_model, 2 * d_model)
        self.glu = nn.GLU(dim=-1)
        self.depthwise = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_project = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)                          # (T, N, d_model)
        x = self.pointwise_expand(x)              # (T, N, 2*d_model)
        x = self.glu(x)                           # (T, N, d_model)
        T, N, d = x.shape
        x = x.permute(1, 2, 0)                    # (N, d_model, T)
        x = self.depthwise(x)                     # (N, d_model, T)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = x.permute(2, 0, 1)                    # (T, N, d_model)
        x = self.pointwise_project(x)
        x = self.dropout(x)
        return residual + x


class ConformerBlock(nn.Module):
    """Single Conformer block: ½FF → MHSA → Conv → ½FF → LayerNorm.

    Input/output shape: (T, N, d_model).

    Args:
        d_model (int): Model dimension.
        nhead (int): Number of attention heads.
        ff_expansion_factor (int): FFN hidden dim multiplier. (default: 4)
        conv_kernel_size (int): Depthwise conv kernel size. (default: 31)
        dropout (float): Dropout applied throughout. (default: 0.1)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        ff_expansion_factor: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ff1 = ConformerFeedForward(d_model, ff_expansion_factor, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.conv = ConformerConvolution(d_model, conv_kernel_size, dropout)
        self.ff2 = ConformerFeedForward(d_model, ff_expansion_factor, dropout)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ff1(x)
        residual = x
        x_norm = self.attn_norm(x)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm)
        x = residual + self.attn_dropout(x_attn)
        x = self.conv(x)
        x = self.ff2(x)
        return self.norm_out(x)


class ConformerEncoder(nn.Module):
    """2D CNN feature extractor followed by a stack of Conformer blocks.

    The CNN reduces the (bands, electrode_channels, freq_bins) spatial dims
    to a single feature vector per time step. The Conformer blocks then model
    both local (via depthwise conv) and global (via attention) temporal context.

    Input shape:  (T, N, bands, electrode_channels, freq_bins)
    Output shape: (T, N, model_dim)

    Args:
        bands (int): Number of EMG bands.
        electrode_channels (int): Electrode channels per band.
        cnn_channels (list): Output channels per CNN layer.
            (default: ``(64, 128, 256)``)
        model_dim (int): Conformer model dimension. (default: 256)
        nhead (int): Number of attention heads. (default: 4)
        num_layers (int): Number of Conformer blocks. (default: 6)
        ff_expansion_factor (int): FFN hidden dim multiplier. (default: 4)
        conv_kernel_size (int): Depthwise conv kernel size in each block.
            (default: 31)
        dropout (float): Dropout applied throughout. (default: 0.1)
        attn_chunk_size (int): Max time steps per attention chunk during
            inference to bound memory on long sequences. (default: 1000)
    """

    def __init__(
        self,
        bands: int,
        electrode_channels: int,
        cnn_channels: Sequence[int] = (64, 128, 256),
        model_dim: int = 256,
        nhead: int = 4,
        num_layers: int = 6,
        ff_expansion_factor: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        attn_chunk_size: int = 1000,
    ) -> None:
        super().__init__()
        self.attn_chunk_size = attn_chunk_size

        self.cnn = CNNFeatureExtractor(
            bands=bands,
            electrode_channels=electrode_channels,
            cnn_channels=cnn_channels,
            dropout=dropout,
        )
        self.input_proj = nn.Linear(self.cnn.out_channels, model_dim)
        self.pos_encoding = PositionalEncoding(model_dim, dropout=dropout)
        self.blocks = nn.ModuleList([
            ConformerBlock(
                d_model=model_dim,
                nhead=nhead,
                ff_expansion_factor=ff_expansion_factor,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)           # (T, N, cnn_out_channels)
        x = self.input_proj(x)    # (T, N, model_dim)
        x = self.pos_encoding(x)  # (T, N, model_dim)
        if not self.training and x.size(0) > self.attn_chunk_size:
            # Chunk only the attention — depthwise conv handles boundaries locally
            chunks = x.split(self.attn_chunk_size, dim=0)
            outs = []
            for chunk in chunks:
                for block in self.blocks:
                    chunk = block(chunk)
                outs.append(chunk)
            x = torch.cat(outs, dim=0)
        else:
            for block in self.blocks:
                x = block(x)
        return x  # (T, N, model_dim)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs.

    Input/output shape: (T, N, d_model).

    Args:
        d_model (int): Embedding / model dimension.
        dropout (float): Dropout probability applied after adding positional
            encodings. (default: 0.1)
        max_len (int): Maximum sequence length supported. (default: 10000)
    """

    def __init__(
        self, d_model: int, dropout: float = 0.1, max_len: int = 10000
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(0)
        if T > self.pe.size(0):
            # Compute PE on-the-fly for sequences longer than max_len
            d_model = self.pe.size(-1)
            position = torch.arange(T, device=x.device, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, device=x.device, dtype=torch.float)
                * (-math.log(10000.0) / d_model)
            )
            pe = torch.zeros(T, 1, d_model, device=x.device)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            x = x + pe
        else:
            x = x + self.pe[:T]
        return self.dropout(x)


class CNNFeatureExtractor(nn.Module):
    """Extracts local spectrogram features using 2D convolutions.

    Treats all electrode channels across both bands as the channel dimension
    and (freq, time) as the spatial dimensions. Frequency is progressively
    downsampled while the time dimension is kept intact to preserve temporal
    resolution for CTC.

    Input shape:  (T, N, bands, electrode_channels, freq_bins)
    Output shape: (T, N, cnn_channels[-1])

    Args:
        bands (int): Number of EMG bands (e.g. 2 for left/right).
        electrode_channels (int): Electrode channels per band (e.g. 16).
        cnn_channels (list): Number of output channels per Conv2d layer.
            Frequency stride 2 is applied at every even-indexed layer (0-based)
            starting from index 1. (default: ``(64, 128, 256, 256)``)
        dropout (float): Dropout probability after each conv block.
            (default: 0.1)
    """

    def __init__(
        self,
        bands: int,
        electrode_channels: int,
        cnn_channels: Sequence[int] = (64, 128, 256, 256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        in_ch = bands * electrode_channels
        layers: list[nn.Module] = []
        for i, out_ch in enumerate(cnn_channels):
            # Downsample freq every other layer (indices 1, 3, ...)
            freq_stride = 2 if i % 2 == 1 else 1
            layers.extend(
                [
                    nn.Conv2d(
                        in_ch,
                        out_ch,
                        kernel_size=(3, 3),
                        stride=(freq_stride, 1),
                        padding=(1, 1),
                    ),
                    nn.BatchNorm2d(out_ch),
                    nn.GELU(),
                    nn.Dropout2d(p=dropout),
                ]
            )
            in_ch = out_ch

        self.cnn = nn.Sequential(*layers)
        # Pool remaining freq bins to a single value
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.out_channels = cnn_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = x.shape
        # (T, N, bands, C, freq) -> (N, bands*C, freq, T)
        x = x.permute(1, 2, 3, 4, 0).reshape(N, bands * C, freq, T)
        # (N, bands*C, freq, T) -> (N, out_channels, freq', T)
        x = self.cnn(x)
        # (N, out_channels, freq', T) -> (N, out_channels, 1, T)
        x = self.freq_pool(x)
        # (N, out_channels, 1, T) -> (T, N, out_channels)
        return x.squeeze(2).permute(2, 0, 1)


class CNNTransformerEncoder(nn.Module):
    """A 2D CNN feature extractor followed by a Transformer encoder.

    The CNN captures local frequency-electrode patterns at each time step
    while the Transformer models long-range temporal dependencies.
    At inference time, very long sequences are processed in non-overlapping
    chunks to keep memory bounded.

    Input shape:  (T, N, bands, electrode_channels, freq_bins)
    Output shape: (T, N, model_dim)

    Args:
        bands (int): Number of EMG bands.
        electrode_channels (int): Electrode channels per band.
        cnn_channels (list): Output channels per CNN layer.
            (default: ``(64, 128, 256, 256)``)
        model_dim (int): Transformer model dimension. (default: 256)
        nhead (int): Number of attention heads. (default: 8)
        num_layers (int): Number of Transformer encoder layers. (default: 4)
        dim_feedforward (int): FFN inner dimension. (default: 1024)
        dropout (float): Dropout applied throughout. (default: 0.1)
        attn_chunk_size (int): Maximum sequence length per attention chunk
            during inference to avoid OOM on long sessions. (default: 1000)
    """

    def __init__(
        self,
        bands: int,
        electrode_channels: int,
        cnn_channels: Sequence[int] = (64, 128, 256, 256),
        model_dim: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        attn_chunk_size: int = 1000,
    ) -> None:
        super().__init__()

        self.attn_chunk_size = attn_chunk_size

        self.cnn = CNNFeatureExtractor(
            bands=bands,
            electrode_channels=electrode_channels,
            cnn_channels=cnn_channels,
            dropout=dropout,
        )
        self.input_proj = nn.Linear(self.cnn.out_channels, model_dim)
        self.pos_encoding = PositionalEncoding(model_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=False,  # TNC format (time-first)
            norm_first=True,    # Pre-LN for better stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training and x.size(0) > self.attn_chunk_size:
            # Chunk CNN with overlap to avoid boundary artifacts.
            # Overlap covers the CNN receptive field (9 frames for 4x 3x3 conv layers).
            overlap = 16
            chunk_list = list(x.split(self.attn_chunk_size, dim=0))
            out_chunks = []
            for i, chunk in enumerate(chunk_list):
                parts = []
                if i > 0:
                    parts.append(chunk_list[i - 1][-overlap:])
                parts.append(chunk)
                if i < len(chunk_list) - 1:
                    parts.append(chunk_list[i + 1][:overlap])
                out = self.cnn(torch.cat(parts, dim=0))
                l = overlap if i > 0 else 0
                r = out.size(0) - overlap if i < len(chunk_list) - 1 else out.size(0)
                out_chunks.append(out[l:r])
            x = torch.cat(out_chunks, dim=0)
        else:
            x = self.cnn(x)          # (T, N, cnn_out_channels)
        x = self.input_proj(x)   # (T, N, model_dim)
        x = self.pos_encoding(x) # (T, N, model_dim)

        if not self.training and x.size(0) > self.attn_chunk_size:
            # Chunk along time for memory-efficient inference on long sessions
            chunks = x.split(self.attn_chunk_size, dim=0)
            x = torch.cat([self.transformer(chunk) for chunk in chunks], dim=0)
        else:
            x = self.transformer(x)

        return x  # (T, N, model_dim)


class ConvRNNEncoder(nn.Module):
    """A custom encoder composing a 1D convolution over time followed by a
    multi-layer bidirectional LSTM.

    Args:
        in_features (int): Input feature size per timestep.
        conv_channels (int): Number of channels for the 1D convolution.
        kernel_size (int): Kernel size for the 1D convolution (same padding applied).
        rnn_hidden_size (int): The hidden size of the LSTM.
        rnn_num_layers (int): The number of recurrent layers in the LSTM.
        dropout (float): Dropout probability applied between LSTM layers. (default: 0.1)
    """

    def __init__(
        self,
        in_features: int,
        conv_channels: int,
        kernel_size: int,
        rnn_hidden_size: int,
        rnn_num_layers: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.conv1d = nn.Conv1d(
            in_channels=in_features,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            padding="same",
        )
        self.batch_norm = nn.BatchNorm1d(conv_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            bidirectional=True,
            dropout=dropout if rnn_num_layers > 1 else 0.0,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.permute(1, 2, 0)  # (N, in_features, T)
        x = self.conv1d(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.permute(2, 0, 1)      # (T, N, conv_channels)
        x, _ = self.lstm(x)          # (T, N, rnn_hidden_size * 2)
        return x


class TransformerEncoder(nn.Module):
    """Standard Transformer encoder with chunked inference to prevent OOM.

    Args:
        num_features (int): Model dimension (d_model).
        num_heads (int): Number of attention heads.
        num_layers (int): Number of encoder layers.
        dim_feedforward (int): FFN inner dimension.
        dropout (float): Dropout probability.
        attn_chunk_size (int): Maximum sequence length per attention chunk
            during inference. Set to 0 to disable. (default: 1000)
    """

    def __init__(
        self,
        num_features: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        attn_chunk_size: int = 1000,
    ) -> None:
        super().__init__()
        self.attn_chunk_size = attn_chunk_size

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_features,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training and self.attn_chunk_size > 0 and x.size(0) > self.attn_chunk_size:
            chunks = x.split(self.attn_chunk_size, dim=0)
            return torch.cat([self.transformer_encoder(c) for c in chunks], dim=0)
        return self.transformer_encoder(x)
