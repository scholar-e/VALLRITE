"""Dependency-free port of the published Auto-AVSR visual conformer encoder.

Module and parameter names follow the ESPnet state dict in
`LRS3_V_WER19.1/model.pth`, so `load_autoavsr_encoder` loads it strictly.
Configuration from its `model.json`: 12 macaron conformer blocks, width 768,
12 heads, ReLU feed-forward 3072, swish convolution kernel 31 and "latest"
relative positional attention. Unlike ESPnet, padded frames are
masked in attention and zeroed before the depthwise convolution, so batched
and single-clip outputs agree.
"""
import math
from pathlib import Path

import torch
from torch import nn

from VisualPhoneme.model import AutoAvsrVisualFrontend

# Applied after centre-cropping 96x96 mouth crops to 88x88 and scaling to [0, 1].
AUTOAVSR_MEAN = 0.421
AUTOAVSR_STD = 0.165
AUTOAVSR_CROP = 88


def autoavsr_video_transform(frames: torch.Tensor) -> torch.Tensor:
    """Map uint8 [time,96,96] grayscale mouth crops to normalized [time,1,88,88]."""
    if frames.ndim != 3 or frames.dtype != torch.uint8:
        raise ValueError("frames must be uint8 [time,height,width]")
    height, width = frames.shape[1:]
    top, left = (height - AUTOAVSR_CROP) // 2, (width - AUTOAVSR_CROP) // 2
    cropped = frames[:, top:top + AUTOAVSR_CROP, left:left + AUTOAVSR_CROP]
    return ((cropped.float() / 255 - AUTOAVSR_MEAN) / AUTOAVSR_STD).unsqueeze(1)


class RelPositionalEncoding(nn.Module):
    """ESPnet "latest" relative encoding: scales inputs, returns 2T-1 positions."""

    def __init__(self, width: int):
        super().__init__()
        self.width = width
        self.scale = math.sqrt(width)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        steps = inputs.shape[1]
        position = torch.arange(steps, device=inputs.device, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, self.width, 2, device=inputs.device,
                                           dtype=torch.float32)
                              * -(math.log(10000.0) / self.width))
        positive = torch.zeros(steps, self.width, device=inputs.device)
        negative = torch.zeros(steps, self.width, device=inputs.device)
        positive[:, 0::2] = torch.sin(position * frequency)
        positive[:, 1::2] = torch.cos(position * frequency)
        negative[:, 0::2] = torch.sin(-position * frequency)
        negative[:, 1::2] = torch.cos(-position * frequency)
        # Relative offsets T-1 ... 0 ... -(T-1).
        encoding = torch.cat((positive.flip(0), negative[1:]), dim=0).unsqueeze(0)
        return inputs * self.scale, encoding.to(inputs.dtype)


class RelPositionMultiHeadedAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.head_width = heads, width // heads
        self.linear_q = nn.Linear(width, width)
        self.linear_k = nn.Linear(width, width)
        self.linear_v = nn.Linear(width, width)
        self.linear_out = nn.Linear(width, width)
        self.linear_pos = nn.Linear(width, width, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(heads, self.head_width))
        self.pos_bias_v = nn.Parameter(torch.zeros(heads, self.head_width))
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def rel_shift(scores: torch.Tensor) -> torch.Tensor:
        batch, heads, steps, positions = scores.shape
        padded = torch.cat((scores.new_zeros(batch, heads, steps, 1), scores), dim=-1)
        padded = padded.view(batch, heads, positions + 1, steps)
        return padded[:, :, 1:].view_as(scores)[:, :, :, :positions // 2 + 1]

    def forward(self, inputs: torch.Tensor, positions: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = inputs.shape
        split = lambda x: x.view(x.shape[0], -1, self.heads, self.head_width)
        query = split(self.linear_q(inputs))
        key = split(self.linear_k(inputs)).transpose(1, 2)
        value = split(self.linear_v(inputs)).transpose(1, 2)
        position = split(self.linear_pos(positions)).transpose(1, 2)
        content = (query + self.pos_bias_u).transpose(1, 2) @ key.transpose(-2, -1)
        relative = self.rel_shift(
            (query + self.pos_bias_v).transpose(1, 2) @ position.transpose(-2, -1)
        )
        scores = (content + relative) / math.sqrt(self.head_width)
        blocked = ~valid[:, None, None, :]
        weights = scores.masked_fill(blocked, torch.finfo(scores.dtype).min).softmax(-1)
        weights = self.dropout(weights.masked_fill(blocked, 0.0))
        context = (weights @ value).transpose(1, 2).reshape(batch, steps, -1)
        return self.linear_out(context)


class FeedForward(nn.Module):
    def __init__(self, width: int, hidden: int, dropout: float):
        super().__init__()
        self.w_1 = nn.Linear(width, hidden)
        self.w_2 = nn.Linear(hidden, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # ESPnet's PositionwiseFeedForward uses ReLU; only the frontend and
        # convolution module use swish.
        return self.w_2(self.dropout(torch.relu(self.w_1(inputs))))


class ConvolutionModule(nn.Module):
    def __init__(self, width: int, kernel: int):
        super().__init__()
        # "cov" spelling matches the published state dict.
        self.pointwise_cov1 = nn.Conv1d(width, 2 * width, 1)
        self.depthwise_conv = nn.Conv1d(width, width, kernel, padding=(kernel - 1) // 2,
                                        groups=width)
        self.norm = nn.BatchNorm1d(width)
        self.pointwise_cov2 = nn.Conv1d(width, width, 1)

    def forward(self, inputs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        outputs = nn.functional.glu(self.pointwise_cov1(inputs.transpose(1, 2)), dim=1)
        outputs = outputs * valid[:, None, :].to(outputs.dtype)
        outputs = nn.functional.silu(self.norm(self.depthwise_conv(outputs)))
        return self.pointwise_cov2(outputs).transpose(1, 2)


class ConformerLayer(nn.Module):
    def __init__(self, width: int, heads: int, hidden: int, kernel: int, dropout: float):
        super().__init__()
        self.self_attn = RelPositionMultiHeadedAttention(width, heads, dropout)
        self.feed_forward = FeedForward(width, hidden, dropout)
        self.feed_forward_macaron = FeedForward(width, hidden, dropout)
        self.conv_module = ConvolutionModule(width, kernel)
        for name in ("norm_ff", "norm_mha", "norm_ff_macaron", "norm_conv", "norm_final"):
            setattr(self, name, nn.LayerNorm(width, eps=1e-12))
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, positions: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        x = inputs + 0.5 * self.dropout(self.feed_forward_macaron(self.norm_ff_macaron(inputs)))
        x = x + self.dropout(self.self_attn(self.norm_mha(x), positions, valid))
        x = x + self.dropout(self.conv_module(self.norm_conv(x), valid))
        x = x + 0.5 * self.dropout(self.feed_forward(self.norm_ff(x)))
        return self.norm_final(x)


class AutoAvsrEncoder(nn.Module):
    """Conv3D/ResNet-18 frontend plus conformer; returns [batch,time,768] features."""

    def __init__(self, width: int = 768, heads: int = 12, hidden: int = 3072,
                 layers: int = 12, kernel: int = 31, dropout: float = 0.1):
        super().__init__()
        self.frontend = AutoAvsrVisualFrontend()
        self.embed = nn.Sequential(nn.Linear(512, width), RelPositionalEncoding(width))
        self.encoders = nn.ModuleList(
            ConformerLayer(width, heads, hidden, kernel, dropout) for _ in range(layers)
        )
        self.after_norm = nn.LayerNorm(width, eps=1e-12)
        self.width = width

    def forward(self, video: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5 or video.shape[2] != 1:
            raise ValueError("video must have shape [batch,time,1,height,width]")
        valid = torch.arange(video.shape[1], device=video.device)[None, :] < lengths[:, None]
        # Zero padded frames so the 3D stem's temporal kernel cannot read them.
        video = video * valid[:, :, None, None, None].to(video.dtype)
        x, positions = self.embed[1](self.embed[0](self.frontend(video)))
        for layer in self.encoders:
            x = layer(x, positions, valid)
        return self.after_norm(x)


class AutoAvsrCtc(nn.Module):
    """Published encoder plus its 5,049-way unigram CTC head (index 0 is blank)."""

    def __init__(self, vocabulary: int = 5049):
        super().__init__()
        self.encoder = AutoAvsrEncoder()
        self.ctc_lo = nn.Linear(self.encoder.width, vocabulary)

    def forward(self, video: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.ctc_lo(self.encoder(video, lengths)).log_softmax(-1)


def load_autoavsr_ctc(checkpoint: Path) -> AutoAvsrCtc:
    """Strictly load encoder and CTC weights; the attention decoder is skipped."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    selected = {key: value for key, value in state.items() if key.startswith("encoder.")}
    selected.update({key.removeprefix("ctc."): value for key, value in state.items()
                     if key.startswith("ctc.")})
    model = AutoAvsrCtc(state["ctc.ctc_lo.weight"].shape[0])
    model.load_state_dict(selected, strict=True)
    return model


def load_units(path: Path) -> list[str]:
    """Return CTC labels: blank, the numbered unigram units, then <eos>."""
    units = ["<blank>"]
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        token, index = line.rsplit(" ", 1)
        if int(index) != number:
            raise ValueError(f"unexpected unit numbering at line {number}")
        units.append(token)
    return units + ["<eos>"]


def ctc_greedy_text(log_probabilities: torch.Tensor, units: list[str]) -> str:
    """Collapse repeats, drop blank/<eos>, and join sentencepiece units."""
    pieces, previous = [], None
    for index in log_probabilities.argmax(-1).tolist():
        if index != previous and index != 0 and index != len(units) - 1:
            pieces.append(units[index])
        previous = index
    return "".join(pieces).replace("▁", " ").strip()
