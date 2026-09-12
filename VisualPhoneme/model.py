"""Small frame encoder and temporal recognizer for visual phoneme CTC."""
from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
        )


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2 * dilation,
                      dilation=dilation, groups=channels, bias=False),
            nn.BatchNorm1d(channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.layers(inputs))


def frame_encoder(width: int) -> nn.Sequential:
    return nn.Sequential(
        ConvBlock(1, 16),
        ConvBlock(16, 32),
        ConvBlock(32, 64),
        ConvBlock(64, width),
        nn.AdaptiveAvgPool2d(1),
    )


def temporal_encoder(width: int) -> nn.Sequential:
    return nn.Sequential(*(TemporalBlock(width, dilation) for dilation in (1, 2, 4, 8)))


def landmark_encoder(landmark_points: int, coordinate_dimensions: int, width: int,
                     bottleneck: int | None = None) -> nn.Sequential:
    inputs = landmark_points * coordinate_dimensions + 1
    if bottleneck:
        return nn.Sequential(
            nn.Linear(inputs, bottleneck), nn.LayerNorm(bottleneck), nn.SiLU(inplace=True),
            nn.Linear(bottleneck, width), nn.LayerNorm(width), nn.SiLU(inplace=True),
        )
    return nn.Sequential(nn.Linear(inputs, width), nn.LayerNorm(width), nn.SiLU(inplace=True))


class CompactVisualPhoneme(nn.Module):
    """Encode grayscale face frames and preserve one CTC step per video frame."""

    def __init__(self, classes: int = 40, width: int = 96):
        super().__init__()
        self.frame_encoder = frame_encoder(width)
        self.temporal = temporal_encoder(width)
        self.classifier = nn.Linear(width, classes)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Return time-major logits from video shaped ``[batch,time,1,h,w]``."""
        if video.ndim != 5 or video.shape[2] != 1:
            raise ValueError("video must have shape [batch,time,1,height,width]")
        batch, steps, channels, height, width = video.shape
        encoded = self.frame_encoder(video.reshape(batch * steps, channels, height, width))
        encoded = encoded.flatten(1).reshape(batch, steps, -1).transpose(1, 2)
        return self.classifier(self.temporal(encoded).transpose(1, 2)).transpose(0, 1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class CompactLandmarkPhoneme(nn.Module):
    """Recognize phonemes from VPA geometry without an image encoder."""

    def __init__(self, classes: int = 40, width: int = 96, landmark_points: int = 41,
                 coordinate_dimensions: int = 2, landmark_bottleneck: int | None = None):
        super().__init__()
        self.landmark_points = landmark_points
        self.coordinate_dimensions = coordinate_dimensions
        self.landmark_encoder = landmark_encoder(landmark_points, coordinate_dimensions,
                                                 width, landmark_bottleneck)
        self.temporal = temporal_encoder(width)
        self.classifier = nn.Linear(width, classes)

    def forward(self, landmarks: torch.Tensor, landmark_mask: torch.Tensor) -> torch.Tensor:
        expected = (self.landmark_points, self.coordinate_dimensions)
        if landmarks.ndim != 4 or landmarks.shape[2:] != expected:
            raise ValueError(f"landmarks must have trailing shape {expected}")
        if landmark_mask.shape != landmarks.shape[:2]:
            raise ValueError("landmark_mask must have shape [batch,time]")
        batch, steps = landmarks.shape[:2]
        visibility = landmark_mask.reshape(batch * steps, 1).to(landmarks.dtype)
        coordinates = torch.nan_to_num(landmarks).reshape(batch * steps, -1)
        encoded = self.landmark_encoder(torch.cat((coordinates, visibility), dim=1)) * visibility
        encoded = encoded.reshape(batch, steps, -1).transpose(1, 2)
        return self.classifier(self.temporal(encoded).transpose(1, 2)).transpose(0, 1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class CompactFusionVisualPhoneme(nn.Module):
    """Fuse mouth pixels with normalized VPA coordinates before temporal modeling."""

    def __init__(self, classes: int = 40, width: int = 96, landmark_points: int = 41,
                 coordinate_dimensions: int = 2):
        super().__init__()
        self.landmark_points = landmark_points
        self.coordinate_dimensions = coordinate_dimensions
        self.frame_encoder = frame_encoder(width)
        self.landmark_encoder = nn.Sequential(
            nn.Linear(landmark_points * coordinate_dimensions + 1, 32),
            nn.LayerNorm(32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 32),
            nn.SiLU(inplace=True),
        )
        self.fusion = nn.Sequential(nn.Linear(width + 32, width), nn.LayerNorm(width), nn.SiLU(inplace=True))
        self.temporal = temporal_encoder(width)
        self.classifier = nn.Linear(width, classes)

    def forward(self, video: torch.Tensor, landmarks: torch.Tensor,
                landmark_mask: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5 or video.shape[2] != 1:
            raise ValueError("video must have shape [batch,time,1,height,width]")
        expected = (self.landmark_points, self.coordinate_dimensions)
        if landmarks.shape[:2] != video.shape[:2] or landmarks.shape[2:] != expected:
            raise ValueError(f"landmarks must have trailing shape {expected}")
        if landmark_mask.shape != video.shape[:2]:
            raise ValueError("landmark_mask must have shape [batch,time]")
        batch, steps, channels, height, width = video.shape
        images = self.frame_encoder(video.reshape(batch * steps, channels, height, width)).flatten(1)
        visibility = landmark_mask.reshape(batch * steps, 1).to(landmarks.dtype)
        coordinates = torch.nan_to_num(landmarks).reshape(batch * steps, -1)
        geometry = self.landmark_encoder(torch.cat((coordinates, visibility), dim=1)) * visibility
        encoded = self.fusion(torch.cat((images, geometry), dim=1)).reshape(batch, steps, -1).transpose(1, 2)
        return self.classifier(self.temporal(encoded).transpose(1, 2)).transpose(0, 1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class CompactGatedFusionVisualPhoneme(nn.Module):
    """Coordinate-first fusion with a conservatively initialized image residual."""

    def __init__(self, classes: int = 40, width: int = 96, landmark_points: int = 41,
                 coordinate_dimensions: int = 2, landmark_bottleneck: int | None = 32):
        super().__init__()
        self.landmark_points = landmark_points
        self.coordinate_dimensions = coordinate_dimensions
        self.frame_encoder = frame_encoder(width)
        self.landmark_encoder = landmark_encoder(landmark_points, coordinate_dimensions,
                                                 width, landmark_bottleneck)
        self.image_projection = nn.Sequential(nn.Linear(width, width), nn.LayerNorm(width),
                                              nn.SiLU(inplace=True))
        self.image_gate_logit = nn.Parameter(torch.tensor(-6.0))
        self.temporal = temporal_encoder(width)
        self.classifier = nn.Linear(width, classes)

    def forward(self, video: torch.Tensor, landmarks: torch.Tensor,
                landmark_mask: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5 or video.shape[2] != 1:
            raise ValueError("video must have shape [batch,time,1,height,width]")
        expected = (self.landmark_points, self.coordinate_dimensions)
        if landmarks.shape[:2] != video.shape[:2] or landmarks.shape[2:] != expected:
            raise ValueError(f"landmarks must have trailing shape {expected}")
        if landmark_mask.shape != video.shape[:2]:
            raise ValueError("landmark_mask must have shape [batch,time]")
        batch, steps, channels, height, width = video.shape
        images = self.frame_encoder(video.reshape(batch * steps, channels, height, width)).flatten(1)
        images = self.image_projection(images)
        visibility = landmark_mask.reshape(batch * steps, 1).to(landmarks.dtype)
        coordinates = torch.nan_to_num(landmarks).reshape(batch * steps, -1)
        geometry = self.landmark_encoder(torch.cat((coordinates, visibility), dim=1)) * visibility
        gate = self.image_gate_logit.sigmoid()
        encoded = geometry + gate * images
        encoded = encoded.reshape(batch, steps, -1).transpose(1, 2)
        return self.classifier(self.temporal(encoded).transpose(1, 2)).transpose(0, 1)

    @property
    def image_gate(self) -> float:
        return float(self.image_gate_logit.detach().sigmoid())

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
