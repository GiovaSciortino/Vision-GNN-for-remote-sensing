import torch
import torch.nn as nn

from torch import functional as F
from torch import Tensor
from model.utils import get_activation_layer

from typing import Union, List, Dict, Tuple, Optional

class ViGDecoder(nn.Module):
    def __init__(self, in_channels: int, channels: int, n_classes: int, act: str='relu', dropout_p: float=0.5) -> None:
        super(ViGDecoder, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.n_classes = n_classes
        self.act = act
        self.dropout_p = dropout_p

        self.prediction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(channels),
            nn.Dropout(dropout_p),
            nn.Conv2d(channels, n_classes, kernel_size=1, stride=1)
        )

        return
    
    def forward(self, x: Tensor) -> Tensor:
        y = self.prediction(x)
        return y