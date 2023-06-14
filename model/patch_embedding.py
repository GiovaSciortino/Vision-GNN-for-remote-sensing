import torch.nn as nn
import math

from torch import Tensor
from typing import List, Union
from model.utils import get_activation_layer


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int, in_channels: int, out_channels: int, reshape: bool=True) -> None:
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reshape = reshape
        
        # non-overlapped patch embeddings
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(out_channels)
        )

        return

    def forward(self, x: Tensor) -> Tensor:
        # x.shape is B, C, H, W
        x = self.encoder(x) # B, C, #Patches in height, #Patches in width
        if self.reshape:
            x = x.flatten(2).transpose(1, 2) # B, #Nodes, #Features (= out_channels)
        return x

class PatchEmbeddingV2(nn.Module):
    def __init__(self,
                reduce_factor: int,
                in_channels: int,
                out_channels: Union[int, List[int]],
                overlapped: bool=True,
                act: str='relu',
                reshape: bool=True) -> None:
        super(PatchEmbeddingV2, self).__init__()
        
        assert math.log(reduce_factor, 2).is_integer()
        
        n_steps = int(math.log(reduce_factor, 2))
        self.reduce_factor = reduce_factor
        self.in_channels = in_channels
        self.overlapped = overlapped
        self.act = act
        self.reshape = reshape
        if isinstance(out_channels, int):
            out_channels = [out_channels for _ in range(n_steps)]
        else:
            assert len(out_channels) == n_steps, f"Invalid length for out_channels ({len(out_channels)}) with {n_steps} of reduction performed"
        self.out_channels = out_channels

        features = [in_channels] + out_channels
        if overlapped:
            cnn_params = {
                'kernel_size': 3,
                'stride': 2,
                'padding': 1
            }
        else:
            cnn_params = {
                'kernel_size': 2,
                'stride': 2
            }
        encoder_block = []
        for idx in range(len(features) - 1):
            in_ch, out_ch = features[idx], features[idx + 1]
            encoder_block.extend([
                nn.Conv2d(in_ch, out_ch, **cnn_params),
                nn.BatchNorm2d(out_ch)
            ])
            if idx < len(features) - 1:
                encoder_block.append(get_activation_layer(act))

        self.encoder = nn.Sequential(*encoder_block)
        return

    def forward(self, x: Tensor) -> Tensor:
        # x.shape == B, C, H, W
        x = self.encoder(x) # B, C, #Patches in height, #Patches in width
        if self.reshape:
            x = x.flatten(2).transpose(2, 1) # B, N, F (n = number of nodes (i.e. number of patches), f = number of features)
        return x