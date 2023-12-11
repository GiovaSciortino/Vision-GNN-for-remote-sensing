import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
import numpy as np
import torch

from torch import Tensor
from torch_scatter import scatter
from typing import Optional

from model.utils import graph_to_image, image_to_graph, compute_edge_index, get_activation_layer, batched_graph_to_graph
from model.pos_embed import get_2d_relative_pos_embed

# ----STANDARD IMPLEMENTATION OF THE GRAPHER MODULE-----

class Grapher(gnn.MessagePassing):
    def __init__(self, in_features: int, heads: int, out_features: int, flow: str = 'target_to_source', **kwargs) -> None:
        super(Grapher, self).__init__(flow=flow, **kwargs)
        self.in_features = in_features
        self.heads = heads
        self.out_features = out_features

        # 2 * in_features because the output is concatenated
        self.w_update = nn.Linear(2 * in_features, heads * out_features)
        self.reset()
        return

    def reset(self) -> None:
        self.w_update.reset_parameters()
        return

    def forward(self, x: Tensor, edge_index: Tensor, y: Optional[Tensor] = None, size=None) -> Tensor:
        if y is None:
            messages = self.propagate(edge_index, x=(x, x), size=size)
        else:
            messages = self.propagate(edge_index, x=(x, y), size=size)
        res = torch.cat([x, messages], dim=-1)
        res = self.w_update(res)
        return res

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return x_j - x_i

    def aggregate(self, inputs: Tensor, index: Tensor) -> Tensor:
        y = scatter(inputs, index, dim=0, reduce='max')
        return y


class GrapherFC(Grapher):
    def __init__(self, in_features: int, heads: int, out_features: int, reconstruct_image: bool = False, k: int = 9, act: str = 'relu', r=2, H=60, W=60, relative_positional_embedding=False) -> None:
        super(GrapherFC, self).__init__(in_features, heads, out_features)
        self.in_features = in_features
        self.reconstruct_image = reconstruct_image
        self.k = k
        self.act = act

        self.H = H
        self.W = W
        self.r = r
        self.relative_positional_embedding = relative_positional_embedding

        self.fc1 = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features)
        )

        self.act_l = get_activation_layer(self.act)

        self.fc2 = nn.Sequential(
            nn.Linear(heads * out_features, in_features),
            nn.BatchNorm1d(in_features)
        )
        self.relative_pos = None
        if self.relative_positional_embedding == True:
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(
                self.in_features, int((self.H*self.W)**0.5)))).unsqueeze(0).unsqueeze(1)  
            relative_pos_tensor = F.interpolate(
                relative_pos_tensor, size=(self.H*self.W, (self.H//self.r) * (self.W//self.r)), mode='bicubic', align_corners=False)
            self.relative_pos = nn.Parameter(
                -relative_pos_tensor.squeeze(1), requires_grad=False)
        return

    def forward(self, x: Tensor, y: Optional[Tensor] = None, size=None) -> Tensor:
        B, _, H, W = x.shape
        x = image_to_graph(x, batched=True)
        # x.shape == B, N, C, being N == H * W (1 patch = 1 node)

        if y is None:
            edge_index = compute_edge_index(x, x, k=self.k, is_batched=True)
        else:
            y = image_to_graph(y, batched=True)
            edge_index = compute_edge_index(
                x, y, self.k, is_batched=True, relative_pos=self.relative_pos)

        # convert from batched graph to graph
        x = batched_graph_to_graph(x)
        y = batched_graph_to_graph(y)

        x_proj = self.fc1(x)
        x_proj = super(GrapherFC, self).forward(x_proj, edge_index, y, size)
        x_proj = self.act_l(x_proj)
        x = self.fc2(x_proj) + x
        if self.reconstruct_image:
            x = graph_to_image(x, B, H, W)
        return x

# ----GAT CONVOLUTIONAL LAYER-----

class GATGrapherFC(gnn.GATConv):
    def __init__(self, in_features: int, heads: int, out_features: int, reconstruct_image: bool = False, k: int = 9, act: str = 'relu', r=2, H=60, W=60, relative_positional_embedding=False, flow: str = 'target_to_source', **kwargs) -> None:
        super(GATGrapherFC, self).__init__(in_channels=in_features,
                                           out_channels=out_features, heads=heads, flow=flow, **kwargs)
        self.in_features = in_features
        self.reconstruct_image = reconstruct_image
        self.k = k
        self.act = act

        self.H = H
        self.W = W
        self.r = r
        self.relative_positional_embedding = relative_positional_embedding

        self.fc1 = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features)
        )

        self.act_l = get_activation_layer(self.act)

        self.fc2 = nn.Sequential(
            nn.Linear(heads * out_features, in_features),
            nn.BatchNorm1d(in_features)
        )
        self.relative_pos = None
        if self.relative_positional_embedding == True:
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(
                self.in_features, int((self.H*self.W)**0.5)))).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(
                relative_pos_tensor, size=(self.H*self.W, (self.H//self.r) * (self.W//self.r)), mode='bicubic', align_corners=False)
            self.relative_pos = nn.Parameter(
                -relative_pos_tensor.squeeze(1), requires_grad=False)
        return

    def forward(self, x: Tensor, y: Optional[Tensor] = None, size=None) -> Tensor:
        B, _, H, W = x.shape
        x = image_to_graph(x, batched=True)
        # x.shape == B, N, C, being N == H * W (1 patch = 1 node)

        if y is None:
            edge_index = compute_edge_index(x, x, k=self.k, is_batched=True)
        else:
            y = image_to_graph(y, batched=True)
            edge_index = compute_edge_index(
                x, y, self.k, is_batched=True, relative_pos=self.relative_pos)

        # convert from batched graph to graph
        x = batched_graph_to_graph(x)
        y = batched_graph_to_graph(y)

        x_proj = self.fc1(x)
        x_proj = super(GATGrapherFC, self).forward(
            (x_proj, y), edge_index, size)
        x_proj = self.act_l(x_proj)
        x = self.fc2(x_proj) + x
        if self.reconstruct_image:
            x = graph_to_image(x, B, H, W)
        return x


# ----GCN CONVOLUTIONAL LAYER-----

class GCNGrapherFC(gnn.GCNConv):
    def __init__(self, in_features: int, heads: int, out_features: int, reconstruct_image: bool = False, k: int = 4, act: str = 'relu', r=2, H=60, W=60, relative_positional_embedding=False, flow: str = 'target_to_source', **kwargs) -> None:
        super(GCNGrapherFC, self).__init__(in_channels=in_features,
                                           out_channels=(heads*out_features), **kwargs)
        self.in_features = in_features
        self.reconstruct_image = reconstruct_image
        self.k = k
        self.act = act

        self.H = H
        self.W = W
        self.r = r
        self.relative_positional_embedding = relative_positional_embedding

        self.fc1 = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features)
        )

        self.act_l = get_activation_layer(self.act)

        self.fc2 = nn.Sequential(
            nn.Linear(heads * out_features, in_features),
            nn.BatchNorm1d(in_features)
        )
        self.relative_pos = None
        if self.relative_positional_embedding == True:
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(
                self.in_features, int((self.H*self.W)**0.5)))).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(
                relative_pos_tensor, size=(self.H*self.W, (self.H//self.r) * (self.W//self.r)), mode='bicubic', align_corners=False)
            self.relative_pos = nn.Parameter(
                -relative_pos_tensor.squeeze(1), requires_grad=False)
        return

    def forward(self, x: Tensor, y: Optional[Tensor] = None, size=None) -> Tensor:
        B, _, H, W = x.shape
        y = None
        x = image_to_graph(x, batched=True)
        # x.shape == B, N, C, being N == H * W (1 patch = 1 node)

        if y is None:
            edge_index = compute_edge_index(x, x, k=self.k, is_batched=True)
        else:
            y = image_to_graph(y, batched=True)
            edge_index = compute_edge_index(
                x, y, self.k, is_batched=True, relative_pos=self.relative_pos)

        # convert from batched graph to graph
        x = batched_graph_to_graph(x)

        x_proj = self.fc1(x)
        x_proj = super(GCNGrapherFC, self).forward(x_proj, edge_index, size)
        x_proj = self.act_l(x_proj)
        x = self.fc2(x_proj) + x
        if self.reconstruct_image:
            x = graph_to_image(x, B, H, W)
        return x
