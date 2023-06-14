import torch
import torch.nn as nn

from torch import Tensor

from typing import Optional


def get_activation_layer(act: str, **kwargs):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'gelu':
        return nn.GELU(**kwargs)
    elif act == 'sigmoid':
        return nn.Sigmoid()
    elif act == 'softmax':
        return nn.Softmax(**kwargs)
    elif act == 'leakyrelu':
        return nn.LeakyReLU(**kwargs)
    else:
        raise NotImplementedError(
            f'Activation layer not implemented yet {act}')


def convert_neigh_list_to_edge_index(nn_idx: Tensor, is_batched: bool = True) -> Tensor:
    # nn_idx is of shape B, N, K is if_batched == True, being k the number of neighbors
    # the output should be of shape 2, |E|
    if is_batched:
        B, N, K = nn_idx.shape
    else:
        N, K = nn_idx.shape
    with torch.no_grad():
        if is_batched:
            # compute an index offset to be added to every entry
            idx_offset = (torch.arange(B).to(
                nn_idx.device) * N).reshape(B, 1, 1)
            idx_offset = idx_offset.repeat(1, N, K)

            center_idx = torch.arange(N).to(nn_idx.device).reshape(1, N, 1)
            center_idx = center_idx.repeat(B, 1, K)

            # add offset to both the central index and the neighbor index
            center_idx += idx_offset
            nn_idx += idx_offset

            edge_index = torch.stack([center_idx, nn_idx], dim=1).to(
                nn_idx.device)  # B, 2, N, K
            edge_index = edge_index.flatten(2)  # B, 2, |E| per batch
            edge_index = edge_index.transpose(
                0, 1).transpose(1, 2)  # 2, |E| per batch, B
            edge_index = edge_index.flatten(1).contiguous()  # 2, |E|
        else:
            center_idx = torch.arange(N).to(
                nn_idx.device).unsqueeze(-1).repeat(1, K)
            edge_index = torch.stack([center_idx, nn_idx], dim=0).to(
                nn_idx.device)  # 2, N, K
            edge_index = edge_index.flatten(1).contiguous()  # 2, |E|

    return edge_index


def compute_edge_index(x: Tensor, y: Tensor, k: int, distance: str = 'pointcloud', is_batched: bool = True, relative_pos=None) -> Tensor:
    assert distance in {'pointcloud',
                        'euclidean'}, f'Distance metric invalid ({distance})'

    # TODO add relative positional encoding: DONE
    with torch.no_grad():
        if distance == 'pointcloud':
            dist = pointcloud_dist(x, y, is_batched)
        elif distance == 'euclidean':
            dist = euclidean_dist(x, y)
        else:
            raise NotImplementedError(
                f"Distance {distance} not implemented yet")

        if relative_pos is not None:
            relative_pos = torch.squeeze(relative_pos, 0).contiguous()
            dist += relative_pos
        _, nn_idx = torch.topk(-dist, k=k)
        edge_index = convert_neigh_list_to_edge_index(nn_idx, is_batched)
    return edge_index


def pointcloud_dist(x: Tensor, y: Tensor, is_batched: bool = True) -> Tensor:
    # This function comes from
    # https://github.com/huawei-noah/Efficient-AI-Backbones/blob/860e89a0fdb45f55510fa5c3a5580a9d3afd69eb/vig_pytorch/gcn_lib/torch_edge.py#L39
    # All credits to the authors of the ViG Paper "Vision GNN: An Image is Worth Graph of Nodes"
    # https://arxiv.org/abs/2206.00272
    features_index, node_index = (2, 1) if is_batched else (1, 0)
    with torch.no_grad():
        xy_inner = torch.matmul(x, y.transpose(features_index, node_index))
        x_square = (x * x).sum(dim=-1, keepdim=True)
        y_square = (y * y).sum(dim=-1,
                               keepdim=True).transpose(features_index, node_index)
        return x_square - 2 * xy_inner + y_square


def euclidean_dist(x: Tensor, y: Optional[Tensor]) -> Tensor:
    with torch.no_grad():
        return torch.cdist(x, y, p=2)


def image_to_graph(x: Tensor) -> Tensor:
    # B, C, H, W -> B, C, N -> NB, C
    x = x.flatten(2).permute(0, 2, 1).reshape(-1, x.shape[1]).contiguous()
    return x


def graph_to_image(x: Tensor, b: int, h: int, w: int) -> Tensor:
    c = x.shape[1]
    # NB, C -> B, N, C -> B, C, N -> B, C, H, W
    x = x.reshape(b, -1, c) \
        .transpose(1, 2) \
        .reshape(b, -1, h, w).contiguous()
    return x
