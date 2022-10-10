# The based unit of graph convolutional networks.

import torch
import torch.nn as nn

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# class ConvTemporalGraphical(nn.Module):
#
#     r"""The basic module for applying a graph convolution.
#
#     Args:
#         in_channels (int): Number of channels in the input sequence data
#         out_channels (int): Number of channels produced by the convolution
#         kernel_size (int): Size of the graph convolving kernel
#         t_kernel_size (int): Size of the temporal convolving kernel
#         t_stride (int, optional): Stride of the temporal convolution. Default: 1
#         t_padding (int, optional): Temporal zero-padding added to both sides of
#             the input. Default: 0
#         t_dilation (int, optional): Spacing between temporal kernel elements.
#             Default: 1
#         bias (bool, optional): If ``True``, adds a learnable bias to the output.
#             Default: ``True``
#
#     Shape:
#         - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
#         - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
#         - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
#         - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
#
#         where
#             :math:`N` is a batch size,
#             :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
#             :math:`T_{in}/T_{out}` is a length of input/output sequence,
#             :math:`V` is the number of graph nodes.
#     """
#
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size,
#                  t_kernel_size=1,
#                  t_stride=1,
#                  t_padding=0,
#                  t_dilation=1,
#                  bias=True):
#         super().__init__()
#
#         self.kernel_size = kernel_size
#         self.conv = nn.Conv2d(
#             in_channels,
#             out_channels * kernel_size,
#             kernel_size=(t_kernel_size, 1),
#             padding=(t_padding, 0),
#             stride=(t_stride, 1),
#             dilation=(t_dilation, 1),
#             bias=bias)
#
#     def forward(self, x, A):
#         assert A.size(0) == self.kernel_size
#         x = self.conv(x) # 256, 48, 50, 25
#         n, kc, t, v = x.size()
#         x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
#         x = torch.einsum('nkctv,kvw->nctw', (x, A))
#
#         return x.contiguous(), A


class ConvTemporalGraphical(nn.Module):

    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        # assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        # import pdb; pdb.set_trace()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v) # 256, 6, 3, 50, 25
        x = torch.einsum('nkctv,kvw->nctw', (x, A)) # 256, 16, 50, 25
        return x.contiguous(), A


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class DGCNN(nn.Module):
    def __init__(self, in_channels,
                    out_channels):
        super(DGCNN, self).__init__()
        self.k = 3

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels*2, out_channels, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (9, 1),
                (1, 1),
                'same',
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.5, inplace=True),
        )

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        # import pdb; pdb.set_trace()
        x1 = self.tcn(x1.unsqueeze(3))
        # N, C, T, _ = x1.shape
        x1 = x1.squeeze()
        return x1
