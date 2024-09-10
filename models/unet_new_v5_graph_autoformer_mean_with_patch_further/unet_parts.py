""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcn_lib import GINConv2d,DenseDilatedKnnGraph,DenseDilatedKnnGraph,batched_index_select# `batched_index_select` is a
# function that is used to
# select elements from the
# input tensor based on the
# indices provided. It is
# typically used in scenarios
# where you have a batch of
# tensors and you want to
# select specific elements
# from each tensor in the
# batch based on the indices
# provided.

import torch
import torch.nn.functional as F
import torch.nn as nn
from .autoformer_mean import FourierCrossAttention
from typing import Optional, Callable, List, Any

import numpy as np
import torch
import torchvision.ops
from einops import rearrange
from torch import nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import trunc_normal_

def window_partition_3d(x, window_size):
    """
    Args:
        x: (B, L, W, H)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, W, H)
    """
    B, L, W, H = x.shape
    #print(B, L, W, H,"B, L, W, H")
    x = x.contiguous().view(B*window_size, L // window_size, W, H)
    #windows = x.permute(0, 1, 2, 3).contiguous().view(-1, window_size, C)
    return x

def window_reverse_3d(windows, window_size):
    """
    Args:
        windows: (num_windows*B, L//window_size, W, C)
        window_size (int): Window size
        L (int): Sequence length

    Returns:
        x: (B, L, C)
    """
    B, L, W, H = windows.shape
    B = int(windows.shape[0] //window_size) # L / window_size should be   2   
    
    x = windows.contiguous().view(B, L * window_size, W, H)
    x = x.permute(0, 1, 2, 3).contiguous().view(B, -1, W, H)
    return x

class Graph_weight(nn.Module):
    def __init__(self,in_channels=128,out_channels=128):
        super(Graph_weight, self).__init__()
        self.knn_step1 = DenseDilatedKnnGraph(1, 1, stochastic=False, epsilon=0.0)
        #self.knn_conv = nn.Conv2d(in_channels,in_channels,1,1,0)
        self.knn_step2 = batched_index_select
        #self.linear1 = nn.Conv2d(in_channels,in_channels,1,1,0)

        self.conv_seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, bias=False),
            )
        
        #self.frequency = FourierCrossAttention(in_channels,in_channels,1,1)
        #in_channels, out_channels

        #eps_init = 0.0
        #self.eps = nn.Parameter(torch.Tensor([eps_init]))
        #(a2,knn[0])

    def forward(self, x):
        x_mean = (F.adaptive_avg_pool2d(x, output_size=1))#.transpose(1,2)

        #print("x_mean",x_mean.shape)
        x1 = self.knn_step1(x_mean)
        
        x2 = self.knn_step2(x_mean,x1[0])
        
        x3 = torch.sum(x2, -1, keepdim=True).transpose(1,2)

        x3 = F.sigmoid(x3)


        #x3 = self.frequency(x_mean,x_mean,x_mean)[0].transpose(1,3)

        #print(x3.shape,x.shape,"x 进行乘法 -=-=-=-==")
        out = self.conv_seq((x3.transpose(1,2)) * x + x)  #from v1 to residual
        return out
    
class Graph_weight_space(nn.Module):
    def __init__(self,in_channels=128,out_channels=128):
        super(Graph_weight_space, self).__init__()
        self.knn_step1 = DenseDilatedKnnGraph(5, 1, stochastic=False, epsilon=0.0)
        #self.knn_conv = nn.Conv2d(in_channels,in_channels,1,1,0)
        self.knn_step2 = batched_index_select
        #self.linear1 = nn.Conv2d(in_channels,in_channels,1,1,0)

        self.conv_seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, bias=False)
            )
        #in_channels, out_channels

        #eps_init = 0.0
        #self.eps = nn.Parameter(torch.Tensor([eps_init]))
        #(a2,knn[0])

    def forward(self, x):
        x_mean = (F.adaptive_avg_pool2d(x.transpose(1,2), output_size=1)).transpose(1,2)
        x_mean += (F.adaptive_avg_pool2d(x.transpose(1,3), output_size=1)).transpose(1,2)
        #print(x_mean.shape,"x_mean")
        x1 = self.knn_step1(x_mean/2)
        
        x2 = self.knn_step2(x_mean,x1[0])
        
        x3 = torch.sum(x2, -1, keepdim=True).transpose(1,2)
        #print(x3.shape,"x3x3x3x3") #bs 128,1,1,
        #x3 = self.linear1(x3)
        #out = (1 + self.eps) * x + x3
        x3 = F.sigmoid(x3)
        #print(x3.shape,x.shape,"x 进行乘法 -=-=-=-==")
        out = self.conv_seq((x3.transpose(1,2)) * x + x)  #from v1 to residual
        return out

class Temporal_Space(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        self.space1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False,groups=in_channels),
            #nn.LayerNorm()
            #nn.ReLU6()
            )
        self.temporal1 = nn.Linear(in_channels,out_channels)

        #self.temporal2 = nn.Linear(out_channels,2*out_channels)
        #self.temporal2 = nn.Linear(int(0.5*out_channels),out_channels)

        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.space1(x)
        x = x.permute(0, 2, 3, 1)
        
        x = self.temporal1(x)
        #x = self.temporal2(x)
        x = self.relu(x)
        x = x.permute(0, 3, 1, 2)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        #window_partition_3d   
        #window_reverse_3d
        self.temporal_space1 = Temporal_Space(in_channels,mid_channels)
        self.graph_weight =  Graph_weight(int(mid_channels//16),int(mid_channels//16))
        self.graph_weight_space =  Graph_weight_space(int(mid_channels//16),int(out_channels//16))
        self.temporal_space2 =  Temporal_Space(out_channels,out_channels)

        # self.double_conv = nn.Sequential(
    
        #     Temporal_Space(in_channels,mid_channels),

        #     Graph_weight(mid_channels,mid_channels),
        .0

        #     #Temporal_Space(mid_channels,out_channels),
            
        #     #Temporal_Space(out_channels,out_channels),
            
        #     Graph_weight_space(mid_channels,out_channels),
            
        #     Temporal_Space(out_channels,out_channels),
        # )

    def forward(self, x):
        #print("use new unet")
        x1 = self.temporal_space1(x)

        x1 = window_partition_3d(x1,16)

        x2 = self.graph_weight(x1)

        x3 = self.graph_weight_space(x2)

        x3 = window_reverse_3d(x3,16)


        x4 = self.temporal_space2(x3)

        return x4


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,1,1,0),
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# up = Up(1024,1024, 0)
# tensor = torch.ones(3,1024,256,256)
# tensor2 = torch.ones(3,512,256,256)
# up(tensor,tensor2)
        
        
    
    
# down = Down(96,192)
# tensor = torch.ones(3,96,256,256)
# print(down(tensor).shape)