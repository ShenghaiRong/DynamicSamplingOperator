import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import numpy as np
import scipy.stats as st
import random

from functions.adaptive_conv import adaptive_conv, adaptive_conv_m


def gkern(kernlen=21, nsig=3):
  """Returns a 2D Gaussian kernel."""

  x = np.linspace(-nsig, nsig, kernlen + 1)
  kern1d = np.diff(st.norm.cdf(x))
  kern2d = np.outer(kern1d, kern1d)
  return kern2d / kern2d.sum()

def create_unit_mapping(in_channels, out_channels, kernel_size):
  res = torch.zeros((out_channels, in_channels, kernel_size, kernel_size))
  inter_ind = int(kernel_size / 2)
  for i in range(out_channels):
    for j in range(in_channels):
      if j == i:
        res[i, j, inter_ind, inter_ind] = 1
  return res

class AdaptiveConv(nn.Module):

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size=3,
               padding=0,
               groups=1,
               adaptive_groups=1,
               dilation_factor=1,
               bias=False,
               fix=False,
               pool=False):
    super(AdaptiveConv, self).__init__()

    assert not bias
    assert in_channels % groups == 0, \
      'in_channels {} cannot be divisible by groups {}'.format(
        in_channels, groups)
    assert out_channels % groups == 0, \
      'out_channels {} cannot be divisible by groups {}'.format(
        out_channels, groups)

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = _pair(kernel_size)
    self.padding = _pair(padding)
    self.groups = groups
    self.adaptive_groups = adaptive_groups
    self.d_factor = dilation_factor
    self.pool = pool


    if fix == True:
      #print('kernel_size{}'.format(kernel_size))
      weight = create_unit_mapping(in_channels, out_channels, kernel_size)
      self.weight = nn.Parameter(weight)
      print(self.weight.shape)
    else:
      #sig = random.random() * 8 + 1.
      #w = gkern(kernel_size, 9)
      weight = create_unit_mapping(in_channels, out_channels, kernel_size)
      #weight = torch.Tensor(w).unsqueeze(0).repeat(in_channels // groups, 1, 1)
      #weight = weight.unsqueeze(0).repeat(out_channels, 1, 1, 1).cuda()
      self.weight = nn.Parameter(weight)
      #self.weight = nn.Parameter(
      #  torch.Tensor(out_channels, in_channels // self.groups,
      #               *self.kernel_size))
      #self.reset_parameters()

  def reset_parameters(self):
    n = self.in_channels
    for k in self.kernel_size:
      n *= k
    stdv = 1. / math.sqrt(n)
    self.weight.data.uniform_(-stdv, stdv)
    #torch.nn.init.ones_(self.weight)

  def forward(self, x, stride_h, stride_w, dilation=None):
    input_h = x.size(2)
    input_w = x.size(3)

    s_h = stride_h.size(1)
    s_w = stride_w.size(1)
    if self.pool:
      stride_h = stride_h.transpose(1, 2)
      stride_w = stride_w.transpose(1, 2)
      stride_h = F.max_pool1d(stride_h, kernel_size=1, stride=s_h//input_h*2)
      stride_w = F.max_pool1d(stride_w, kernel_size=1, stride=s_w//input_w*2)
      stride_h = stride_h.transpose(1, 2)
      stride_w = stride_w.transpose(1, 2)

    stride_h = (stride_h + 1) * (input_h - 1) / 2
    stride_w = (stride_w + 1) * (input_w - 1) / 2
    dilation_h = stride_h.size(1)
    dilation_w = stride_w.size(1)
    if dilation is None:
      batch_size = x.size(0)
      dilation = torch.ones((batch_size, 1, dilation_h, dilation_w),
                            dtype=torch.float32).cuda()
      dilation = dilation * self.d_factor
      dilation = dilation.detach()
    return adaptive_conv(x, dilation, self.weight, stride_h, stride_w,
                         self.padding, self.groups, self.adaptive_groups)


class AdaptiveConv_m(nn.Module):

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size=3,
               padding=0,
               groups=1,
               adaptive_groups=1,
               dilation_factor=1,
               bias=False):
    super(AdaptiveConv_m, self).__init__()

    assert not bias
    assert in_channels % groups == 0, \
      'in_channels {} cannot be divisible by groups {}'.format(
        in_channels, groups)
    assert out_channels % groups == 0, \
      'out_channels {} cannot be divisible by groups {}'.format(
        out_channels, groups)

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = _pair(kernel_size)
    self.padding = _pair(padding)
    self.groups = groups
    self.adaptive_groups = adaptive_groups
    self.d_factor = dilation_factor

    #sig = random.random() * 8 + 1.
    #w = gkern(kernel_size, 9)
    #weight = torch.Tensor(w).unsqueeze(0).repeat(in_channels//groups, 1, 1)
    #weight = weight.unsqueeze(0).repeat(out_channels, 1, 1, 1).cuda()
    self.weight = nn.Parameter(
      torch.Tensor(out_channels, in_channels // self.groups,
                   *self.kernel_size))
    #self.reset_parameters()
    #weight = torch.Tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    #                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    #                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
    #                       [[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    #                        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    #                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
    #                       [[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    #                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    #                        [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]]).cuda()
    #weight = torch.Tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]] * out_channels).cuda()
    #self.weight = nn.Parameter(weight)

  def reset_parameters(self):
    n = self.in_channels
    for k in self.kernel_size:
      n *= k
    stdv = 1. / math.sqrt(n)
    self.weight.data.uniform_(-stdv, stdv)
    #torch.nn.init.ones_(self.weight)

  def forward(self, x, stride_h, stride_w, dilation=None):
    input_h = x.size(2)
    input_w = x.size(3)

    #s_h = stride_h.size(1)
    #s_w = stride_w.size(1)
    #stride_h = stride_h.transpose(1, 2)
    #stride_w = stride_w.transpose(1, 2)
    #stride_h = F.max_pool1d(stride_h, kernel_size=1, stride=s_h//input_h*2)
    #stride_w = F.max_pool1d(stride_w, kernel_size=1, stride=s_w//input_w*2)
    #stride_h = stride_h.transpose(1, 2)
    #stride_w = stride_w.transpose(1, 2)

    stride_h = (stride_h + 1) * (input_h - 1) / 2
    stride_w = (stride_w + 1) * (input_w - 1) / 2
    dilation_h = stride_h.size(2)
    dilation_w = stride_w.size(2)
    if dilation is None:
      batch_size = x.size(0)
      dilation = torch.ones((batch_size, self.adaptive_groups,
                             dilation_h, dilation_w),
                            dtype=torch.float32).cuda()
      dilation = dilation * self.d_factor
      dilation = dilation.detach()
    return adaptive_conv_m(x, dilation, self.weight, stride_h, stride_w,
                         self.padding, self.groups, self.adaptive_groups)

class AdaptiveSam(nn.Module):

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size=3,
               padding=0,
               groups=1,
               adaptive_groups=1,
               bias=False):
    super(AdaptiveSam, self).__init__()

    assert not bias
    assert in_channels % groups == 0, \
      'in_channels {} cannot be divisible by groups {}'.format(
        in_channels, groups)
    assert out_channels % groups == 0, \
      'out_channels {} cannot be divisible by groups {}'.format(
        out_channels, groups)

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = _pair(kernel_size)
    self.padding = _pair(padding)
    self.groups = groups
    self.adaptive_groups = adaptive_groups


    #self.weight = nn.Parameter(
    #  torch.Tensor(out_channels, in_channels // self.groups,
    #               *self.kernel_size))
    weight = torch.Tensor([[[[0, 0, 0], [0, 1, 0], [0, 0 ,0]],
                            [[0, 0, 0], [0, 0 ,0], [0, 0, 0]],
                            [[0, 0, 0], [0, 0, 0], [0, 0 ,0]]],
                           [[[0, 0, 0], [0, 0, 0], [0, 0 ,0]],
                            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                            [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
                          [[[0, 0, 0], [0, 0, 0], [0, 0 ,0]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]]).cuda()
    self.weight = nn.Parameter(weight)
    self.weight.requires_grad = False
    #self.weight = nn.Parameter(
    #  torch.ones(out_channels, in_channels // self.groups,
    #               *self.kernel_size) * 1/9)
    #self.reset_parameters()

  def reset_parameters(self):
    n = self.in_channels
    for k in self.kernel_size:
      n *= k
    stdv = 1. / math.sqrt(n)
    #self.weight.data.uniform_(-stdv, stdv)
    torch.nn.init.ones_(self.weight)

  def forward(self, x, stride_h, stride_w):
    batch_size = x.size(0)
    dilation = torch.ones((batch_size, self.adaptive_groups, 224, 224),
                          dtype=torch.float32).cuda()
    dilation = dilation
    dilation = dilation.detach()
    input_h = x.size(2)
    input_w = x.size(3)
    stride_h = (stride_h + 1) * (input_h - 1) / 2
    stride_w = (stride_w + 1) * (input_w - 1) / 2
    return adaptive_conv(x, dilation, self.weight, stride_h, stride_w,
                         self.padding, self.groups, self.adaptive_groups)


class AdaptiveConvPack(AdaptiveConv):

  def __init__(self, *args, **kwargs):
    super(AdaptiveConvPack, self).__init__(*args, **kwargs)

    self.conv_dilation = nn.Conv2d(
      self.in_channels,
      self.adaptive_groups * 1,
      kernel_size=self.kernel_size,
      stride=_pair(2),
      padding=_pair(1),
      bias=True)
    self.init_dilation()

  def init_dilation(self):
    n = self.in_channels
    for k in self.kernel_size:
      n *= k
    stdv = 1. / math.sqrt(n)
    self.conv_dilation.weight.data.uniform_(-stdv, stdv)
    self.conv_dilation.bias.data.zero_()

  def forward(self, x, stride_h=None, stride_w=None):
    dilation = self.conv_dilation(x)
    if stride_h is None or stride_w is None:
      b = x.size(0)
      h = x.size(2)
      stride_h = torch.arange(0, h, 2, dtype=torch.float32).repeat(b, 1)
      stride_h = stride_h.unsqueeze(-1).cuda()
      stride_w = stride_h
    return adaptive_conv(x, dilation, self.weight, stride_h, stride_w,
                         self.padding, self.groups, self.adaptive_groups)


class Pos2Weight(nn.Module):
  def __init__(self, inC, kernel_size=3, outC=3):
    super(Pos2Weight, self).__init__()
    self.inC = inC
    self.kernel_size = kernel_size
    self.outC = outC
    self.meta_block = nn.Sequential(
      nn.Linear(2, 256),
      nn.ReLU(inplace=True),
      nn.Linear(256, self.kernel_size * self.kernel_size * self.inC * self.outC)
    )

  def forward(self, x):
    output = self.meta_block(x)
    return output


