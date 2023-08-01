import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import math

#import resnet

from adaptive_conv import AdaptiveConv, AdaptiveConv_m, AdaptiveConvPack, AdaptiveSam
import sys

from losses import multi_ce_loss, multi_smooth_loss
sys.path.append('/ghome/rongsh/.local/lib/python3.7/site-packages/att_grid_generator-0.0.0-py3.7-linux-x86_64.egg')
import att_grid_generator_cuda
from torch.nn.modules.batchnorm import _BatchNorm
from bn_lib.nn.modules import SynchronizedBatchNorm2d
from functools import partial
#sbn_layer = partial(SynchronizedBatchNorm2d, momentum=3e-4)
sbn_layer = nn.BatchNorm2d

model_dirs = {
  'resnet18': './output2/resnet18.pth.tar',
  'resnet50': './output2/resnet50.pth.tar',
  'resnet101': './output2/resnet101.pth.tar',
  'resnet152': './output2/resnet152.pth.tar',
}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, asc=False):
  """3x3 convolution with padding"""
  if asc is True:
    return AdaptiveConv(in_planes, out_planes, kernel_size=3)
  else:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, asc=False):
  """1x1 convolution"""
  if asc is True:
    return AdaptiveConv(in_planes, out_planes, kernel_size=1,
                        bias=False)
  else:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
               base_width=64, dilation=1, norm_layer=None):
    super(BasicBlock, self).__init__()
    if norm_layer is None:
      norm_layer = nn.BatchNorm2d
    if groups != 1 or base_width != 64:
      raise ValueError('BasicBlock only supports groups=1 and base_width=64')
    if dilation > 1:
      raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
    # Both self.conv1 and self.downsample layers downsample the input when stride != 1
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = norm_layer(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = norm_layer(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
               base_width=64, dilation=1, norm_layer=None):
    super(Bottleneck, self).__init__()
    if norm_layer is None:
      norm_layer = nn.BatchNorm2d
    width = int(planes * (base_width / 64.)) * groups
    # Both self.conv2 and self.downsample layers downsample the input when stride != 1
    self.conv1 = conv1x1(inplanes, width)
    self.bn1 = norm_layer(width)
    self.conv2 = conv3x3(width, width, stride, groups, dilation)
    self.bn2 = norm_layer(width)
    self.conv3 = conv1x1(width, planes * self.expansion)
    self.bn3 = norm_layer(planes * self.expansion)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out


class myResNet(nn.Module):

  def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
               groups=1, width_per_group=64, replace_stride_with_dilation=None,
               norm_layer=None, sam_size=224):
    super(myResNet, self).__init__()
    if norm_layer is None:
      norm_layer = nn.BatchNorm2d
    self._norm_layer = norm_layer

    self.inplanes = 64
    self.dilation = 1
    if replace_stride_with_dilation is None:
      # each element in the tuple indicates if we should replace
      # the 2x2 stride with a dilated convolution instead
      replace_stride_with_dilation = [False, False, False]
    if len(replace_stride_with_dilation) != 3:
      raise ValueError("replace_stride_with_dilation should be None "
                       "or a 3-element tuple, got {}".format(
        replace_stride_with_dilation))
    self.groups = groups
    self.base_width = width_per_group

    self.bn_data = norm_layer(3, eps=2e-5)  # extra

    self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = norm_layer(self.inplanes, eps=2e-5)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=1,
                                   dilate=replace_stride_with_dilation[0])
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                   dilate=replace_stride_with_dilation[1])
    self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                   dilate=replace_stride_with_dilation[2])
    # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    # self.fc = nn.Linear(512 * block.expansion, num_classes)

    # self.ads_conv = AdaptiveConv(256, 512)
    #self.att_conv1_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1,
    #                             padding=1,
    #                             dilation=1)
    self.sam_size = sam_size
    print('sam_size:{}'.format(self.sam_size))
    self.att_act1_1 = nn.ReLU(inplace=True)
    #self.att_conv1_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1,
    #                             padding=2,
    #                             dilation=2)
    self.att_act1_2 = nn.ReLU(inplace=True)

    #self.att_conv2_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1,
    #                             padding=1,
    #                             dilation=1)
    self.att_act2_1 = nn.ReLU(inplace=True)
    #self.att_conv2_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1,
    #                             padding=2,
    #                             dilation=2)
    self.att_act2_2 = nn.ReLU(inplace=True)
    #self.att_conv2_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1,
    #                             padding=3,
    #                             dilation=3)
    self.att_act2_3 = nn.ReLU(inplace=True)

    out_channels = 512
    self.in_channels = 512
    groups = 1
    self.kernel_size = _pair(3)

    self.weight_1 = nn.Parameter(
      torch.Tensor(out_channels, self.in_channels // groups,
                   *self.kernel_size))
    self.bias_1 = nn.Parameter(torch.Tensor(out_channels))
    self.weight_2 = nn.Parameter(
      torch.Tensor(out_channels, self.in_channels // groups,
                   *self.kernel_size))
    self.bias_2 = nn.Parameter(torch.Tensor(out_channels))
    self.reset_parameters()

    self.att_bn1 = norm_layer(512, eps=2e-5)

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.fc_att = nn.Linear(512, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    if zero_init_residual:
      for m in self.modules():
        if isinstance(m, Bottleneck):
          nn.init.constant_(m.bn3.weight, 0)
        elif isinstance(m, BasicBlock):
          nn.init.constant_(m.bn2.weight, 0)

  def reset_parameters(self):
    nn.init.kaiming_normal_(self.weight_1, a=0.25)
    nn.init.kaiming_normal_(self.weight_2, a=0.25)
    nn.init.constant_(self.bias_1, 0)
    nn.init.constant_(self.bias_2, 0)
    #n = self.in_channels
    #for k in self.kernel_size:
    #  n *= k
    #stdv = 1. / math.sqrt(n)
    #self.weight_1.data.uniform_(-stdv, stdv)
    #self.weight_2.data.uniform_(-stdv, stdv)

  def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
    norm_layer = self._norm_layer
    downsample = None
    previous_dilation = self.dilation
    if dilate:
      self.dilation *= stride
      stride = 1
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        conv1x1(self.inplanes, planes * block.expansion, stride),
        # norm_layer(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(block(self.inplanes, planes, groups=self.groups,
                          base_width=self.base_width, dilation=self.dilation,
                          norm_layer=norm_layer))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = F.interpolate(x, (self.sam_size, self.sam_size), mode='bilinear',
                      align_corners=False)
    #x = self.bn_data(x)
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    att_conv1_1 = F.conv2d(x, self.weight_1, bias=self.bias_1, stride=1, padding=1,
                           dilation=1)
    att_act1_1 = self.att_act1_1(att_conv1_1)
    att_conv1_2 = F.conv2d(x, self.weight_1, bias=self.bias_1, stride=1, padding=2,
                           dilation=2)
    att_act1_2 = self.att_act1_2(att_conv1_2)
    att_conv1 = att_act1_1 + att_act1_2
    att_conv2_1 = F.conv2d(att_conv1, self.weight_2, bias=self.bias_2, stride=1,
                           padding=1, dilation=1)
    att_act2_1 = self.att_act2_1(att_conv2_1)
    att_conv2_2 = F.conv2d(att_conv1, self.weight_2, bias=self.bias_2, stride=1,
                           padding=2, dilation=2)
    att_act2_2 = self.att_act2_2(att_conv2_2)
    att_conv2_3 = F.conv2d(att_conv1, self.weight_2, bias=self.bias_2, stride=1,
                           padding=3, dilation=3)
    att_act2_3 = self.att_act2_3(att_conv2_3)
    att_conv2 = att_act2_1 + att_act2_2 + att_act2_3


    out = self.att_bn1(att_conv2)
    out = self.relu(out)

    out = self.global_pooling(out)
    out = out.view(out.size(0), -1)
    out = self.fc_att(out)


    return att_conv2, out


def att_resnet18(pretrained=False, num_classes=1000, sam_size=224, **kwargs):

  #return _attresnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
  #                 sam_size=sam_size, **kwargs)
  model = myResNet(BasicBlock, [2, 2, 2, 2], num_classes, sam_size=sam_size,
                   norm_layer=None, **kwargs)
  if pretrained:
    state_dict = torch.load(model_dirs['resnet18'])
    model.load_state_dict(state_dict, strict=False)
    print('load resnet18 checkpoint')
  return model


class ResNet(nn.Module):

  def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
               groups=1, width_per_group=64, replace_stride_with_dilation=None,
               norm_layer=None, droprate=None):
    super(ResNet, self).__init__()
    if norm_layer is None:
      norm_layer = nn.BatchNorm2d
    self._norm_layer = norm_layer

    self.inplanes = 64
    self.dilation = 1
    self.droprate = droprate
    if droprate is not None:
      print('droprate: {}'.format(droprate))
    if replace_stride_with_dilation is None:
      # each element in the tuple indicates if we should replace
      # the 2x2 stride with a dilated convolution instead
      replace_stride_with_dilation = [False, False, False]
    if len(replace_stride_with_dilation) != 3:
      raise ValueError("replace_stride_with_dilation should be None "
                       "or a 3-element tuple, got {}".format(
        replace_stride_with_dilation))
    self.groups = groups
    self.base_width = width_per_group
    self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = norm_layer(self.inplanes)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                   dilate=replace_stride_with_dilation[0])
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                   dilate=replace_stride_with_dilation[1])
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                   dilate=replace_stride_with_dilation[2])
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512 * block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    if zero_init_residual:
      for m in self.modules():
        if isinstance(m, Bottleneck):
          nn.init.constant_(m.bn3.weight, 0)
        elif isinstance(m, BasicBlock):
          nn.init.constant_(m.bn2.weight, 0)

  def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
    norm_layer = self._norm_layer
    downsample = None
    previous_dilation = self.dilation
    if dilate:
      self.dilation *= stride
      stride = 1
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        conv1x1(self.inplanes, planes * block.expansion, stride),
        norm_layer(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(block(self.inplanes, planes, groups=self.groups,
                          base_width=self.base_width, dilation=self.dilation,
                          norm_layer=norm_layer))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.reshape(x.size(0), -1)
    x = self.fc(x)
    if self.droprate:
      x = F.dropout(x, p=self.droprate, training=self.training)

    return x

def resnet50(pretrained=False, progress=True, **kwargs):
  #return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
  #               **kwargs)
  model = ResNet(Bottleneck, [3, 4, 6, 3], norm_layer=None, **kwargs)
  if pretrained:
    state_dict = torch.load(model_dirs['resnet50'])
    model.load_state_dict(state_dict, strict=True)
    print('load resnet50 checkpiont')
  return model

def resnet101(pretrained=False, progress=True, **kwargs):
  #return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
  #               **kwargs)
  model = ResNet(Bottleneck, [3, 4, 23, 3], norm_layer=None, **kwargs)
  if pretrained:
    state_dict = torch.load(model_dirs['resnet101'])
    model.load_state_dict(state_dict, strict=True)
    print('load resnet101 checkpiont')
  return model

class trilinear_att(nn.Module):
  def __init__(self, ori_size, scale=5):
    super(trilinear_att, self).__init__()
    self.feature_norm = nn.Softmax(dim=2)
    self.bilinear_norm = nn.Softmax(dim=2)
    self.ori_size = ori_size
    self.dense_factor = scale

  def forward(self, feature_maps):
    batch_size = feature_maps.size(0)
    inplanes = feature_maps.size(1)
    h_in = feature_maps.size(2)
    w_in = feature_maps.size(3)
    feature_ori = feature_maps.reshape(batch_size, inplanes, -1)

    # *7 to obtain an appropriate scale for the input of softmax function.
    feature_norm = self.feature_norm(feature_ori * self.dense_factor)

    bilinear = torch.matmul(feature_norm, feature_ori.transpose(1, 2))
    bilinear = self.bilinear_norm(bilinear)
    trilinear_atts = torch.matmul(bilinear, feature_ori) \
      .view(batch_size, inplanes, h_in, w_in)
    # .view(batch_size, inplanes, h_in, w_in).detach()

    # sum_att = torch.sum(trilinear_atts, dim=1).unsqueeze(1)
    structure_att = torch.sum(trilinear_atts, dim=(2, 3))
    structure_att_sorted, _ = torch.sort(structure_att, dim=1)
    structure_att_mask = (structure_att_sorted[:, 1:] !=
                          structure_att_sorted[:, :-1])
    one_vector = torch.ones((batch_size, 1), dtype=torch.bool).cuda()
    #one_vector = torch.ones((batch_size, 1), dtype=torch.uint8).cuda()
    structure_att_mask = torch.cat((one_vector, structure_att_mask), dim=1)
    structure_att_mask = structure_att_mask.unsqueeze(-1).transpose(1, 2).float()
    structure_att_ori = trilinear_atts.reshape(batch_size, inplanes, -1)

    structure_att = torch.matmul(structure_att_mask, structure_att_ori)
    structure_att = structure_att.view(batch_size, 1, h_in, w_in)
    structure_att = F.interpolate(structure_att, (self.ori_size, self.ori_size),
                                  mode='bilinear', align_corners=False).squeeze(1)
    structure_att = structure_att * structure_att

    return structure_att

class FT_Resnet(nn.Module):
  def __init__(self, mode, num_classes=200, pretrained=True):
    super(FT_Resnet, self).__init__()
    if mode == 'resnet50':
      model = resnet50(pretrained=pretrained)
    else:
      model = resnet101(pretrained=pretrained)

    self.num_classes = num_classes
    self.num_features = model.layer4[1].conv1.in_channels

    self.features = nn.Sequential(
      model.conv1,
      model.bn1,
      model.relu,
      model.maxpool,
      model.layer1,
      model.layer2,
      model.layer3,
      model.layer4
    )

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(self.num_features, self.num_classes)
    #self.fc = nn.Sequential(
    #  nn.BatchNorm1d(self.num_features),
    #  nn.Linear(self.num_features, self.num_classes)
    #)

  def forward(self, x):
    features = self.features(x)
    x = self.avgpool(features).view(-1, self.num_features)
    p = self.fc(x)
    return p




def ft_resnet(mode='resnet50', num_classes=200, pretrained=True):
  return FT_Resnet(mode=mode, num_classes=num_classes, pretrained=pretrained)



class asResnet(nn.Module):
  def __init__(self, att_model, base_model, num_classes=200, as_kernel_size=3,
               sam_size=392, input_size=512, dense=2, crit=None):
    super(asResnet, self).__init__()

    self.input_size = input_size
    self.sam_size = sam_size
    self.dense = dense
    self.crit = crit
    self.as_conv = AdaptiveConv(3, 3, kernel_size=as_kernel_size)
    self.features = base_model.features
    self.num_features = base_model.num_features
    self.num_classes = num_classes

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    self.base_classifier = nn.Linear(self.num_features, num_classes)
    #self.base_classifier = nn.Sequential(
    #  nn.BatchNorm1d(self.num_features),
    #  nn.Linear(self.num_features, self.num_classes)
    #)
    #self.att_classifier = nn.Linear(self.num_att_features, num_classes)
    self.att_model = att_model

    self.trilinear = trilinear_att(input_size)


  def att_sample(self, data, structure_att, out_size, dense=2, get_sample=False,
                 use_ads=False):
    batch_size = structure_att.size(0)
    input_size = structure_att.size(1)
    map_sx, _ = torch.max(structure_att, 2)
    map_sx = map_sx.unsqueeze(2)
    map_sy, _ = torch.max(structure_att, 1)
    map_sy = map_sy.unsqueeze(2)
    sum_sx = torch.sum(map_sx, (1, 2), keepdim=True)
    sum_sy = torch.sum(map_sy, (1, 2), keepdim=True)
    map_sx = torch.div(map_sx, sum_sx)
    map_sy = torch.div(map_sy, sum_sy)
    map_xi = torch.zeros_like(map_sx)
    map_yi = torch.zeros_like(map_sy)
    # index_x = data.new_empty((batch_size, out_size, 1))
    # index_y = data.new_empty((batch_size, out_size, 1)).cuda()
    index_x = torch.zeros((batch_size, out_size, 1)).cuda()
    index_y = torch.zeros((batch_size, out_size, 1)).cuda()
    att_grid_generator_cuda.forward(map_sx, map_sy, map_xi, map_yi,
                                    index_x, index_y,
                                    input_size, out_size, dense,
                                    5, out_size / input_size)
    if get_sample and not use_ads:
      one_vector = torch.ones_like(index_x)
      grid_x = torch.matmul(one_vector, index_x.transpose(1, 2)).unsqueeze(-1)
      grid_y = torch.matmul(index_y, one_vector.transpose(1, 2)).unsqueeze(-1)
      grid = torch.cat((grid_x, grid_y), 3)
      structure_data = F.grid_sample(data, grid)
      return structure_data
    elif get_sample and use_ads:
      adsam = AdaptiveSam(3, 3, 3, groups=3, adaptive_groups=1)
      structure_data = adsam(data, index_y, index_x)
      return structure_data
    else:
      return index_x, index_y


  def forward(self, x, lbl=None):
    x_att, att_logits = self.att_model(x)

    structure_att = self.trilinear(x_att)
    idx_x, idx_y = self.att_sample(x, structure_att, self.sam_size, self.dense)

    x_rei = self.as_conv(x, idx_y, idx_x)
    x_rei = self.features(x_rei)
    x_rei = self.avgpool(x_rei)
    logits = self.base_classifier(x_rei.view(x_rei.size(0), -1))
    if self.training and lbl is None:
      '''
      if self.crit is 'multi_smooth_loss':
        loss = multi_smooth_loss((logits, att_logits), lbl, [1.0, 0.2]).unsqueeze(0)
      else:
        loss = multi_ce_loss((logits, att_logits), lbl, [1.0, 0.2]).unsqueeze(0)
      '''
      return logits, att_logits
    else:
      return logits






def as_resnet(mode, num_classes, sam_size, crit=None, dense=2,
              as_kernel_size=3, input_size=512):
  att_model = att_resnet18(pretrained=True, num_classes=num_classes,
                                  sam_size=sam_size)
  base_model = ft_resnet('resnet50', num_classes)
  model = asResnet(att_model, base_model, num_classes, as_kernel_size,
                   sam_size, input_size, dense, crit)
  return model













