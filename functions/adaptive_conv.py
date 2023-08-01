import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _pair

#import adaptive_conv_cuda
import sys
sys.path.append('/ghome/rongsh/.local/lib/python3.7/site-packages/ads_conv_cuda10-0.0.0-py3.7-linux-x86_64.egg')
import ads_conv_cuda
#import ads_m_conv_cuda
import time
import numpy as np
from torch.autograd import gradcheck


class AdaptiveConvFunction(Function):

  @staticmethod
  def forward(ctx,
              input,
              dilation,
              weight,
              stride_h,
              stride_w,
              padding=0,
              groups=1,
              adaptive_groups=1):
    if input is not None and input.dim() != 4:
      raise ValueError(
        "Expected 4D tensor as input, got {}D tensor instead.".format(
          input.dim()))
    #ctx.stride = _pair(stride)
    ctx.padding = _pair(padding)
    ctx.groups = groups
    ctx.adaptive_groups = adaptive_groups
    ctx.im2col_step = input.size(0)

    ctx.save_for_backward(input, dilation, weight, stride_h, stride_w)

    output = input.new_empty(
      AdaptiveConvFunction._output_size(input, weight, dilation))

    ctx.bufs_ = [input.new_empty(0), input.new_empty(0)]  # colums, ones

    if not input.is_cuda:
      raise NotImplementedError
    else:
      cur_im2col_step = min(ctx.im2col_step, input.shape[0])
      assert (input.shape[0] %
              cur_im2col_step) == 0, 'im2col step must divide batchsize'
      #adaptive_conv_cuda.adaptive_conv_forward_cuda(
      ads_conv_cuda.ads_conv_forward_cuda(
        input, weight, dilation, output, stride_h, stride_w,
        ctx.bufs_[0], ctx.bufs_[1], weight.size(3), weight.size(2),
        ctx.padding[1], ctx.padding[0], ctx.groups, ctx.adaptive_groups,
        cur_im2col_step)
    return output

  @staticmethod
  def backward(ctx, grad_output):
    input, dilation, weight, stride_h, stride_w = ctx.saved_tensors

    grad_input = grad_dilation = grad_weight = None

    if not grad_output.is_cuda:
      raise NotImplementedError
    else:
      cur_im2col_step = min(ctx.im2col_step, input.shape[0])
      assert (input.shape[0] %
              cur_im2col_step) == 0, 'im2col step must divide batchsize'

      if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
        grad_input = torch.zeros_like(input)
        grad_dilation = torch.zeros_like(dilation)
        ads_conv_cuda.ads_conv_backward_input_cuda(
          input, dilation, stride_h, stride_w, grad_output,
          grad_input, grad_dilation, weight, ctx.bufs_[0],
          weight.size(3), weight.size(2),
          ctx.padding[1], ctx.padding[0], ctx.groups, ctx.adaptive_groups,
          cur_im2col_step)

      if ctx.needs_input_grad[2]:
        grad_weight = torch.zeros_like(weight)
        ads_conv_cuda.ads_conv_backward_parameters_cuda(
          input, dilation, grad_output,
          grad_weight, stride_h, stride_w,
          ctx.bufs_[0], ctx.bufs_[1],
          weight.size(3), weight.size(2),
          ctx.padding[1], ctx.padding[0], ctx.groups, ctx.adaptive_groups,
          1, cur_im2col_step)

    return (grad_input, grad_dilation, grad_weight, None, None, None, None,
            None, None)

  @staticmethod
  def _output_size(input, weight, dilation, padding=0, stride=0):
    channels = weight.size(0)
    output_size = (input.size(0), channels, dilation.size(2), dilation.size(3))
    return output_size


adaptive_conv = AdaptiveConvFunction.apply

class AdaptiveConvmFunction(Function):

  @staticmethod
  def forward(ctx,
              input,
              dilation,
              weight,
              stride_h,
              stride_w,
              padding=0,
              groups=1,
              adaptive_groups=1):
    if input is not None and input.dim() != 4:
      raise ValueError(
        "Expected 4D tensor as input, got {}D tensor instead.".format(
          input.dim()))
    #ctx.stride = _pair(stride)
    ctx.padding = _pair(padding)
    ctx.groups = groups
    ctx.adaptive_groups = adaptive_groups
    ctx.im2col_step = input.size(0)

    ctx.save_for_backward(input, dilation, weight, stride_h, stride_w)

    output = input.new_empty(
      AdaptiveConvFunction._output_size(input, weight, dilation))

    ctx.bufs_ = [input.new_empty(0), input.new_empty(0)]  # colums, ones

    if not input.is_cuda:
      raise NotImplementedError
    else:
      cur_im2col_step = min(ctx.im2col_step, input.shape[0])
      assert (input.shape[0] %
              cur_im2col_step) == 0, 'im2col step must divide batchsize'
      #adaptive_conv_cuda.adaptive_conv_forward_cuda(
      ads_m_conv_cuda.ads_conv_m_forward_cuda(
        input, weight, dilation, output, stride_h, stride_w,
        ctx.bufs_[0], ctx.bufs_[1], weight.size(3), weight.size(2),
        ctx.padding[1], ctx.padding[0], ctx.groups, ctx.adaptive_groups,
        cur_im2col_step)
    return output

  @staticmethod
  def backward(ctx, grad_output):
    input, dilation, weight, stride_h, stride_w = ctx.saved_tensors

    grad_input = grad_dilation = grad_weight = None

    if not grad_output.is_cuda:
      raise NotImplementedError
    else:
      cur_im2col_step = min(ctx.im2col_step, input.shape[0])
      assert (input.shape[0] %
              cur_im2col_step) == 0, 'im2col step must divide batchsize'

      if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
        grad_input = torch.zeros_like(input)
        grad_dilation = torch.zeros_like(dilation)
        ads_m_conv_cuda.ads_conv_m_backward_input_cuda(
          input, dilation, stride_h, stride_w, grad_output,
          grad_input, grad_dilation, weight, ctx.bufs_[0],
          weight.size(3), weight.size(2),
          ctx.padding[1], ctx.padding[0], ctx.groups, ctx.adaptive_groups,
          cur_im2col_step)

      if ctx.needs_input_grad[2]:
        grad_weight = torch.zeros_like(weight)
        ads_m_conv_cuda.ads_conv_m_backward_parameters_cuda(
          input, dilation, grad_output,
          grad_weight, stride_h, stride_w,
          ctx.bufs_[0], ctx.bufs_[1],
          weight.size(3), weight.size(2),
          ctx.padding[1], ctx.padding[0], ctx.groups, ctx.adaptive_groups,
          1, cur_im2col_step)

    return (grad_input, grad_dilation, grad_weight, None, None, None, None,
            None, None)

  @staticmethod
  def _output_size(input, weight, dilation, padding=0, stride=0):
    channels = weight.size(0)
    output_size = (input.size(0), channels, dilation.size(2), dilation.size(3))
    return output_size

adaptive_conv_m = AdaptiveConvmFunction.apply

def main():
  device = torch.device("cuda")

  kwargs = {'dtype': torch.float32,
            'device': device,
            'requires_grad': True}
  kwargs_n = {'dtype': torch.float32,
            'device': device,
            'requires_grad': False}

  batch_size = 32
  in_channels = 3
  out_channels = 3
  height = 224
  width = 224
  padding = 0
  groups = 3
  adaptive_groups = 1
  output_size = (batch_size, out_channels, height//2, width//2)
  dilation_size = (batch_size, adaptive_groups, height//2, width//2)
  weight_size = (out_channels, in_channels//groups, 3, 3)

  #a = [-1, 0, 1]
  #b = [-1, 0, 1]
  #d = 2.4
  #v1 = []
  #v2 = []
  #v3 = []
  #v4 = []
  #for i in a:
  #  for j in b:
  #    v1.append(2*i*j*d-i*np.floor(j*d)-j*np.floor(i*d)-i-j)
  #    v2.append(-2*i*j*d+i*np.floor(j*d)+j*np.floor(i*d)+j)
  #    v3.append(-2*i*j*d+i*np.floor(j*d)+j*np.floor(i*d)+i)
  #    v4.append(2*i*j*d-i*np.floor(j*d)-j*np.floor(i*d))

  #x_1 = [1, 4, 6, 1, 4, 6, 1, 4, 6]
  #x_2 = [2, 0, 7, 2, 0, 7, 2, 0, 7]
  #x_3 = [1, 4, 6, 0, 0, 0, 1, 4, 6]
  #x_4 = [2, 0, 7, 0, 0, 0, 2, 0, 7]
  #val1 = np.array(v1) * np.array(x_1)
  #val2 = np.array(v2) * np.array(x_2)
  #val3 = np.array(v3) * np.array(x_3)
  #val4 = np.array(v4) * np.array(x_4)
  #val = np.sum(val1 + val2 + val3 + val4)
  #print(val)




  #x = torch.arange(height, **kwargs).repeat(width, 1).\
  #  view(height, width).repeat(in_channels, 1)
  #x = x.repeat(batch_size, 1).view(batch_size, in_channels, height, width)
  x = torch.randn((batch_size, in_channels, height, width), **kwargs)
  #x2 = torch.tensor(x.clone().detach())
  #x0 = torch.arange(width).to(dtype=torch.float64, device=device)
  #x1 = x0.repeat(batch_size, height, 1).unsqueeze(1).requires_grad_()
  x2 = x.clone().detach().requires_grad_()
  d2 = torch.ones(dilation_size, **kwargs)
  #d = d2.clone().detach().requires_grad_()
  d = torch.randn(dilation_size, **kwargs)
  w = torch.ones(weight_size, **kwargs)
  weight = torch.Tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]]],
                         [[[0, 0, 0], [0, 1, 0], [0, 0, 0]]],
                         [[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]]).cuda()
  #w2 = torch.tensor(w.clone().detach())
  w2 = w.clone().detach().requires_grad_()
  stride_h_1 = torch.arange(0, height, 2, **kwargs_n).repeat(batch_size, 1)
  stride_h = stride_h_1.unsqueeze(-1)
  stride_w_1 = torch.arange(0, width, 2, **kwargs_n).repeat(batch_size, 1)
  stride_w = stride_w_1.unsqueeze(-1)
  one_vector = torch.ones_like(stride_h)
  grid_x = torch.matmul(one_vector, stride_w.transpose(1, 2)).unsqueeze(-1)
  grid_x = grid_x / 111 - 1
  grid_y = torch.matmul(stride_h, one_vector.transpose(1, 2)).unsqueeze(-1)
  grid_y = grid_y / 111 - 1
  grid = torch.cat((grid_x, grid_y), 3)
  str_data = F.grid_sample(x, grid)
  grad_output = torch.ones(output_size, **kwargs)
  grad_output2 = grad_output.clone().detach()

  start_time = time.time()
  output = adaptive_conv(x, d2, weight, stride_h, stride_w, padding, groups,
                         adaptive_groups)
  ok4 = torch.allclose(output, str_data)
  output.backward(grad_output2)
  time_ads = time.time() - start_time
  start_time = time.time()
  output_ori = F.conv2d(x2, w2, padding=2, dilation=2, stride=2, groups=groups)
  output_ori.backward(grad_output)
  time_ori = time.time() - start_time
  ok = torch.allclose(output_ori, output)
  ok2 = torch.allclose(x.grad, x2.grad)
  ok3 = torch.allclose(w.grad, w2.grad)
  print(ok, ok2, ok3)
  print(time_ori, time_ads)




if __name__ == '__main__':
  main()
