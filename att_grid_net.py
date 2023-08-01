import os
import copy
import random
from collections import OrderedDict
from typing import List, Dict, Tuple, Callable, Optional, Union

import torch
import numpy as np
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torch.autograd import Function
from model import ft_resnet
from losses import multi_smooth_loss, multi_ce_loss
import sys
sys.path.append('/ghome/rongsh/.local/lib/python3.7/site-packages/att_grid_generator-0.0.0-py3.7-linux-x86_64.egg')
import att_grid_generator_cuda
from feat import MultiHeadAttention




def makeGaussian(size, fwhm=3, center=None):
	x = np.arange(0, size, 1, float)
	y = x[:, np.newaxis]

	if center is None:
		x0 = y0 = size // 2
	else:
		x0 = center[0]
		y0 = center[1]

	return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)


class KernelGenerator(nn.Module):
	def __init__(self, size, offset=None):
		super(KernelGenerator, self).__init__()

		self.size = self._pair(size)
		xx, yy = np.meshgrid(np.arange(0, size), np.arange(0, size))
		if offset is None:
			offset_x = offset_y = size // 2
		else:
			offset_x, offset_y = self._pair(offset)
		self.factor = torch.from_numpy(
			-(np.power(xx - offset_x, 2) + np.power(yy - offset_y, 2)) / 2).float()

	@staticmethod
	def _pair(x):
		return (x, x) if isinstance(x, int) else x

	def forward(self, theta):
		pow2 = torch.pow(theta * self.size[0], 2)
		kernel = 1.0 / (2 * np.pi * pow2) * torch.exp(
			self.factor.to(theta.device) / pow2)
		return kernel / kernel.max()


def kernel_generate(theta, size, offset=None):
	return KernelGenerator(size, offset)(theta)


def _mean_filter(input):
	batch_size, num_channels, h, w = input.size()
	threshold = torch.mean(input.view(batch_size, num_channels, h * w), dim=2)
	return threshold.contiguous().view(batch_size, num_channels, 1, 1)


class PeakStimulation(Function):

	@staticmethod
	def forward(ctx, input, return_aggregation, win_size, peak_filter):
		ctx.num_flags = 4

		assert win_size % 2 == 1, 'Window size for peak finding must be odd.'
		offset = (win_size - 1) // 2
		padding = torch.nn.ConstantPad2d(offset, float('-inf'))
		padded_maps = padding(input)
		batch_size, num_channels, h, w = padded_maps.size()
		element_map = torch.arange(0, h * w).long().view(1, 1, h, w)[:, :,
		              offset: -offset, offset: -offset]
		element_map = element_map.to(input.device)
		_, indices = F.max_pool2d(
			padded_maps,
			kernel_size=win_size,
			stride=1,
			return_indices=True)
		peak_map = (indices == element_map)

		if peak_filter:
			mask = input >= peak_filter(input)
			peak_map = (peak_map & mask)
		peak_list = torch.nonzero(peak_map, as_tuple=False)
		ctx.mark_non_differentiable(peak_list)

		if return_aggregation:
			peak_map = peak_map.float()
			ctx.save_for_backward(input, peak_map)
			return peak_list, (input * peak_map).view(batch_size, num_channels,
			                                          -1).sum(2) / \
			       peak_map.view(batch_size, num_channels, -1).sum(2)
		else:
			return peak_list

	@staticmethod
	def backward(ctx, grad_peak_list, grad_output):
		input, peak_map, = ctx.saved_tensors
		batch_size, num_channels, _, _ = input.size()
		grad_input = peak_map * grad_output.view(batch_size, num_channels, 1, 1) / \
		             (peak_map.view(batch_size, num_channels, -1).sum(2).view(
			             batch_size, num_channels, 1, 1) + 1e-6)
		return (grad_input,) + (None,) * ctx.num_flags


def peak_stimulation(input, return_aggregation=True, win_size=3,
                     peak_filter=None):
	return PeakStimulation.apply(input, return_aggregation, win_size, peak_filter)


class ScaleLayer(nn.Module):

	def __init__(self, init_value=1e-3):
		super().__init__()
		self.scale = nn.Parameter(torch.FloatTensor([init_value]))

	def forward(self, input):
		return input * self.scale

class MASNET(nn.Module):

	def __init__(self, base_model, num_classes, task_input_size, base_ratio,
	             radius, radius_inv, crit, args):
		super(MASNET, self).__init__()

		self.crit = crit
		self.dense = args['dense']
		self.sam_size = args['sam_size']
		self.guide_weight = args['guide_weight']
		self.thresh_rate = args['thresh_rate']
		self.use_transformer = args['use_transformer']
		if self.use_transformer is True:
			self.transformer = MultiHeadAttention(3, 2048, 512, 2048, dropout=0.5)

		self.grid_size = 31
		self.padding_size = 30
		self.global_size = self.grid_size + 2 * self.padding_size
		self.input_size_net = task_input_size
		gaussian_weights = torch.FloatTensor(
			makeGaussian(2 * self.padding_size + 1, fwhm=13))
		self.base_ratio = base_ratio
		self.radius = ScaleLayer(radius)
		self.radius_inv = ScaleLayer(radius_inv)

		self.filter = nn.Conv2d(1, 1, kernel_size=(
		2 * self.padding_size + 1, 2 * self.padding_size + 1), bias=False)
		self.filter.weight[0].data[:, :, :] = gaussian_weights

		self.P_basis = torch.zeros(2, self.grid_size + 2 * self.padding_size,
		                           self.grid_size + 2 * self.padding_size)
		for k in range(2):
			for i in range(self.global_size):
				for j in range(self.global_size):
					self.P_basis[k, i, j] = k * (i - self.padding_size) / (
							self.grid_size - 1.0) + (1.0 - k) * (j - self.padding_size) / (
							                        self.grid_size - 1.0)

		self.features = base_model.features
		self.num_features = base_model.num_features

		self.raw_classifier = nn.Linear(2048, num_classes)
		self.sampler_buffer = nn.Sequential(
			nn.Conv2d(2048, 2048, kernel_size=3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(2048),
			nn.ReLU(),
			)
		self.sampler_classifier = nn.Linear(2048, num_classes)

		self.sampler_buffer1 = nn.Sequential(
			nn.Conv2d(2048, 2048, kernel_size=3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(2048),
			nn.ReLU(),
			)
		self.sampler_classifier1 = nn.Linear(2048, num_classes)

		self.con_classifier = nn.Linear(int(self.num_features * 3), num_classes)

		self.avg = nn.AdaptiveAvgPool2d(1)
		self.max_pool = nn.AdaptiveMaxPool2d(1)

		self.map_origin = nn.Conv2d(2048, num_classes, 1, 1, 0)

	def create_grid(self, x):
		P = torch.autograd.Variable(
			torch.zeros(1, 2, self.grid_size + 2 * self.padding_size,
			            self.grid_size + 2 * self.padding_size).cuda(),
			requires_grad=False)
		P[0, :, :, :] = self.P_basis
		P = P.expand(x.size(0), 2, self.grid_size + 2 * self.padding_size,
		             self.grid_size + 2 * self.padding_size)

		x_cat = torch.cat((x, x), 1)
		p_filter = self.filter(x)
		x_mul = torch.mul(P, x_cat).view(-1, 1, self.global_size, self.global_size)
		all_filter = self.filter(x_mul).view(-1, 2, self.grid_size, self.grid_size)

		x_filter = all_filter[:, 0, :, :].contiguous().view(-1, 1, self.grid_size,
		                                                    self.grid_size)
		y_filter = all_filter[:, 1, :, :].contiguous().view(-1, 1, self.grid_size,
		                                                    self.grid_size)

		x_filter = x_filter / p_filter
		y_filter = y_filter / p_filter

		xgrids = x_filter * 2 - 1
		ygrids = y_filter * 2 - 1
		xgrids = torch.clamp(xgrids, min=-1, max=1)
		ygrids = torch.clamp(ygrids, min=-1, max=1)

		xgrids = xgrids.view(-1, 1, self.grid_size, self.grid_size)
		ygrids = ygrids.view(-1, 1, self.grid_size, self.grid_size)

		grid = torch.cat((xgrids, ygrids), 1)

		grid = F.interpolate(grid, size=(self.input_size_net, self.input_size_net),
		                     mode='bilinear', align_corners=True)

		grid = torch.transpose(grid, 1, 2)
		grid = torch.transpose(grid, 2, 3)

		return grid

	def att_sample(self, data, structure_att, out_size, dense=2, get_sample=True,
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
			structure_data = F.grid_sample(data, grid, align_corners=True)
			return structure_data
		elif get_sample and use_ads:
			adsam = AdaptiveSam(3, 3, 3, groups=3, adaptive_groups=1)
			structure_data = adsam(data, index_y, index_x)
			return structure_data
		else:
			return index_x, index_y

	def generate_att_map(self, input_x, class_response_maps, thresh=None):
		N, C, H, W = class_response_maps.size()

		score_pred, sort_number = torch.sort(
			F.softmax(F.adaptive_avg_pool2d(class_response_maps, 1), dim=1), dim=1,
			descending=True)

		xs = []
		masks = []

		for idx_i in range(N):
			attention_map = class_response_maps[idx_i, sort_number[idx_i, 0], :, :]
			xs.append(attention_map)

		xs = torch.cat(xs, 0)
		att_map = F.interpolate(xs, size=(self.input_size_net, self.input_size_net),
		                        mode='bilinear', align_corners=True)
		x_sampled_zoom = self.att_sample(input_x, att_map.squeeze(1),
		                                 self.sam_size, self.dense)

		if thresh is not None:
			for idx_i in range(N):
				threshold, _ = att_map[idx_i].view(-1).max(-1)
				threshold = threshold * thresh
				mask = torch.ones_like(att_map[idx_i])
				mask[att_map[idx_i] > threshold] = 0.
				masks.append(mask.unsqueeze(0))
			masks = torch.cat(masks, 0)
			new_input_x = input_x * masks
			return x_sampled_zoom, new_input_x, att_map

		return x_sampled_zoom, att_map

	def forward(self, input_x, p, lbl=None):

		self.map_origin.weight.data.copy_(
			self.raw_classifier.weight.data.unsqueeze(-1).unsqueeze(-1))
		self.map_origin.bias.data.copy_(self.raw_classifier.bias.data)

		feature_raw = self.features(input_x)


		with torch.no_grad():
			class_response_maps = F.interpolate(self.map_origin(feature_raw),
			                                    size=self.grid_size, mode='bilinear',
			                                    align_corners=True)
		x_sampled_zoom, new_input_x, att_zoom = self.generate_att_map(input_x,
		                                                          class_response_maps, self.thresh_rate)
		#x_sampled_zoom, new_input_x, att_zoom = self.generate_map(input_x,
		#                                                  class_response_maps, 0.7)

		#new_agg_origin = self.raw_classifier(self.avg(new_feature_raw).view(-1, 2048))
		with torch.no_grad():
			new_feature_raw = self.features(new_input_x)
			new_class_response_maps = F.interpolate(self.map_origin(new_feature_raw),
			                                    size=self.grid_size, mode='bilinear',
			                                    align_corners=True)
		#x_sampled_inv, att_inv = self.generate_map(input_x, new_class_response_maps)
		x_sampled_inv, att_inv = self.generate_att_map(input_x, new_class_response_maps)


		feature_D = self.sampler_buffer(self.features(x_sampled_zoom))

		feature_C = self.sampler_buffer1(self.features(x_sampled_inv))

		if self.use_transformer is True:
			protos = torch.cat([self.avg(feature_raw).view(-1, 2048).unsqueeze(1),
			                    self.avg(feature_D).view(-1, 2048).unsqueeze(1),
			                    self.avg(feature_C).view(-1, 2048).unsqueeze(1)], 1)
			protos = self.transformer(protos, protos, protos)
			agg_origin = self.raw_classifier(protos[:, 0, :])
			agg_sampler = self.sampler_classifier(protos[:, 1, :])
			agg_sampler1 = self.sampler_classifier1(protos[:, 2, :])
			protos = protos.reshape(-1, 2048 * 3)
			aggregation = self.con_classifier(protos)
		else:
			agg_origin = self.raw_classifier(self.avg(feature_raw).view(-1, 2048))
			agg_sampler = self.sampler_classifier(self.avg(feature_D).view(-1, 2048))
			agg_sampler1 = self.sampler_classifier1(
				self.avg(feature_C).view(-1, 2048))
			aggregation = self.con_classifier(torch.cat(
				[self.avg(feature_raw).view(-1, 2048), self.avg(feature_D).view(-1, 2048),
				 self.avg(feature_C).view(-1, 2048)], 1))

		logits_tuple = (aggregation, agg_origin, agg_sampler, agg_sampler1)

		if self.training and lbl is not None:
			if self.crit == 'multi_smooth_loss':
				loss = multi_smooth_loss(logits_tuple, lbl, [1.0, 1.0, 1.0, 1.0])
			else:
				loss = multi_ce_loss(logits_tuple, lbl, [1.0, 1.0, 1.0, 1.0])
			if self.guide_weight is not None:
				batch_idx = torch.arange(agg_origin.size(0), dtype=torch.int64)
				p_b = F.log_softmax(agg_origin, dim=1)[batch_idx, lbl]
				p_c = F.log_softmax(aggregation, dim=1)[batch_idx, lbl]
				p_d = F.log_softmax(agg_sampler, dim=1)[batch_idx, lbl]
				p_s = F.log_softmax(agg_sampler1, dim=1)[batch_idx, lbl]
				#loss_gud = F.relu(p_b - p_d).mean() + F.relu(p_b - p_s).mean()
				loss_rela = F.relu(p_d - p_c).mean() + F.relu(p_s - p_c).mean() + F.relu(p_b - p_c).mean()
				loss += loss_rela * self.guide_weight
			return logits_tuple, loss.unsqueeze(0), att_zoom, att_inv, \
			       new_input_x, x_sampled_zoom, x_sampled_inv
		else:
			return aggregation


#def s3n(
#	mode: str = 'resnet50',
#	num_classes: int = 200,
#	crit: str = 'multi_ce_loss',
#	task_input_size: int = 448,
#	base_ratio: float = 0.09,
#	radius: float = 0.09,
#	radius_inv: float = 0.3) -> nn.Module:
def masnet(mode, num_classes, crit, input_size=448, base_ratio=0.09, radius=0.09,
        radius_inv=0.3, args=None):
	""" Selective sparse sampling.
	"""
	if mode != 'masnet':
		raise NotImplementedError

	classify_network = ft_resnet(mode='resnet50', num_classes=num_classes)
	model = MASNET(classify_network, num_classes, input_size, base_ratio,
	            radius, radius_inv, crit, args)

	return model

'''
def three_stage(
	ctx: Context,
	train_ctx: Context) -> None:
	"""Three stage.
	"""

	if train_ctx.is_train:
		p = 0 if train_ctx.epoch_idx <= 20 else 1
	else:
		p = 1 if train_ctx.epoch_idx <= 20 else 2

	train_ctx.output = train_ctx.model(train_ctx.input, p)

	raise train_ctx.Skip
'''