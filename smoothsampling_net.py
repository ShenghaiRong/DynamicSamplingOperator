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
sys.path.append('/ghome/rongsh/.local/lib/python3.7/site-packages/att_smooth-0.0.0-py3.7-linux-x86_64.egg')
import att_smooth_cuda
from feat import MultiHeadAttention


def makeGaussian1D(size, peak=1.4, fwhm=130, center=None):
	x = np.arange(0, size, 1, float)

	if center is None:
		x0 = size // 2
	else:
		x0 = center
	return np.exp(-4 * np.log(peak) * (x - x0) ** 2 / fwhm ** 2)

#def makeGaussian1D(size, sigma=3, center=None):
#	x = np.arange(0, size, 1, float)
#
#	if center is None:
#		x0 = size // 2
#	else:
#		x0 = center
#	#return np.exp(-4 * np.log(peak) * (x - x0) ** 2 / fwhm ** 2)
#	return np.exp(-(x - x0) ** 2 / sigma ** 2)

class MASNET2(nn.Module):

	def __init__(self, base_model, num_classes, task_input_size, base_ratio,
	             radius, radius_inv, crit, args):
		super(MASNET2, self).__init__()

		self.crit = crit
		self.dense = args['dense']
		self.sam_size = args['sam_size']
		self.guide_weight = args['guide_weight']
		self.thresh_rate = args['thresh_rate']
		self.use_transformer = args['use_transformer']
		self.droprate = args['droprate']
		self.t = 2
		if self.use_transformer is True:
			self.transformer = MultiHeadAttention(3, 2048, 512, 2048, dropout=0.5)

		self.grid_size = self.sam_size
		self.padding_size = self.sam_size - 1
		self.global_size = self.grid_size + 2 * self.padding_size
		self.input_size_net = task_input_size
		gaussian_weights = torch.FloatTensor(
			makeGaussian1D(2 * self.padding_size + 1, peak=args['peak'], fwhm=130))
		#gaussian_weights = torch.FloatTensor(
		#	makeGaussian1D(2 * self.padding_size + 1, sigma=args['sigma']))

		self.filter = nn.Conv1d(1, 1, kernel_size=2 * self.padding_size + 1, bias=False)
		self.filter.weight[0].data[:, :] = gaussian_weights

		self.P_basis = torch.zeros(1, 1, self.global_size)

		for i in range(self.global_size):
			self.P_basis[0, 0, i] = (i - self.padding_size) / (self.grid_size - 1.0)

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

	def adaptive_sample(self, data, structure_att):
		batch_size = structure_att.size(0)
		input_size = structure_att.size(1)
		map_sx, _ = torch.max(structure_att, 1)
		map_sx = map_sx.unsqueeze(2)
		map_sy, _ = torch.max(structure_att, 2)
		map_sy = map_sy.unsqueeze(2)

		#map_sx = map_sx ** self.t
		#map_sy = map_sy ** self.t

		sum_sx = torch.sum(map_sx, (1, 2), keepdim=True)
		sum_sy = torch.sum(map_sy, (1, 2), keepdim=True)
		# map_sx size : (B, input_size, 1)
		map_sx = torch.div(map_sx, sum_sx)
		map_sy = torch.div(map_sy, sum_sy)

		att_smooth_cuda.forward(map_sx, map_sy,
		                        input_size, self.sam_size, self.dense,
		                        5, self.sam_size / input_size)

		# map_sx size : (B, 1, input_size)
		map_sx = map_sx.transpose(1, 2)
		map_sy = map_sy.transpose(1, 2)

		map_sx_n = F.interpolate(map_sx, size=self.sam_size, mode='linear',
		                       align_corners=True)
		map_sy_n = F.interpolate(map_sy, size=self.sam_size, mode='linear',
		                       align_corners=True)

		map_sx = nn.ReflectionPad1d(self.padding_size)(map_sx_n)
		map_sy = nn.ReflectionPad1d(self.padding_size)(map_sy_n)
		#map_sx = nn.ConstantPad1d(self.padding_size, 0.)(map_sx_n)
		#map_sy = nn.ConstantPad1d(self.padding_size, 0.)(map_sy_n)

		P = torch.autograd.Variable(
			torch.zeros(1, 1, self.global_size).cuda(), requires_grad=False)
		P[:, :, :] = self.P_basis
		P = P.expand(batch_size, 1, self.global_size)

		px_filter = self.filter(map_sx)
		py_filter = self.filter(map_sy)

		x_mul = torch.mul(P, map_sx).view(-1, 1, self.global_size)
		y_mul = torch.mul(P, map_sy).view(-1, 1, self.global_size)

		x_filter = self.filter(x_mul).view(-1, 1, self.grid_size)
		y_filter = self.filter(y_mul).view(-1, 1, self.grid_size)

		x_filter = x_filter / px_filter
		y_filter = y_filter / py_filter

		xgrids = x_filter * 2 - 1
		ygrids = y_filter * 2 - 1
		#xgrids, ygrids size : (B, 1, sam_size)
		xgrids = torch.clamp(xgrids, min=-1, max=1).transpose(1, 2)
		ygrids = torch.clamp(ygrids, min=-1, max=1).transpose(1, 2)
		one_vector = torch.ones_like(xgrids)
		grid_x = torch.matmul(one_vector, xgrids.transpose(1, 2)).unsqueeze(-1)
		grid_y = torch.matmul(ygrids, one_vector.transpose(1, 2)).unsqueeze(-1)
		#grid size: (B, sam_size, sam_size, 2)
		grid = torch.cat((grid_x, grid_y), 3)
		structure_data = F.grid_sample(data, grid, align_corners=True)

		return structure_data

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
		x_sampled_zoom = self.adaptive_sample(input_x, att_map.squeeze(1))

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

		return x_sampled_zoom, att_map,

	def forward(self, input_x, p=None, lbl=None):

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
			x_concat = torch.cat([self.avg(feature_raw).view(-1, 2048),
			                      self.avg(feature_D).view(-1, 2048),
			                      self.avg(feature_C).view(-1, 2048)], 1)
			if self.droprate:
				x_concat = F.dropout(x_concat, p=self.droprate, training=self.training)
			aggregation = self.con_classifier(x_concat)
			#aggregation = self.con_classifier(torch.cat(
			#	[self.avg(feature_raw).view(-1, 2048), self.avg(feature_D).view(-1, 2048),
			#	 self.avg(feature_C).view(-1, 2048)], 1))

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
def masnet2(mode, num_classes, crit, input_size=448, base_ratio=0.09, radius=0.09,
        radius_inv=0.3, args=None):
	""" Selective sparse sampling.
	"""
	if mode != 'masnet2':
		raise NotImplementedError

	classify_network = ft_resnet(mode='resnet50', num_classes=num_classes,
	                             sync_bn=args['sync_bn'])
	model = MASNET2(classify_network, num_classes, input_size, base_ratio,
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