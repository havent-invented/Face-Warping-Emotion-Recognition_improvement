"""
Defines models
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

from torchvision.models.video import r3d_18, r2plus1d_18, swin3d_s, swin3d_t, mvit_v2_s

def set_requires_grad(model, requires_grad=True):
	for param in model.parameters():
		param.requires_grad = requires_grad

def init_weights(m):
	if type(m) == nn.Linear or type(m) == nn.Conv2d:
		torch.nn.init.xavier_uniform_(m.weight)
		if m.bias is not None:
			m.bias.data.fill_(0.01)

class Flatten(nn.Module):
	def forward(self, input):
		
		return input.view(input.size(0), -1)

class Emotion_Recognizer(nn.Module):
	def __init__(self, opt):
		super(Emotion_Recognizer, self).__init__()

		self.encoder = Encoder(opt)
		self.decoder = Decoder(opt)

	def forward(self, images):
		features = self.encoder(images)
		labels = self.decoder(features)

		return labels

	def get_parameters(self) -> List[Dict]:
		params = [
			{"params": self.encoder.parameters(), "lr_mult": 1.},
			{"params": self.decoder.parameters(), "lr_mult": 1.}
		]
		return params

class Emotion_Recognizer_Fine_Tune(nn.Module):
	def __init__(self, opt):
		super(Emotion_Recognizer_Fine_Tune, self).__init__()

		self.lr_mult = opt.lr_mult
		self.encoder = Encoder(opt)
		self.decoder = Decoder(opt)

	def forward(self, images):
		features = self.encoder(images)
		labels = self.decoder(features)

		return labels

	def get_parameters(self) -> List[Dict]:
		params = [
			{"params": self.encoder.parameters(), "lr_mult": self.lr_mult},
			{"params": self.decoder.parameters(), "lr_mult": 1.}
		]
		return params

class Encoder(nn.Module):
	def __init__(self, opt):
		super(Encoder, self).__init__()
		self.stock_type = (opt.backbone == 'r3d' or opt.backbone == 'r21d' or opt.backbone == 'resnet18')
		fc_in_sz = 512
		
		self.use_OF = opt.use_OF

		if self.use_OF:
			from FastFlowNet.models.FastFlowNet_v2 import FastFlowNet
			self.OF_model = FastFlowNet().eval().requires_grad_(False)


		if opt.backbone == 'r3d':
			r3d = r3d_18(pretrained=True)
			if opt.use_OF:
				r3d.stem[0] = nn.Conv3d(5, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
			r3d_layers = list(r3d.children())[:-1]
			
			self.model = nn.Sequential(*r3d_layers)
			
			
		elif opt.backbone == 'r21d':
			r21d = r2plus1d_18(pretrained=True)
			if opt.use_OF:
				r21d.stem[0] = nn.Conv3d(5, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
			r21d_layers = list(r21d.children())[:-1]
			
			self.model = nn.Sequential(*r21d_layers)
			

		elif opt.backbone == 'resnet18':
			resnet18 = torchvision.models.resnet18(pretrained=True)
			resnet18_layers = list(resnet18.children())[:-1]
			self.model = nn.Sequential(*resnet18_layers)

		elif opt.backbone == "swin3d_s":
			self.model = swin3d_s(pretrained=True)
			self.model.head = nn.Linear(768, 128, bias=True)
			
		elif opt.backbone == "swin3d_t":
			swin3d_t_model = swin3d_t(pretrained=True)
			#swin3d_t_layers = list(swin3d_t_model.children())[:-2]
			self.model = swin3d_t_model#nn.Sequential(*swin3d_t_layers)
			self.model.head = nn.Linear(768, 128, bias=True)

		elif opt.backbone == "mvit_v2_s":
			self.model == mvit_v2_s(pretrained=True)
			self.model.head = nn.Linear(768, 128, bias=True)
		
		if self.use_OF:
			if opt.backbone == "swin3d_s" or opt.backbone == "swin3d_t":
				self.model.patch_embed.proj = nn.Conv3d(5, 96, kernel_size=(2, 4, 4), stride=(2, 4, 4))
			if opt.backbone == "resnet18":
				self.model.conv1 = nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

				
		self.dropout_rate = opt.dropout_rate
		self.dropout = nn.Dropout(self.dropout_rate)
		self.flatten = Flatten()
		self.fc1 = nn.Linear(fc_in_sz, 256)
		self.fc2 = nn.Linear(256, 128)
		self.relu = nn.ReLU()

	def forward(self, x):
		#print(x.shape)
		if self.use_OF:
			X2 = torch.zeros([x.shape[0] * x.shape[2],   6,          x.shape[3], x.shape[4]], device = x.device)
			fr_len = x.shape[2]
			batchsz = x.shape[0]
			for i_ch in range(fr_len - 1):
				X2[i_ch*batchsz:(i_ch + 1) * batchsz, :3, ...] = x[:,:,i_ch, :, : ]
				X2[i_ch*batchsz:(i_ch + 1) * batchsz, 3:, ...] = x[:,:,i_ch+1, :, : ]
            
			X2 = torch.nn.functional.interpolate(X2, size=[128,128], mode='bilinear', align_corners=False)

            #resize OF to match x
			with torch.no_grad():
				self.OF_model.eval()
				OF = self.OF_model(X2).detach()
			OF = torch.nn.functional.interpolate(OF, size=x.shape[-2:], mode='bilinear', align_corners=False)

			OF_converted =  torch.zeros([x.shape[0], 2,fr_len, x.shape[-2], x.shape[-1]], device = x.device)

			for i_ch in range(fr_len - 1):
				OF_converted[:,:,i_ch, ...] = OF[i_ch*batchsz:(i_ch + 1) * batchsz, :, ...] 

			OF_converted[:,:,fr_len-1, ...] =  OF_converted[:,:,fr_len-2, ...]
		
			x = torch.cat([x, OF_converted], dim=1)
			#resize OF to match x

		x = self.model(x)

		

		if self.stock_type:
			#print(f"stock: {x.shape}")
			x = self.flatten(x)
			x = self.fc1(x)
			x = self.relu(x)
			x = self.fc2(x)
		#print(x.shape)
		x = F.normalize(x, p=2, dim=1)
		x = self.dropout(x)

		return x

class Decoder(nn.Module):
	def __init__(self, opt):
		super(Decoder, self).__init__()

		self.fc = nn.Linear(128, 1)
		self.dropout_rate = opt.dropout_rate
		self.dropout = nn.Dropout(self.dropout_rate)
		self.tanh = nn.Tanh()

	def forward(self, x):
		x = self.fc(x)
		x = self.dropout(x)
		x = self.tanh(x)

		return x
