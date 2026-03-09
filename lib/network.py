import torch
import torchvision.models as models
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from lib.transformer.ADConv import ADConv
from lib.blocks import *
from lib.pvtv2 import pvt_v2_b2
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os
import numpy as np


import torch
import torch.nn as nn


   

class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, hidden_channels=None, generate_mask=True, mask_channels=3):
        super(ResidualBlock, self).__init__()
        self.generate_mask = generate_mask
        if hidden_channels is None:
            hidden_channels = out_channels
        
        # 主分支卷积层 - 增加了隐藏层
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # 捷径连接
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # 掩码生成分支 - 也增加了隐藏层
        if generate_mask:
            self.mask_branch = nn.Sequential(
                nn.Conv2d(out_channels, out_channels//2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels//2, out_channels//4, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels//4, mask_channels, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
        else:
            self.mask_branch = None
        
        self.final_relu = nn.ReLU(inplace=True)
        self.conv_mask = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1, bias=True)
     
    def forward(self, x):
        # 确保输入是张量而不是元组
        if isinstance(x, tuple):
            x = x[0]  # 如果是元组，取第一个元素（特征图）
       
        
        residual = self.shortcut(x)
        out = self.conv_layers(x)
        
        # 残差连接
        out += residual
        out = self.final_relu(out)
        #mask = self.mask_branch(out) if self.generate_mask and self.mask_branch is not None else None
        
        return out




class UpSampleBlock(nn.Module):
   
    def __init__(self, in_channels, out_channels):
        super(UpSampleBlock, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.res_block = ResidualBlock(in_channels, out_channels)
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.res_block(x)
        return x


class DSHNet(nn.Module):
    def __init__( self):
        super(DSHNet, self).__init__()

        self.backbone = pvt_v2_b2()

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.channel_layer1 = Double_ConvBnRule(64)
        self.channel_layer2 = Double_ConvBnRule(128)
        self.channel_layer3 = Double_ConvBnRule(320)
        self.channel_layer4 = Double_ConvBnRule(512)
        self.AHFAModule = AHFA(64,64,64,128)
      
        self.channel_layer = Double_ConvBnRule(192)
    
        self.global_boundary = GlobalBoundaryFeature(channels_list=[64, 128, 320, 512], out_channels=64)

        
     
        self.bottom_block = nn.Sequential(
            ResidualBlock(64, 64, hidden_channels=128),  # 增加隐藏层通道数
            ResidualBlock(64, 64, hidden_channels=128)
        )

        self.merge4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),  # 合并后通道调整
            ResidualBlock(64, 64)
        )
        self.merge3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            ResidualBlock(64, 64)
        )
        
        # 第二层 (64,176,176)
        self.up2 = UpSampleBlock(64, 64)
        self.merge2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            ResidualBlock(64, 64)
        )
        
        # 第一层 (64,325,325)
        self.up1 = nn.Upsample(size=(325,325), mode='bilinear', align_corners=True)
        self.merge1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            ResidualBlock(64, 64)
        )

        self.skip_connect_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )

        self.skip_connect_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )

        self.skip_connect_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )

        self.DSEBlock = DSEBlock(64)
        self.predict_layer_4 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, bias=True)
        self.predict_layer_3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, bias=True)
        self.predict_layer_2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, bias=True)
        self.predict_layer_final = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, bias=True)

        self.HGAM1 = FrequencyRefineModule(64, 192, 64)
        self.HGAM2 = FrequencyRefineModule(64, 192, 64)
        self.HGAM3 = FrequencyRefineModule(64, 192, 64)
        self.HGAM4 = FrequencyRefineModule(64, 192, 64)
      
        self.conv_l = nn.Conv2d(192, 64, 1)
        self.channel_layera = nn.Conv2d(64,64,kernel_size=1)
        self.channel_layerb = nn.Conv2d(128,64,kernel_size=1)
        self.channel_layerc = nn.Conv2d(320,64,kernel_size=1)
        self.channel_layerd = nn.Conv2d(512,64,kernel_size=1)

        if self.training:
            self.initialize_weights()

    def forward(self, x, x_DCT=None):

        # print('x',x.size())
        x1, x2, x3, x4 = self.backbone(x)
       
        boundary_feature = self.global_boundary([x1, x2, x3, x4])
        x1_t = self.channel_layera(x1)
        x2_t = self.channel_layerb(x2)
        x3_t = self.channel_layerc(x3)
        x4_t = self.channel_layerd(x4)
   

  
        x4_t = self.upsample2(x4_t)#22
        x4_u = self.bottom_block(x4_t)#22
        mid_pred_4 = self.predict_layer_4(x4_u)
        c4_u = self.AHFAModule(x3_t,boundary_feature)
        x4_u = self.skip_connect_conv4(c4_u)  
     
        
        x4_u = self.upsample2(x4_u)  
        x3_u = self.merge4(x4_u)
        mid_pred_3 = self.predict_layer_3(x3_u)
        c3_u = self.AHFAModule( x2_t,boundary_feature)  # 128,44,44
       
        x3_u = self.skip_connect_conv4(c3_u)   
        x3_u = self.upsample2(x3_u)
        x2_u = self.merge3(x3_u)
        mid_pred_2 = self.predict_layer_2(x2_u)
    
        c2_u = self.AHFAModule(x1_t,boundary_feature)
        x2_u = self.skip_connect_conv4(c2_u) 

        x2_u = self.upsample2(x2_u)
  
        x1_u = self.merge2(x2_u)

        x1_u = self.upsample2(x1_u)

        pred = self.predict_layer_final(x1_u)#最终的损失

      #  masks = [x4_mask, x3_mask, x2_mask, x1_mask]
        mid_preds = [mid_pred_4, mid_pred_3, mid_pred_2]

        return pred,  mid_preds

    def initialize_weights(self):
        path = './pretrain/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

