import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .hamburger import HamBurger
from lib.transformer.transformer_predictor import MLP


class SobelNorm(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        kernel = torch.stack([sobel_x, sobel_y]).unsqueeze(1)  # (2,1,3,3)
        self.weight = nn.Parameter(kernel.repeat(in_channels, 1, 1, 1), requires_grad=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 降到64维

    def forward(self, x):
       # x: (B, C, H, W)
        x = F.conv2d(x, self.weight, padding=1, groups=self.in_channels)  # (B, 2*C, H, W)
        x = torch.norm(x, dim=1, keepdim=True)  # (B, 1, H, W)
        x = x.repeat(1, self.in_channels, 1, 1)  # (B, C, H, W)
        x = self.bn(x)
        x = self.conv1x1(x)  # (B, out_channels, H, W)
        return x


class BilateralFilter(nn.Module):  # 滤波
    def __init__(self, channels):
        super().__init__()
        # 简单实现，实际可换更复杂实现
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        nn.init.constant_(self.conv.weight, 1 / 9)
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)



class FrequencyRefineModule(nn.Module):
    def __init__(self, in_channels1=64, in_channels2=192, in_channels3=64, out_channels=64, spatial_kernel=7, generate_mask=True, mask_channels=3):
        super().__init__()
        self.generate_mask = generate_mask
        self.sobel_norm = SobelNorm(in_channels1)
        self.conv1x3 = nn.Conv2d(in_channels2, in_channels2, (1, 3), padding=(0, 1))
        self.conv3x1 = nn.Conv2d(in_channels2, in_channels2, (3, 1), padding=(1, 0))
        self.conv1x1_fh = nn.Conv2d(in_channels1, out_channels, 1)
        self.bilateral = BilateralFilter(out_channels)

        self.conv3x3 = nn.Conv2d(in_channels3, in_channels3, 3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels3, in_channels3, 5, padding=2)
        self.conv1x1_fl = nn.Conv2d(in_channels1 + in_channels3, out_channels, 1)

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
          
            nn.Conv2d(in_channels1, in_channels1 // 16, 1, bias=False),
          
            nn.ReLU(inplace=True),
           
            nn.Conv2d(in_channels1 // 16, in_channels1, 1, bias=False)
        )

    

        self.sigmoid = nn.Sigmoid()

        self.Double_ConvBnRule = Double_ConvBnRule(64)
        self.conv_mask = nn.Conv2d(in_channels=in_channels1, out_channels=64, kernel_size=3, padding=1, bias=True)
        self.sm = nn.Softmax2d()
        self.conv_l = nn.Conv2d(192, 64, 1)
        self.conv_ll = nn.Conv2d(64, 64, 1)
        self.seq = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)       # ... 保持原有的初始化代码不变 ...
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
        self.conv_mask = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True)

    def forward(self, x, high, freq):  # freq是64维，high192
        Fx = x
        
        high = self.conv_l(high)
        high = torch.cat([high, freq], dim=1)
        high = self.conv1x1_fl(high)
        high = F.interpolate(high, x.size()[2:], mode='bilinear', align_corners=True)
       
        high = self.conv1x1_fh(high)
        high = self.Double_ConvBnRule(high)
        
        high_max = self.max_pool(high)
        high_avg = self.avg_pool(high)
        high_max_out = self.mlp(high_max)
        high_avg_out = self.mlp(high_avg)
        high_channel_out = self.sigmoid(high_max_out + high_avg_out)
        high_x = high_channel_out * high

        high_max_out, _ = torch.max(high_x, dim=1, keepdim=True)  # 返回(max_value, max_index)
        high_mean_out = torch.mean(high_x, dim=1, keepdim=True)

        max_pool_x = self.max_pool(x)  # 重命名避免混淆
        avg_pool_x = self.avg_pool(x)
        
        high_max_combined = high_avg + max_pool_x
        high_avg_combined = high_max+ avg_pool_x
        high_max_out1 = self.mlp(high_max_combined)
        high_avg_out1 = self.mlp(high_avg_combined)
        high_channel_out1 = self.sigmoid(high_max_out1 + high_avg_out1)
        high_x_combined = high_channel_out1 * x
        
      
        max_out_high, _ = torch.max(high_x_combined, dim=1, keepdim=True)
        avg_out_high = torch.mean(high_x_combined, dim=1, keepdim=True)
        
        # 修正：直接相加，不要使用元组解包
        high_max_combined_out = high_mean_out + max_out_high
        high_mean_combined_out = high_max_out + avg_out_high
        
        high_spatial_out = self.sigmoid(self.conv(torch.cat([high_max_combined_out, high_mean_combined_out], dim=1)))
        high_xx = high_spatial_out * x

    
        
        out = high_xx
        out = self.seq(out)
        mask = self.mask_branch(out) if self.generate_mask and self.mask_branch is not None else None
        return out, mask




class DoubleCBR(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleCBR, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.seq(x)


class GlobalBoundaryFeature(nn.Module):
    def __init__(self, channels_list=[64, 128, 320, 512], out_channels=64):
        super(GlobalBoundaryFeature, self).__init__()
        # 为每个尺度创建独立的DoubleCBR，保持输出通道一致
        self.double_cbrs = nn.ModuleList([
            DoubleCBR(in_channels, out_channels)
            for in_channels in channels_list
        ])
        self.conv1x1 = nn.Conv2d(out_channels * len(channels_list), out_channels, kernel_size=1)

    def forward(self, xm_list):
        # 1. 经过DoubleCBR得到边界敏感特征
        fb_list = [cbr(xm) for cbr, xm in zip(self.double_cbrs, xm_list)]
        # 2. 上采样到最高分辨率
        target_size = fb_list[0].shape[2:]
        fb_upsampled = [
            F.interpolate(fb, size=target_size, mode='bilinear', align_corners=False)
            if fb.shape[2:] != target_size else fb
            for fb in fb_list
        ]
        # 3. 拼接 + 1x1卷积
        cat_fb = torch.cat(fb_upsampled, dim=1)  # [B, 32*4, H, W]
        fb_g = self.conv1x1(cat_fb)  # [B, 32, H, W]
        return fb_g


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)

    def forward(self, x):
        # Global Average Pooling
        b, c, h, w = x.size()
        y = F.adaptive_avg_pool2d(x, 1)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y

class AHFA(nn.Module):
    def __init__(self, in_channels1=64, in_channels2=64, in_channels3=128, out_channels=128, hamburger_config=None):
        super(AHFA, self).__init__()
        # Transform input features to the same size (1x1 conv + upsample)
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels3, in_channels3, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(in_channels3),
                nn.ReLU(inplace=True)
            ) for _ in range(2)  # 最多支持4倍下采样(2^2)
        ])
        self.conv1x1_up = nn.Sequential(
            nn.Conv2d(in_channels=in_channels1 + in_channels3, out_channels=in_channels2, kernel_size=1),
            nn.Upsample(scale_factor=1, mode='nearest')  # No upsampling if already same size
        )
        self.conv1x2_up = nn.Sequential(
            nn.Conv2d(in_channels1, in_channels2, kernel_size=1),
            nn.BatchNorm2d(in_channels2),
            nn.ReLU(inplace=True),
        )
        # Concatenation will double the channel number
        self.conv3x3_branch = nn.Sequential(
            nn.Conv2d(in_channels2, in_channels2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels2, in_channels2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels2),
            nn.ReLU(inplace=True),
        )
        self.conv1x1_branch = nn.Sequential(
            nn.Conv2d(in_channels2, in_channels2, kernel_size=1),
            nn.BatchNorm2d(in_channels2),
            nn.ReLU(inplace=True),
        )
        if hamburger_config is None:
            hamburger_config = {
                'put_cheese': True,
                'MD_D': out_channels,
                'SPATIAL': True,
                'MD_S': 1,
                'MD_R': 4,
                'TRAIN_STEPS': 3,
                'EVAL_STEPS': 3,
                'INV_T': 1,
                'Eta': 0.1,
                'RAND_INIT': False,
            }
        self.hamburger = HamBurger(in_channels2, hamburger_config)
        # For recalibrating with global context
        self.reduce_channels = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1x1_1 = nn.Conv2d(in_channels1, in_channels1, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(in_channels2, in_channels2, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.ss = nn.Conv2d(64, 128, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.global_pool = nn.AdaptiveMaxPool2d(1)
        self.SqueezeExcitation =  SqueezeExcitation(64)
    def forward(self, fp5_i, boundary_feature):
        # Step 1: Align boundary_feature to fp5_i if needed
        if boundary_feature.shape[2:] != fp5_i.shape[2:]:
            scale_factor = boundary_feature.shape[2] // fp5_i.shape[2]
            if scale_factor > 1:
                required_downsamples = int(math.log2(scale_factor))
                for i in range(required_downsamples):
                    boundary_feature = self.downsample_layers[i](boundary_feature)
    
        # Step 2: Feature fusion
        x = torch.cat([boundary_feature, fp5_i], dim=1)
        x = self.conv1x1_up(x)
    


        
        # x1= fp5_or_di_1
        # x = fp5_i
        # xand = x1 + x
        # xsub = x1 - x
        # xand1 = self.gap(xand)
        # # print(xand1.size())
        # xand1 = self.conv1x1_1(xand1)
        # xand1 = self.relu(xand1)
        # xand1 = self.conv1x1_1(xand1)
        
        # xand3 = self.conv1x1_1(x)
        # xand3 = self.relu(xand3)
        # xand3 = self.conv1x1_1(xand3)        
        # xand1 = xand1 +xand3
        # xand2 = self.global_pool(xsub)
        # xand2 = self.conv1x1_1(xand2)
        # xand2 = self.relu(xand2)
        # xand2 = self.conv1x1_1(xand2)
        # xand4 = self.conv1x1_1(x)
        # xand4 = self.relu(xand4)
        # xand4 = self.conv1x1_1(xand4)
        
        # xand2 = xand4 +xand2
        
        # xand = xand2 + xand1
        # xand = self.sigmoid(xand)
        # x1 = xand*x1
        # x = xand*x 


        # x = x1 + x
       
        # x = self.SqueezeExcitation(x)

        context = self.conv1x1_2(x)
        context = self.conv1x1_2(context)
       
        return context


class DSEBlock(nn.Module):  # 处理低频信息
    def __init__(self, in_channels=64, out_channels=64):
        super(DSEBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=5,
            padding=2
        )
        self.resconv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.bn1(feat)
        feat = self.relu1(feat)
        feat = self.conv2(feat)
        feat = self.bn1(feat) + self.resconv(x)
        feat = self.relu1(feat)

        return feat


class Double_ConvBnRule(nn.Module):

    def __init__(self, in_channels, out_channels=64):
        super(Double_ConvBnRule, self).__init__()

        # VC: 标准3x3卷积
        self.vc = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        # CPDC-HV: 水平/垂直差分
        self.pdc_hv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, groups=in_channels)
        # CPDC-DG: 对角差分
        self.pdc_dg = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, groups=in_channels)
        # APDC: 角度差分
        self.apdc = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, groups=in_channels)
        self._init_pdc_kernels()

        # 输出通道调整为64
        self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # 残差分支
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def _init_pdc_kernels(self):
        # 初始化PDC卷积核为差分模板
        # CPDC-HV
        hv_kernel = torch.zeros((1, 1, 3, 3))
        hv_kernel[0, 0, 1, 0] = 1  # 左
        hv_kernel[0, 0, 1, 2] = -1  # 右
        hv_kernel[0, 0, 0, 1] = 1  # 上
        hv_kernel[0, 0, 2, 1] = -1  # 下
        hv_kernel = hv_kernel.repeat(self.pdc_hv.in_channels, 1, 1, 1)
        self.pdc_hv.weight.data.copy_(hv_kernel)

        # CPDC-DG
        dg_kernel = torch.zeros((1, 1, 3, 3))
        dg_kernel[0, 0, 0, 0] = 1  # 左上
        dg_kernel[0, 0, 2, 2] = -1  # 右下
        dg_kernel[0, 0, 0, 2] = 1  # 右上
        dg_kernel[0, 0, 2, 0] = -1  # 左下
        dg_kernel = dg_kernel.repeat(self.pdc_dg.in_channels, 1, 1, 1)
        self.pdc_dg.weight.data.copy_(dg_kernel)

        # APDC
        apdc_kernel = torch.zeros((1, 1, 3, 3))
        apdc_kernel[0, 0, 0, 1] = 1  # 上
        apdc_kernel[0, 0, 1, 2] = 1  # 右
        apdc_kernel[0, 0, 2, 1] = -1  # 下
        apdc_kernel[0, 0, 1, 0] = -1  # 左
        apdc_kernel = apdc_kernel.repeat(self.apdc.in_channels, 1, 1, 1)
        self.apdc.weight.data.copy_(apdc_kernel)

       
        self.resconv = nn.Sequential(nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64))


    def forward(self, x):
        out_vc = self.vc(x)
        out_hv = self.pdc_hv(x)
        out_dg = self.pdc_dg(x)
        out_ap = self.apdc(x)
        # 四分支相加并激活
        out = F.relu(out_vc + out_hv + out_dg + out_ap)
        out = self.out_conv(out)  # 输出64通道
        res = self.res_conv(x)  # 残差分支
        out = out + res  # 残差连接
    
        return out


class Double_ConvBnRule_CBAM(nn.Module):  # 这是处理的高频信息

    def __init__(self, in_channels, out_channels=64):
        super(Double_ConvBnRule_CBAM, self).__init__()
        self.channel_adjust = nn.Conv2d(
            out_channels * 2,  # 因为 cat([feat, feat1]) 会变成 128 通道
            out_channels,  # 降回 64 通道
            kernel_size=1
        )
        self.conv1_1x3 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 3),  # 1x3卷积
            padding=(0, 1)  # 只在宽度方向填充
        )
        self.conv1_3x1 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 1),  # 3x1卷积
            padding=(1, 0)  # 只在高度方向填充
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=False)

        self.conv2_1x5 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 5),  # 1x3卷积
            padding=(0, 2)  # 只在宽度方向填充
        )
        self.conv2_5x1 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(5, 1),  # 3x1卷积
            padding=(2, 0)  # 只在高度方向填充
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=False)
        self.resconv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        # self.cbam=CBAM(out_channels)

    def forward(self, x):
        feat = self.conv1_1x3(x)
        feat = self.conv1_3x1(feat)
        feat = self.bn1(feat)
        feat = self.relu1(feat)

        feat1 = self.conv2_1x5(x)
        feat1 = self.conv2_5x1(feat1)
        feat1 = self.bn2(feat1)
        # feat = self.cbam(feat)+self.resconv(x)
        feat = self.relu2(feat1)
        out = torch.cat([feat, feat1], dim=1)
        out = self.channel_adjust(out)
        return out


def fHb(freq=4, lf_mask=0, hf_mask=0):
    freq = np.uint8(freq)
    if freq > 8: freq = 8
    f = torch.zeros(1, 64, 1, 1) + torch.tensor([1 - hf_mask])
    for i in range(freq):
        for j in range(freq):
            f[0, i + j * 8, 0, 0] = lf_mask
    f[0, 0, 0, 0] = 0

    return torch.bernoulli(f.repeat(1, 3, 1, 1))


def fLb(freq=4, lf_mask=0, hf_mask=0):
    freq = np.uint8(freq)
    if freq > 8: freq = 8
    f = torch.zeros(1, 64, 1, 1) + torch.tensor([hf_mask])
    for i in range(freq):
        for j in range(freq):
            f[0, i + j * 8, 0, 0] = 1 - lf_mask
    f[0, 0, 0, 0] = 1

    return torch.bernoulli(f.repeat(1, 3, 1, 1))


class HGAM(nn.Module):
    def __init__(self, dim_spa, dim_freq, dim_high, dropout_rate=0.1):
        super(HGAM, self).__init__()
        # 主卷积层（添加Dropout）
        self.conv = nn.Sequential(
            nn.Conv2d(dim_spa + dim_freq + dim_high, dim_spa, 3, padding=1),
            nn.BatchNorm2d(dim_spa),
            nn.ReLU(inplace=False),
            # nn.Dropout2d(dropout_rate)  # 添加2D Dropout
        )

        # 频率分支处理（Double_ConvBnRule_CBAM假设已定义）
        self.freq_conv = Double_ConvBnRule_CBAM(192, 64)
        self.cbam = CBAM(64)

    def forward(self, spa, freq, high):
        # print('11111111',freq.size())
        # 处理频率分支
        freq = self.freq_conv(freq)  # [16, 64, 44, 44]
        freq = F.interpolate(freq, spa.size()[2:], mode='bilinear', align_corners=True)
        high = F.interpolate(high, spa.size()[2:], mode='bilinear', align_corners=True)

        out = self.conv(torch.cat([spa, freq, high], dim=1))  # [8, 64, 88, 88]

        out = self.cbam(out)

        return out


class EdgeAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # 固定Sobel核
        self.register_buffer("sobel_x",
                             torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer("sobel_y",
                             torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))

        # 提取边缘特征
        self.edge_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        b, c, h, w = x.size()

        # Sobel边缘提取（逐通道）
        edge = []
        for i in range(c):
            xi = x[:, i:i + 1]
            gx = F.conv2d(xi, self.sobel_x, padding=1)
            gy = F.conv2d(xi, self.sobel_y, padding=1)
            edge_map = (gx.abs() + gy.abs())  # 梯度幅值
            edge.append(edge_map)
        edge = torch.cat(edge, dim=1) * 0.5

        # 提取边缘特征
        edge_feat = self.edge_enhance(edge)

        # 直接返回增强后的边缘特征
        return edge_feat


class EMA(nn.Module):
    def __init__(self, channels, factor=8):  # 默认调整为8组
        super(EMA, self).__init__()
        self.groups = max(min(factor, channels), 1)  # 动态限制组数范围
        assert channels % self.groups == 0, f"Channels {channels} must be divisible by groups {self.groups}"

        # 空间注意力分支
        self.spatial_conv = nn.Conv2d(channels // self.groups, 1, kernel_size=1)

        # 通道注意力分支
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Conv2d(channels, channels // self.groups, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // self.groups, channels, kernel_size=1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.view(b * self.groups, -1, h, w)

        # 空间注意力
        spatial_weights = self.sigmoid(self.spatial_conv(group_x))  # [B*g, 1, H, W]

        # 通道注意力
        channel_weights = self.sigmoid(self.channel_fc(self.gap(x)))  # [B, C, 1, 1]
        channel_weights = channel_weights.view(b, self.groups, -1, 1, 1).expand(-1, -1, -1, h, w)
        channel_weights = channel_weights.reshape(b * self.groups, -1, h, w)

        # 融合权重
        weights = spatial_weights * channel_weights
        return (group_x * weights).view(b, c, h, w)


# 定义融合后的 RCA 模块
class RCA_Module_with_EMA_and_Mask(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, attention=True):
        super(RCA_Module_with_EMA_and_Mask, self).__init__()

        self.out_channels = out_channels  # 输出通道数

        # 第一层卷积 + 批归一化 + ReLU 激活
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 第二层卷积 + 批归一化 + ReLU 激活
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 残差连接，如果输入和输出的通道不一样，使用1x1卷积调整
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                                  padding=0) if in_channels != out_channels else nn.Identity()

        # 掩码生成模块（模仿 DRSM）
        self.conv_mask = nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=3, padding=1, bias=True)
        self.sm = nn.Softmax2d()
        self.conv = nn.Conv2d(in_channels=3072, out_channels=64, kernel_size=1, stride=1, padding=0)
        # EMA 注意力机制
        self.attention = attention
        if self.attention:
            self.ema = EMA(out_channels)
        self.edga = EdgeAttention(out_channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1),  # 1x1卷积压缩通道

        )

    def forward(self, x):
        b, c, h, w = x.size()

        # 生成掩码
        masks = self.conv_mask(x)
        # print('masks', masks.size())
        masks = self.sm(masks)  # Softmax 对掩码进行归一化
        # print('masks=self.sm(masks)', masks.size())  # [1, 3, 176, 176]#[1, 3, 88, 88]
        masks0 = masks
        # 通过卷积生成掩码后，进行掩码加权
        # 掩码加权

        # 进行卷积操作
        residual = self.shortcut(x)  # 残差连接

        out = F.relu(self.bn1(self.conv1(x)))  # 第一步卷积 + 激活
        # print('outoutoutoutout',out.size())
        out = self.bn2(self.conv2(out))  # 第二步卷积

        y = self.edga(out)
        # 如果启用了 EMA 注意力机制，进行注意力加权
        if self.attention:
            out = self.ema(out)

        # 将残差添加到输出中

        fused = torch.cat([out, y], dim=1)
        out = self.fusion(fused)
        out += residual
        return out, masks0


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=False,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
        # torch.Size([1, 64, 44, 44])


class QueryGeneration(nn.Module):
    def __init__(self, query_num, embedding_dim) -> None:
        super().__init__()
        self.emb = nn.Embedding(44 ** 2, embedding_dim)

        self.conv_fl = Double_ConvBnRule(64, query_num * 4)
        self.mlp = MLP(embedding_dim, embedding_dim * 2, embedding_dim, 2)
        self.query_num = query_num

    def forward(self, feat_l):
        feat_l = self.conv_fl(feat_l)
        feat_l = feat_l.flatten(2)
        query = torch.einsum('bnm, mc -> bnc', feat_l, self.emb.weight)
        query = self.mlp(query)

        return query

