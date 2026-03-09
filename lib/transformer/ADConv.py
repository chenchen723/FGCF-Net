import torch
import torch.nn as nn
import torch.nn.functional as F


class ADConv(nn.Module):
    def __init__(self, in_channels1=128, in_channels2=64, out_channels=12, kernel_size=3, 
                 num_kernels=4, groups=4, temperature=1.0):
        super(ADConv, self).__init__()
        self.in_channels1 = in_channels1
        self.in_channels2 = in_channels2
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.groups = groups
        self.temperature = temperature

        # 第一个输入路径的处理 (1, 128, 44, 44)
        self.conv1_1 = nn.Conv2d(in_channels1, out_channels, kernel_size=1)
        self.conv2_1 = nn.Conv2d(64, 256, kernel_size=3, padding=1)

        # 第二个输入路径的处理 (16, 64, 44, 44)
        self.conv1_2 = nn.Conv2d(in_channels2, out_channels, kernel_size=1)
        self.conv2_2 = nn.Conv2d(in_channels2, num_kernels * out_channels, kernel_size=1)  # 修改为 1x1
        
        # 共享的后续处理
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )
        
        # 最终卷积核生成层
        self.kernel_gen = nn.Conv2d(out_channels * 2, out_channels * num_kernels * kernel_size * kernel_size, 
                                   kernel_size=1)

    def forward(self, x1, x2):
        # 处理第一个输入 (1, 128, 44, 44)
        x1_avg = self.avgpool(x1)  # (1, 128, 1, 1)
        x1_fc = self.fc(self.conv1_1(x1_avg))  # (1, 12, 1, 1)
        A_i1 = torch.sigmoid(self.conv2_1(x1_fc) / self.temperature)
        
        # 处理第二个输入 (16, 64, 44, 44)
        x2_avg = self.avgpool(x2)  # (16, 64, 1, 1)
        x2_fc = self.fc(self.conv1_2(x2_avg))  # (16, 12, 1, 1)
        A_i2 = torch.sigmoid(self.conv2_2(x2_fc) / self.temperature)  # (16, 48, 1, 1)
        
        # 拼接两个路径的特征
        combined = torch.cat([x1_fc.expand(16, -1, -1, -1), x2_fc], dim=1)  # (16, 24, 1, 1)
        
        # 生成最终的卷积核
        kernels = self.kernel_gen(combined)  # (16, 12*4*3*3, 1, 1)
        kernels = kernels.view(16, 12, 4, 3, 3)  # 重塑为 (16, 12, 4, 3, 3)
        
        return kernels
