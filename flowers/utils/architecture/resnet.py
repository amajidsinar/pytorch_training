import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2)
        self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=2)
    def forward(self, x):
        a = F.relu(self.conv1(x))
        a = self.max_pool2d(a)
        return a

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        
    def forward(self, x):
        a = self.conv1(x)
        a = F.relu(self.bn1(a))
        a = self.conv2(a)
        a = F.relu(self.bn2(a))
        
        a += x
        a = F.relu(a)
        return a

class ResNet16(nn.Module):
    def __init__(self):
        super(ResNet16, self).__init__()
        # 1st block
        self.block_1 = BasicBlock()
        # 2nd block
        self.block_2 = self._make_layer(in_channels=64, out_channels=64, kernel_size=3, num_layers=2)
        # 3rd block
        self.block_3 = self._make_layer(in_channels=64, out_channels=128, kernel_size=3, num_layers=2)
        # 4th block
        self.block_4 = self._make_layer(in_channels=128, out_channels=256, kernel_size=3, num_layers=2)
        # 5th block
        self.block_5 = self._make_layer(in_channels=256, out_channels=512, kernel_size=3, num_layers=2)
        
    def _make_layer(self, in_channels, out_channels, kernel_size, num_layers, stride=1, ):
        block = []
        block.append(ResidualBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size))
        for i in range(1, num_layers):
            block.append(ResidualBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size))

        return nn.Sequential(*block)
    def forward(self, x):
        a = self.block_1(x)
        a = self.block_2(a)
        a = self.block_3(a)
        a = self.block_4(a)
        a = self.block_5(a)

        return a 
    
    