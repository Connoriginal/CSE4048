import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os



class RobustModel(nn.Module):
    '''
        Baseline model : Simple CNN model
    '''
    def __init__(self):
        super(RobustModel, self).__init__()
        self.keep_prob = 0.5

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc3 = nn.Linear(4 * 4 * 32, 120, bias=True)
        nn.init.xavier_uniform_(self.fc3.weight)

        self.layer3 = nn.Sequential(
            self.fc3,
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob)
        )

        self.fc4 = nn.Linear(120, 80, bias=True)
        nn.init.xavier_uniform_(self.fc4.weight)
        self.layer4 = nn.Sequential(
            self.fc4,
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob)
        )
        self.fc5 = nn.Linear(80, 10, bias=True)
        nn.init.xavier_uniform_(self.fc5.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.fc5(out)
        return out

'''
    Implement ResNet
'''

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # residual block
        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        
        # shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * BasicBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
    
    def forward(self,x):
        return self.relu(self.residual_block(x) + self.shortcut(x))

class ResNet18(nn.Module):
    def __init__(self, block, num_block, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True) 
        )
        self.layer2 = self.make_layer(block, 64, num_block[0], stride=1)
        self.layer3 = self.make_layer(block, 128, num_block[1], stride=2)
        self.layer4 = self.make_layer(block, 256, num_block[2], stride=2)
        # self.layer5 = self.make_layer(block, 512, num_block[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)
        nn.init.xavier_uniform_(self.fc.weight)
    
    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = self.layer5(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out