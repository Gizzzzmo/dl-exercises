import torch
from torch import nn
from torch.functional import F
from functools import reduce
from torchvision.models import resnext50_32x4d, resnext101_32x8d, resnet18
from itertools import chain
import copy

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 5, stride, padding=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skipconv = nn.Conv2d(in_channels, out_channels, 1, stride)
        self.skipbn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        return y + self.skipbn(self.skipconv(x))

class NextSubBlock(nn.Module):
    def __init__(self, channels, bottleneck):
        super(NextSubBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, bottleneck, 1)
        self.bn1 = nn.BatchNorm2d(bottleneck)
        self.conv2 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(bottleneck)
        self.conv3 = nn.Conv2d(bottleneck, channels, 1)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return x

class NextBlock(nn.Module):
    def __init__(self, channels, cardinality):
        super(NextBlock, self).__init__()
        self.parallel_blocks = nn.ModuleList(NextSubBlock(channels, 128//cardinality) for i in range(cardinality))

    def forward(self, x):
        return F.relu(reduce(lambda a, b: a + b(x), [x] + list(self.parallel_blocks)))

class ResNext(torch.nn.Module):
    def __init__(self):
        super(ResNext, self).__init__()
        self.conv = nn.Conv2d(3, 64, 7, 2)
        self.bn = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(3, 2)
        self.res1 = ResBlock(64, 64, 1)
        self.next1 = NextBlock(64, 32)
        self.next2 = NextBlock(64, 32)
        self.res2 = ResBlock(64, 128, 2)
        self.next3 = NextBlock(128, 32)
        self.next4 = NextBlock(128, 32)
        self.res5 = ResBlock(128, 256, 2)
        self.next5 = NextBlock(256, 32)
        self.next6 = NextBlock(256, 32)
        self.fc = nn.Linear(256, 1)

    def new_params(self):
        return self.parameters()

    def forward(self, x):
        #x = x[:, :, 15:-15, 15:-15]
        x = self.max_pool(F.relu(self.bn(self.conv(x))))
        x = self.next2(self.next1(self.res1(x)))
        x = self.next4(self.next3(self.res2(x)))
        x = self.next6(self.next5(self.res5(x)))
        x = x.mean(dim=(2, 3))

        x = x.view(x.shape[0], -1)

        return self.fc(x)

class ResNet(torch.nn.Module):
    def __init__(self, capacity_factor=1):
        super(ResNet, self).__init__()
        self.conv = nn.Conv2d(3, int(64*capacity_factor), 7, 2)
        self.bn = nn.BatchNorm2d(int(64*capacity_factor))
        self.max_pool = nn.MaxPool2d(3, 2)
        self.res1 = ResBlock(int(64*capacity_factor), int(64*capacity_factor), 1)
        self.res2 = ResBlock(int(64*capacity_factor), int(128*capacity_factor), 2)
        self.res4 = ResBlock(int(128*capacity_factor), int(256*capacity_factor), 2)
        self.res5 = ResBlock(int(256*capacity_factor), int(512*capacity_factor), 2)
        self.fc = nn.Linear(int(512*capacity_factor), 1)

    def new_params(self):
        return self.parameters()

    def forward(self, x):
        x = self.max_pool(F.relu(self.bn(self.conv(x))))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res4(x)
        x = self.res5(x)
        x = x.mean(dim=(2, 3))

        x = x.view(x.shape[0], -1)

        return self.fc(x)

class OrigResNet(nn.Module):
    def __init__(self, pretrained=False):
        super(OrigResNet, self).__init__()
        self.rn = resnet18(pretrained)
        self.rn.fc = nn.Linear(512, 1)

    def new_params(self):
        return self.parameters()

    def forward(self, x):
        #print(x.shape)
        return torch.sigmoid(self.rn(x))

class OrigResNext(nn.Module):
    def __init__(self, pretrained=False, big=False):
        super(OrigResNext, self).__init__()
        self.rn = resnext101_32x8d(pretrained) if big else resnext50_32x4d(pretrained)
        self.rn.fc = nn.Linear(2048, 1)

    def new_params(self):
        return self.parameters()

    def forward(self, x):
        # x = x[:, :, 15:-15, 15:-15]
        return torch.sigmoid(self.rn(x))

class SpecialModel(nn.Module):
    def __init__(self, pretrained=True):
        super(SpecialModel, self).__init__()
        rn = resnext50_32x4d(pretrained)
        #self.rn.fc = nn.Identity()
        self.conv1 = rn.conv1
        self.bn1 = rn.bn1
        self.relu = rn.relu
        self.maxpool = rn.maxpool
        self.layer1 = rn.layer1
        self.layer2 = rn.layer2
        self.layer3 = rn.layer3
        self.layer3[0].conv2 = nn.Conv2d(512, 512, 3, 1, bias=False, padding=1, groups=32)
        self.layer3[0].conv2.weight = rn.layer3[0].conv2.weight
        self.layer3[0].conv3 = nn.Conv2d(512, 1024, 1, 1, dilation=2, bias=False)
        self.layer3[0].conv3.weight = rn.layer3[0].conv3.weight
        self.layer3[0].downsample[0] = nn.Conv2d(512, 1024, 1, 1, bias=False)
        self.layer3[0].downsample[0].weight = rn.layer3[0].downsample[0].weight

        self.final = nn.Conv2d(1024, 1, 1, 1)
        self.avgpool = rn.avgpool


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.final(x)
        x = self.avgpool(x).sum(dim=(2, 3))
        return x

    def new_params(self):
        return self.parameters()

class RegNet(nn.Module):
    def __init__(self, inn):
        super(RegNet, self).__init__()
        self.fc1 = nn.Linear(inn, inn)
        self.fc2 = nn.Linear(inn, 1)
    
    def forward(self, x):
        x = torch.sigmoid(x)
        x = self.fc1(x)
        x = F.relu(x)
        return self.fc2(x)