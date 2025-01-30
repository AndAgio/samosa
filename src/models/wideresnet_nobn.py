import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import EmbeddingRecorder
from torchvision.models import resnet


# Acknowledgement to
# https://github.com/xternalz/WideResNet-pytorch

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(x)
        else:
            out = self.relu1(x)
        out = self.relu2(self.conv1(out if self.equalInOut else x))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet_32x32(nn.Module):
    def __init__(self, depth, num_classes, channel=3, widen_factor=1, drop_rate=0.0, record_embedding=False,
                 no_grad=False):
        super(WideResNet_32x32, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(channel, nChannels[0], kernel_size=3, stride=1,
                               padding=3 if channel == 1 else 1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, drop_rate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad

    def get_last_layer(self):
        return self.fc

    def forward(self, x):
        with torch.set_grad_enabled(not self.no_grad):
            out = self.conv1(x)
            out = self.block1(out)
            out = self.block2(out)
            out = self.block3(out)
            out = self.relu(out)
            out = F.avg_pool2d(out, 8)
            out = out.view(-1, self.nChannels)
            out = self.embedding_recorder(out)
        return self.fc(out)


def WideResNet(arch: str, channel: int, num_classes: int, im_size, record_embedding: bool = False,
               no_grad: bool = False, pretrained: bool = False):
    arch = arch.lower()
    if (channel == 1 and im_size[0] == 28 and im_size[1] == 28) or (
            channel == 3 and im_size[0] == 32 and im_size[1] == 32):
        if arch == "wrn168":
            net = WideResNet_32x32(16, num_classes, channel, 8)
        elif arch == "wrn2810":
            net = WideResNet_32x32(28, num_classes, channel, 10)
        elif arch == "wrn282":
            net = WideResNet_32x32(28, num_classes, channel, 2)
        else:
            raise ValueError("Model architecture not found.")
    else:
        raise NotImplementedError("Network Architecture for current dataset has not been implemented.")
    return net


def WRN168NoBn(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
           pretrained: bool = False):
    return WideResNet("wrn168", channel, num_classes, im_size, record_embedding, no_grad, pretrained)


def WRN2810NoBn(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
            pretrained: bool = False):
    return WideResNet("wrn2810", channel, num_classes, im_size, record_embedding, no_grad, pretrained)


def WRN282NoBn(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
           pretrained: bool = False):
    return WideResNet('wrn282', channel, num_classes, im_size, record_embedding, no_grad, pretrained)


def WRN502NoBn(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
           pretrained: bool = False):
    return WideResNet("wrn502", channel, num_classes, im_size, record_embedding, no_grad, pretrained)


def WRN1012NoBn(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
            pretrained: bool = False):
    return WideResNet("wrn1012", channel, num_classes, im_size, record_embedding, no_grad, pretrained)
