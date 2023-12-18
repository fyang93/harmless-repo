"""
ResNet for CIFAR10

Change conv1 kernel size from 7 to 3
"""

from collections import OrderedDict

import torch
import torch.nn as nn

class Residual(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, WithoutShortCut=False):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)  # maybe
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)  # not, the same
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.WithoutShortCut = WithoutShortCut

        # shortcut down sample
        if stride != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(in_channels, self.expansion*out_channels, kernel_size=1, stride=stride, padding=0, bias=False)), # maybe??
                ("bn", nn.BatchNorm2d(self.expansion*out_channels))
            ]))
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if not self.WithoutShortCut:
            #print("With Short Cut")
            out = out + identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)      # not, only change channels
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)  # maybe
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, self.expansion*out_channels, kernel_size=1, stride=1, padding=0, bias=False)  # not, only change channels
        self.bn3 = nn.BatchNorm2d(self.expansion*out_channels)
        self.relu = nn.ReLU(inplace=True)

        
        if stride != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(in_channels, self.expansion*out_channels, kernel_size=1, stride=stride, padding=0, bias=False)),
                ("bn", nn.BatchNorm2d(self.expansion*out_channels))
            ]))
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        out += identity
        out = self.relu(out)
        
        return out

class ResNet(nn.Module):
    _C = [64, 128, 256, 512]
    def __init__(self, block, num_blocks, num_classes, normalize):
        super(ResNet, self).__init__()
        self.hooks = {}
        self.handles = {}
        self.channels = 64
        self.normalize = normalize    # nn.BatchNorm2d(num_features=3)
        self.layer0 = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)), # not
            ("bn", nn.BatchNorm2d(num_features=64)),
            ("relu", nn.ReLU(inplace=True)),
            # ("maxpool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))
        
        self.layer1 = self.make_layer(block, self._C[0], num_blocks[0], 1)
        self.layer2 = self.make_layer(block, self._C[1], num_blocks[1], 2, WithoutShortCut=True)
        self.layer3 = self.make_layer(block, self._C[2], num_blocks[2], 2)
        self.layer4 = self.make_layer(block, self._C[3], num_blocks[3], 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(self._C[3]*block.expansion, num_classes, bias=True)

    def make_layer(self, block, out_channels, block_num, stride, WithoutShortCut=False):  # add WithoutShortCut for layer 2 block 0 conv 1
        layers = OrderedDict([("block0", block(self.channels, out_channels, stride, WithoutShortCut=WithoutShortCut))])
        self.channels = out_channels * block.expansion
        for i in range(block_num - 1):
            layers[f"block{i+1}"] = block(self.channels, out_channels, 1)
            self.channels = out_channels * block.expansion

        return nn.Sequential(layers)


    def forward(self, x):
        if self.normalize:
            x = self.normalize(x)

        # ori image x  [N, 3, 32, 32]
        if x.shape[1] != self._C[0]:
            #print("layer 0")
            x = self.layer0(x)
            #print("layer 1")
            x = self.layer1(x)

        # x0  [N, 64, 32, 32]
        #print("layer 2")
        x = self.layer2(x)
        #print("layer 3")
        x = self.layer3(x)
        #print("layer 4")
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

    def _get_x0(self, x):
        if self.normalize:
            x = self.normalize(x)
        x = self.layer0(x)
        x = self.layer1(x)     # (N,  64, 16, 16)

        return x

    def _get_fc(self, x):
        if self.normalize:
            x = self.normalize(x)
        x = self.layer0(x)
        x = self.layer1(x)     # (N,  64, 16, 16)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.flatten(x)

        return x


    def _get_value(self, x):
        if self.normalize:
            x = self.normalize(x)
        print("before conv", x.shape)
        x = self.layer0.conv(x)
        print("after conv", x.shape)

        conv = x
        x = self.layer0.bn(x)
        x = self.layer0.relu(x)
        layer0 = x

        print("after relu", x.shape)
        print("after layer1.conv", self.layer1.block0.conv1(x).shape)

        #x = self.layer0(x)     # (N,  64, 16, 16)
        x = self.layer1(x)     # (N,  64, 16, 16)
        print("after layer1", x.shape)
        print("after layer2.conv", self.layer2.block0.conv1(x).shape)

        x = self.layer2(x)     # (N, 128,  8,  8)
        print("after layer2", x.shape)

        x = self.layer3(x)     # (N, 256,  4,  4)
        x = self.layer4(x)     # (N, 512,  2,  2)
        x = self.avgpool(x)    # (N, 512,  1,  1)
        x = self.flatten(x)    # (N, 512)
        x = self.fc(x)         # (N, 10)

        return x, conv, layer0

    def add_hook(self, names):
        for name in names:
            self.handles[name] = getattr(self, name).register_forward_hook(self.forward_hook(name))
    
    def del_hook(self, names):
        for name in names:
            if name in self.handles:
                self.handles[name].remove()
                del self.handles[name]
            if name in self.hooks:
                del self.hooks[name]

    def forward_hook(self, name):
        def hook(module, input, output):
            self.hooks[name] = output
        return hook


def resnet18(num_classes, normalize=None):
    return ResNet(Residual, [2, 2, 2, 2], num_classes, normalize)

def resnet34(num_classes, normalize=None):
    return ResNet(Residual, [3, 4, 6, 3], num_classes, normalize)

def resnet50(num_classes, normalize=None):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, normalize)

def resnet101(num_classes, normalize=None):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, normalize)

def resnet152(num_classes, normalize=None):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, normalize)


# unit test
if __name__ == "__main__":

    net = resnet18(10)
    layers = []
    for name, module in net.named_children():
        layers.append(name)
    del layers[-2]
    del layers[-1]
    
    net.add_hook(layers)
    net.eval()
    images = torch.rand(4, 3, 32, 32)
    net(images)
    for layer in layers:
        B, C, H, W = net.hooks[layer].shape
        print(f"{layer:8s}: {B:2d}, {C:3d}, {H:2d}, {W:2d}")
