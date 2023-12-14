import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchaudio.compliance.kaldi as ta_kaldi
import torchaudio.transforms as TT
from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor

# 暂时没有什么用

# ----------------------------
# Audio Classification Model
# ----------------------------


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class AudioClassifier_SENet(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self,
                 #  block: Type[Union[BasicBlock, Bottleneck]],

                 num_classes: int = 2,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,):
        super().__init__()
        # conv_layers = []
        # self.bn0 = nn.BatchNorm2d(1)
        # # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=(
        #     3, 3), stride=(1, 1), padding=(2, 2))
        # self.relu1 = nn.ReLU()
        # self.bn1 = nn.BatchNorm2d(32)
        # self.mp1 = nn.MaxPool2d(2)
        # self.dp1 = nn.Dropout(p=0.15)
        # init.kaiming_normal_(self.conv1.weight, a=0.1)
        # self.conv1.bias.data.zero_()
        # conv_layers += [self.conv1, self.bn1, self.relu1,  self.mp1]

        # # Second Convolution Block
        # self.conv2 = nn.Conv2d(32, 32, kernel_size=(
        #     3, 3), stride=(1, 1), padding=(1, 1))
        # self.relu2 = nn.ReLU()
        # self.bn2 = nn.BatchNorm2d(32)
        # self.mp2 = nn.MaxPool2d(2)
        # self.dp2 = nn.Dropout(p=0.1)
        # init.kaiming_normal_(self.conv2.weight, a=0.1)
        # self.conv2.bias.data.zero_()
        # conv_layers += [self.conv2, self.bn2, self.relu2, self.mp2, self.dp2]

        # # Third Convolution Block
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=(
        #     3, 3), stride=(1, 1), padding=(1, 1))
        # self.relu3 = nn.ReLU()
        # self.bn3 = nn.BatchNorm2d(32)
        # self.dp3 = nn.Dropout(p=0.1)
        # init.kaiming_normal_(self.conv3.weight, a=0.1)
        # self.conv3.bias.data.zero_()
        # conv_layers += [self.conv3, self.bn3, self.relu3, self.dp3]

        # Fourth Convolution Block
        # self.conv4 = SEBasicBlock(
        #     inplanes=32, planes=64, stride=1, downsample=None, groups=1,)
        # self.relu4 = nn.ReLU()
        # self.bn4 = nn.BatchNorm2d(64)
        # init.kaiming_normal_(self.conv4.weight, a=0.1)
        # self.conv4.bias.data.zero_()
        # conv_layers += [self.conv4]  # , self.bn4, self.relu4
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 32
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(
            1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(SEBasicBlock, 32)
        self.layer2 = self._make_layer(
            SEBasicBlock, 32,  stride=1, dilate=False)
        self.layer3 = self._make_layer(
            SEBasicBlock, 32,  stride=1, dilate=False)
        self.layer4 = self._make_layer(
            SEBasicBlock, 64, stride=1, dilate=False)

        # self.sk_conv = nn.Sequential(*layers)
        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)

        # wide features
        self.wide = nn.Linear(in_features=15, out_features=20)
        self.lin = nn.Linear(in_features=64, out_features=2)
        self.lin1 = nn.Linear(in_features=80, out_features=128)
        # Wrap the Convolutional Blocks
        # self.conv = nn.Sequential(*conv_layers)
        self.dp = nn.Dropout(p=0.3)

    def _make_layer(
        self,
        block,
        planes: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )

    # calculate fbank value

    def preprocess(
            self,
            source: torch.Tensor,
            args=None,
    ) -> torch.Tensor:
        fbanks = []
        # waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        for waveform in source:
            # waveform = waveform.unsqueeze(0) * 2 ** 15  # wavform × 2^15
            waveform = waveform.unsqueeze(0)
            fbank = ta_kaldi.fbank(
                waveform, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10)
            fbank_mean = fbank.mean()
            fbank_std = fbank.std()
            fbank = (fbank - fbank_mean) / fbank_std
            fbanks.append(fbank)
        fbank = torch.stack(fbanks, dim=0)
        return fbank
    # ----------------------------
    # Forward pass computations
    # ----------------------------

    def forward(self, x, x1):
        # Run the convolutional blocks
        fbank = self.preprocess(x, args=None)
        fbank = fbank.unsqueeze(1)
        x = self.conv1(fbank)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.sk_conv(fbank)
        # self.se = SELayer(planes, reduction)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x_all = x.view(x.shape[0], -1)
        # add wide features and concat two layers
        # print(x1.size())
        x1 = self.wide(x1)
        x_all = torch.cat((x_all, x1), dim=1)
        # x = self.dp(x)
        # Linear layer
        x_all = self.lin(x_all)
        # x_all = torch.softmax(x_all, dim=1)
        # Final output
        return x_all
