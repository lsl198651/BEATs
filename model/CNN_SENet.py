import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchaudio.compliance.kaldi as ta_kaldi
import torchaudio.transforms as TT
# 暂时没有什么用

# ----------------------------
# Audio Classification Model
# ----------------------------


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, 32, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


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
    def __init__(self):
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
        layers = []
        layers.append(SEBasicBlock(
            1, 32, 1, None, 1, 64, 1))
        layers.append(SEBasicBlock(
            32, 32, 1, None, 1, 64, False))
        layers.append(SEBasicBlock(
            32, 32, 1, None, 1, 64, False))
        layers.append(SEBasicBlock(
            32, 64, 1, None, 1, 64, False))

        self.sk_conv = nn.Sequential(*layers)
        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)

        # wide features
        self.wide = nn.Linear(in_features=15, out_features=20)
        self.lin = nn.Linear(in_features=64, out_features=2)
        self.lin1 = nn.Linear(in_features=80, out_features=128)
        # Wrap the Convolutional Blocks
        # self.conv = nn.Sequential(*conv_layers)
        self.dp = nn.Dropout(p=0.3)

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
        x = self.sk_conv(fbank)
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