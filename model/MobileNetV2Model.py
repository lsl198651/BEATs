from torch import nn
import torch
import torchaudio.compliance.kaldi as ta_kaldi
import torchaudio.compliance.kaldi as ta_kaldi
from torch.nn import init

def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size,
                      stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(
                in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel,
                       stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=2, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        # ---------------------------------------------------
        conv_layers = []
        self.bn0 = nn.BatchNorm2d(1)
        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(1280, 640, kernel_size=(
            3, 3), stride=(1, 1), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(640)
        self.mp1 = nn.MaxPool2d(2)
        self.dp1 = nn.Dropout(p=0.15)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.bn1, self.relu1,  self.mp1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(640, 160, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(160)
        self.mp2 = nn.MaxPool2d(2)
        self.dp2 = nn.Dropout(p=0.1)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.bn2, self.relu2, self.mp2, self.dp2]

        # Third Convolution Block
        self.conv3 = nn.Conv2d(160, 80, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(80)
        self.dp3 = nn.Dropout(p=0.1)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.bn3, self.relu3, self.dp3]

        # Fourth Convolution Block
        self.conv4 = nn.Conv2d(80, 64, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.bn4, self.relu4]
        self.conv = nn.Sequential(*conv_layers)
        # --------------------------------------------------
        block = InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        inverted_residual_setting = [
            # t, c, n, s
            [1, 8, 1, 1],
            [6, 12, 2, 2],
        ]

        features = []
        # conv1 layer
        features.append(ConvBNReLU(1, input_channel, stride=2))
        # building inverted residual residual blockes
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, last_channel, 1))
        # combine feature layers
        # ------------------------------
        self.features = nn.Sequential(*features)
        # ==================================
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)

        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

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
            # spec = TT.MelSpectrogram(sr=16000, n_fft=512, win_length=25,
            #                                  hop_length=25, n_mels=128, f_min=25, f_max=2000)(waveform)
            # spec = TT.AmplitudeToDB(top_db=20)(spec)
            fbank = ta_kaldi.fbank(
                waveform, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10)
            fbank_mean = fbank.mean()
            fbank_std = fbank.std()
            fbank = (fbank - fbank_mean) / fbank_std
            fbanks.append(fbank)
        fbank = torch.stack(fbanks, dim=0)
        return fbank

    def forward(self, x):
        fbank = self.preprocess(x, args=None)
        fbank = fbank.unsqueeze(1)
        x = self.features(fbank)
        x=self.conv(x)
        x = self.ap(x)
        # x = torch.flatten(x, 1)
        x=x.squeeze()
        x = self.classifier(x)
        return x
