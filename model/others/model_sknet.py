import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchaudio.compliance.kaldi as ta_kaldi


class SKConv(nn.Module):
    def __init__(self, channels, branches=2, reduce=2, stride=1, len=32):
        super(SKConv, self).__init__()
        len = max(int(channels // reduce), len)
        self.convs = nn.ModuleList([])
        for i in range(branches):
            self.convs.append(nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1 + i, dilation=1 + i,
                          bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(channels, len, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(len),
            nn.ReLU(inplace=True)
        )
        self.fcs = nn.ModuleList([])
        for i in range(branches):
            self.fcs.append(
                nn.Conv2d(len, channels, kernel_size=1, stride=1, bias=False)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = [conv(x) for conv in self.convs]
        x = torch.stack(x, dim=1)
        attention = torch.sum(x, dim=1)
        attention = self.gap(attention)
        attention = self.fc(attention)
        attention = [fc(attention) for fc in self.fcs]
        attention = torch.stack(attention, dim=1)
        attention = self.softmax(attention)
        x = torch.sum(x * attention, dim=1)
        return x


# ----------------------------
# Audio Classification Model
# -------------------      ---------


class AudioClassifier(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        # SKConv
        # self.pre = nn.Sequential(
        #     nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(inplace=True),
        # )
        # self.SK = SKConv(32)

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((2, 2), stride=(2, 2))
        )

        # self.block2 = nn.Sequential(
        #     nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1)),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(32),
        #     nn.MaxPool2d((2, 2), stride=(2, 2))
        # )

        # self.block3 = nn.Sequential(
        #     nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1)),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(32),
        #     nn.MaxPool2d((2, 2), stride=(2, 2))
        # )

        # self.block4 = nn.Sequential(
        #     nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1)),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(32),
        #     nn.MaxPool2d((2, 1), stride=(2, 1))
        # )

        self.block2 = nn.Sequential(
            SKConv(32),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((2, 2), stride=(2, 2))
        )

        self.block3 = nn.Sequential(
            SKConv(32),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((2, 2), stride=(2, 2))
        )

        self.block4 = nn.Sequential(
            SKConv(32),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((2, 1), stride=(2, 1))
        )

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.wide = nn.Linear(in_features=6, out_features=20)
        self.dense1 = nn.Linear(in_features=52, out_features=128)
        self.dense2 = nn.Linear(in_features=128, out_features=2)
        self.dp = nn.Dropout(p=0.5)

    def preprocess(self, source: torch.Tensor, args=None,) -> torch.Tensor:
        fbanks = []
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
        fbank = self.preprocess(x)
        # x = self.pre(x)
        # x = self.SK(x)
        fbank = fbank.unsqueeze(1)
        x = self.block1(fbank)
        # x = self.SK(x)
        x = self.block2(x)

        x = self.block3(x)

        # x = self.block4(x)
        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x_all = x.view(x.shape[0], -1)
        x1 = self.wide(x1)
        # x2 = x2.flatten(1)
        x_all = torch.cat((x_all, x1), dim=1)
        x_all = self.dp(x_all)
        x_all = self.dense1(x_all)
        x_all = self.dp(x_all)
        x_all = self.dense2(x_all)

        # Final output
        return x_all
