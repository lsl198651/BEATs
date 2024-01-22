import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchaudio.compliance.kaldi as ta_kaldi
import torchaudio.transforms as TT


# ----------------------------
# Audio Classification Model
# ----------------------------


class CNN1d(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv1d(1, 8, kernel_size=7, stride=1, padding=3)
        self.ap = nn.AvgPool1d(kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        # conv_layers += [self.conv1,  self.ap]  # self.bn1, self.relu1,

        # Second Convolution Block
        self.conv2 = nn.Conv1d(8, 512, kernel_size=5, stride=1, padding=2)
        self.mp2 = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        # conv_layers += [self.conv2, self.mp]

        # Third Convolution Block
        self.conv3 = nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2)
        self.mp3 = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(32)
        # conv_layers += [self.conv3, self.mp3]

        # Fourth Convolution Block
        self.conv4 = nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2)
        self.mp4 = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(32)
        # conv_layers += [self.conv4, self.mp4]
        self.relu = nn.ReLU()
        # wide features
        self.lin1 = nn.Linear(in_features=1024, out_features=128)
        self.lin2 = nn.Linear(in_features=128, out_features=64)
        self.lin3 = nn.Linear(in_features=64, out_features=32)
        self.lin4 = nn.Linear(in_features=32, out_features=2)
    # calculate fbank value

    def preprocess(
            self,
            source: torch.Tensor,
            args=None,
    ) -> torch.Tensor:
        fbanks = []
        for waveform in source:
            # waveform = waveform.unsqueeze(0) * 2 ** 15  # wavform × 2^15
            waveform = waveform.unsqueeze(0)
            fbank = ta_kaldi.fbank(
                waveform, num_mel_bins=128, sample_frequency=4000, frame_length=25, frame_shift=10)
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
        x = self.ap(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.mp2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.mp3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.mp4(x)
        x = torch.flatten(x, 1)

        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        x = self.lin4(x)
        x = F.softmax(x)

        return x

# ----------------------------
# Audio Classification Model
# ----------------------------
