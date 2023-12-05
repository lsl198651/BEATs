import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchaudio.compliance.kaldi as ta_kaldi
import torchaudio.transforms as TT


# ----------------------------
# Audio Classification Model
# ----------------------------


class AudioClassifier_CNNGRU(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        conv_layers = []
        self.bn0 = nn.BatchNorm2d(1)
        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(
            3, 3), stride=(1, 1), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)
        self.mp1 = nn.MaxPool2d(2)
        self.dp1 = nn.Dropout(p=0.15)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.bn1, self.relu1,  self.mp1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(32)
        self.mp2 = nn.MaxPool2d(2)
        self.dp2 = nn.Dropout(p=0.1)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.bn2, self.relu2, self.mp2, self.dp2]

        # Third Convolution Block
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        self.dp3 = nn.Dropout(p=0.1)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.bn3, self.relu3, self.dp3]

        # Fourth Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.bn4, self.relu4]
        self.conv = nn.Sequential(*conv_layers)
        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        # wide features
        # self.wide = nn.Linear(in_features=15, out_features=20)
        self.lin = nn.Linear(in_features=64, out_features=2)
        # Wrap the Convolutional Blocks
        self.gru=nn.GRU(input_size=64,hidden_size=64,num_layers=1,batch_first=True)


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

    def forward(self, x):
        # Run the convolutional blocks
        fbank = self.preprocess(x, args=None)
        fbank = fbank.unsqueeze(1)
        x = self.conv(fbank)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x_all = x.view(x.shape[0], -1)
        x=self.gru(x_all)
        # add wide features and concat two layers
        # print(x1.size())
        # x1 = self.wide(wavfeat)
        # x_all = torch.cat((x_all, x1), dim=1)
        # x = self.dp(x)
        # Linear layer
        x_all = self.lin(x_all)
        # x_all = torch.softmax(x_all, dim=1)
        # Final output
        return x_all

