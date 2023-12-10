# --------------------------------------------------------
# BEATs: Audio Pre-Training with Acoustic Tokenizers (https://arxiv.org/abs/2212.09058)
# Github source: https://github.com/microsoft/unilm/tree/master/beats
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# --------------------------------------------------------


from typing import Optional
import logging
import torch
import torch.nn as nn
from torch.nn import LayerNorm, BatchNorm2d
import torchaudio.compliance.kaldi as ta_kaldi
import torchaudio.transforms as TT
from backbone import TransformerEncoder
from util.BEATs_def import Log_GF, Mel_Time_Frequency_Spectrum_2
from torch.nn import init

logger = logging.getLogger(__name__)


class BEATsConfig:
    def __init__(self, cfg=None):
        self.input_patch_size: int = -1  # path size of patch embedding
        self.embed_dim: int = 512  # patch embedding dimension
        self.conv_bias: bool = False  # include bias in conv encoder

        self.encoder_layers: int = 2  # num encoder layers in the transformer
        self.encoder_embed_dim: int = 768  # encoder embedding dimension
        self.encoder_ffn_embed_dim: int = 1536  # encoder embedding dimension for FFN
        self.encoder_attention_heads: int = 4  # num encoder attention heads
        self.activation_fn: str = "gelu"  # activation function to use

        self.layer_wise_gradient_decay_ratio: float = (
            1  # ratio for layer-wise gradient decay
        )
        self.layer_norm_first: bool = False  # apply layernorm first in the transformer
        self.deep_norm: bool = False  # apply deep_norm first in the transformer

        # dropouts
        self.dropout: float = 0.1  # dropout probability for the transformer
        self.attention_dropout: float = 0.1  # dropout probability for attention weights
        self.activation_dropout: float = (
            0.0  # dropout probability after activation in FFN
        )
        self.encoder_layerdrop: float = (
            0.0  # probability of dropping a tarnsformer layer
        )
        self.dropout_input: float = (
            0.0  # dropout to apply to the input (after feat extr)
        )

        # positional embeddings
        self.conv_pos: int = (
            128  # number of filters for convolutional positional embeddings
        )
        self.conv_pos_groups: int = (
            16  # number of groups for convolutional positional embedding
        )

        # relative position embedding
        self.relative_position_embedding: bool = (
            False  # apply relative position embedding
        )
        self.num_buckets: int = 320  # number of buckets for relative position embedding
        self.max_distance: int = (
            1280  # maximum distance for relative position embedding
        )
        self.gru_rel_pos: bool = False  # apply gated relative position embedding

        # label predictor
        # whether the model is a fine-tuned model.
        self.finetuned_model: bool = False
        self.predictor_dropout: float = 0.1  # dropout probability for the predictor
        self.predictor_class: int = 2  # target class number for the predictor

        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        self.__dict__.update(cfg)


class BEATs(nn.Module):
    def __init__(
        self,
        cfg: BEATsConfig,
    ) -> None:
        super().__init__()
        logger.info(f"BEATs Config: {cfg.__dict__}")

        self.cfg = cfg

        self.embed = cfg.embed_dim

        # layer 1
        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.input_patch_size = cfg.input_patch_size
        # layer 2
        self.patch_embedding = nn.Conv2d(
            1,
            self.embed,
            kernel_size=self.input_patch_size,
            stride=self.input_patch_size,
            bias=cfg.conv_bias,
        )
        # layer 3
        self.dropout_input = nn.Dropout(cfg.dropout_input)

        assert not cfg.deep_norm or not cfg.layer_norm_first
        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)
        # self.batch_norm = BatchNorm2d(self.embed)

        if cfg.finetuned_model:
            self.predictor_dropout = nn.Dropout(cfg.predictor_dropout)
            self.predictor = nn.Linear(
                cfg.encoder_embed_dim, cfg.predictor_class)
        else:
            self.predictor = None

    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        # 计算padding-mask和feature-mask之间的列长度的余数extra
        # 余数extra值大于0则将前extra列作为padding-mask
        # 最后将padding-mask reshape到行数不变，列数与features相同
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(
            padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    # calculate fbank value
    def preprocess(
            self,
            source: torch.Tensor,
            # fbank_mean: float = 15.41663,
            # fbank_std: float = 6.55582,
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
            # S1 = librosa.feature.delta(s)

            if args.mask is True:
                # freqm_value = 30  # 横向
                # timem_value = 1  # 纵向
                # SpecAug, not do for eval set
                freqm = TT.FrequencyMasking(freq_mask_param=args.freqm_value)
                timem = TT.TimeMasking(time_mask_param=args.timem_value)
                fbank = torch.transpose(fbank, 0, 1)
                # this is just to satisfy new torchaudio version, which only accept [1, freq, time]
                fbank = fbank.unsqueeze(0)
                fbank = freqm(fbank)
                fbank = timem(fbank)
                # squeeze it back, it is just a trick to satisfy new torchaudio version
                fbank = fbank.squeeze(0)
                fbank = torch.transpose(fbank, 0, 1)
            fbank_mean = fbank.mean()
            fbank_std = fbank.std()
            fbank = (fbank - fbank_mean) / fbank_std
            fbanks.append(fbank)
        fbank = torch.stack(fbanks, dim=0)
        return fbank

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        args=None,
    ):
        # wav提取fbank系数
        fbank = self.preprocess(
            source,
            args=args,
        )
        # 如果有padding-mask的话进行forward-padding-mask
        # 返回一个值？？？
        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(fbank, padding_mask)

        fbank = fbank.unsqueeze(1)
        # fbank = torch.FloatTensor(fbank)
        # fbank送入卷积层patch_embedding
        features = self.patch_embedding(fbank)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        # 求转置
        features = features.transpose(1, 2)
        # 正则化层
        features = self.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)
        # 若 linear projection network存在
        if self.post_extract_proj is not None:
            # 将特征送入post_extract_proj线性网络转化为patch embeddings E
            features = self.post_extract_proj(features)
        # 送入dropout
        x = self.dropout_input(features)

        # 送入bcakbone的TransformerEncoder to obtain the encoded patch representations R
        x, layer_results = self.encoder(
            x,
            # 若padding_mask非空，将x中与padding_mask对应位置置零
            padding_mask=padding_mask,
        )

        if self.predictor is not None:
            x = self.predictor_dropout(x)
            logits = self.predictor(x)

            if padding_mask is not None and padding_mask.any():
                logits[padding_mask] = 0
                logits = logits.sum(dim=1)
                logits = logits / (~padding_mask).sum(dim=1).unsqueeze(-1).expand_as(
                    logits
                )
            else:
                logits = logits.mean(dim=1)

            lprobs = torch.sigmoid(logits)

            return lprobs, padding_mask
        else:
            return x, padding_mask


class BEATs_Pre_Train_itere3(nn.Module):
    def __init__(self, args):
        self.model_name = args.model
        self.layers = args.layers
        self.args = args
        super(BEATs_Pre_Train_itere3, self).__init__()

        checkpoint = torch.load(
            r"D:\Shilong\murmur\00_Code\LM\LM_Model\BEATs"
            + "\\"
            + self.model_name
            + ".pt"
        )
        cfg = BEATsConfig(checkpoint["cfg"])
        BEATs_model = BEATs(cfg)
        BEATs_model.load_state_dict(checkpoint["model"])
        # BEATs
        self.BEATs = BEATs_model
        conv_layers = []
        conv_layers2 = []
        # Dropout
        self.last_Dropout = nn.Dropout(0.1)
        # conv block
        # ---------------------------------
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

        self.conv3 = nn.Conv2d(32, 32, kernel_size=(
            3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        self.dp3 = nn.Dropout(p=0.1)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.bn3, self.relu3, self.dp3]

        self.conv4 = nn.Conv2d(32, 64, kernel_size=(
            3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.bn4, self.relu4]
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(*conv_layers)
        self.conv2 = nn.Sequential(*conv_layers2)
        # -------------------------------------------------------
        self.fc_layer = nn.Sequential(
            # nn.Linear(32*10, 16*38),
            # nn.ReLU(),
            # nn.Tanh(),
            # nn.Linear(768, 768),
            # nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x,  padding_mask: torch.Tensor = None):  # , gfcc=None
        # with torch.no_grad():
        x, _ = self.BEATs.extract_features(x, padding_mask, args=self.args)
        # dropout
        # power = ta_kaldi.spectrogram(x, frame_length=25, frame_shift=10)
        # with torch.enable_grad():
        # x = x.mean(dim=1)
        # x = x.squeeze(1)
        x = x.unsqueeze(1)
        x = self.conv(x)
        # x = x.mean(dim=3)  # 64*9 cat 64*29
        # x = torch.cat((x, gfcc), dim=2)
        # x = x.transpose(1, 2)
        # x = self.conv2(x)
        x = self.ap(x)
        # x = x.mean(dim=2)
        x = x.reshape(x.shape[0], -1)
        output = self.fc_layer(x)
        # output = torch.softmax(output, dim=1)
        # output = self.last_layer(y)
        # mean
        # output = output.mean(dim=1)
        # sigmoid
        # output = torch.softmax(output, dim=1)
        return output
