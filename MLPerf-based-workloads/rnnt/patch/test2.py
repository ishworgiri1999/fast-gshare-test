# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2019, Myrtle Software Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import torch
import torch.nn as nn
import math
import librosa
from .segment import AudioSegment


class WaveformFeaturizer(object):
    def __init__(self, input_cfg):
        self.cfg = input_cfg

    def process(self, file_path, offset=0, duration=0, trim=False):
        audio = AudioSegment.from_file(file_path,
                                       target_sr=self.cfg['sample_rate'],
                                       int_values=self.cfg.get(
                                           'int_values', False),
                                       offset=offset, duration=duration, trim=trim)
        return self.process_segment(audio)

    def process_segment(self, audio_segment):
        return torch.tensor(audio_segment.samples, dtype=torch.float)

    @classmethod
    def from_config(cls, input_config, perturbation_configs=None):
        return cls(input_config)


constant = 1e-5


def normalize_batch(x, seq_len, normalize_type):
    if normalize_type == "per_feature":
        x_mean = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype,
                             device=x.device)
        x_std = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype,
                            device=x.device)
        for i in range(x.shape[0]):
            x_mean[i, :] = x[i, :, :seq_len[i]].mean(dim=1)
            x_std[i, :] = x[i, :, :seq_len[i]].std(dim=1)
        # make sure x_std is not zero
        x_std += constant
        return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)
    elif normalize_type == "all_features":
        x_mean = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        x_std = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        for i in range(x.shape[0]):
            x_mean[i] = x[i, :, :seq_len[i].item()].mean()
            x_std[i] = x[i, :, :seq_len[i].item()].std()
        # make sure x_std is not zero
        x_std += constant
        return (x - x_mean.view(-1, 1, 1)) / x_std.view(-1, 1, 1)
    else:
        return x


def splice_frames(x, frame_splicing):
    """ Stacks frames together across feature dim

    input is batch_size, feature_dim, num_frames
    output is batch_size, feature_dim*frame_splicing, num_frames

    """
    seq = [x]
    for n in range(1, frame_splicing):
        tmp = torch.zeros_like(x)
        tmp[:, :, :-n] = x[:, :, n:]
        seq.append(tmp)
    return torch.cat(seq, dim=1)[:, :, ::frame_splicing]


class FilterbankFeatures(nn.Module):
    def __init__(self, sample_rate=8000, window_size=0.02, window_stride=0.01,
                 window="hamming", normalize="per_feature", n_fft=None,
                 preemph=0.97,
                 nfilt=64, lowfreq=0, highfreq=None, log=True, dither=constant,
                 pad_to=8,
                 max_duration=16.7,
                 frame_splicing=1):
        super(FilterbankFeatures, self).__init__()
#        print("PADDING: {}".format(pad_to))

        torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'none': None,
        }

        self.win_length = int(sample_rate * window_size)  # frame size
        self.hop_length = int(sample_rate * window_stride)
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))

        self.normalize = normalize
        self.log = log
        self.dither = dither
        self.frame_splicing = frame_splicing
        self.nfilt = nfilt
        self.preemph = preemph
        self.pad_to = pad_to
        # For now, always enable this.
        # See https://docs.google.com/presentation/d/1IVC3J-pHB-ipJpKsJox_SqmDHYdkIaoCXTbKmJmV2-I/edit?usp=sharing for elaboration
        self.use_deterministic_dithering = True
        highfreq = highfreq or sample_rate / 2
        window_fn = torch_windows.get(window, None)
        window_tensor = window_fn(self.win_length,
                                  periodic=False) if window_fn else None
        filterbanks = torch.tensor(
            librosa.filters.mel(sr=sample_rate, n_fft=self.n_fft, n_mels=nfilt, fmin=lowfreq, fmax=highfreq), dtype=torch.float).unsqueeze(0)
        # self.fb = filterbanks
        # self.window = window_tensor
        self.register_buffer("fb", filterbanks)
        self.register_buffer("window", window_tensor)
        # Calculate maximum sequence length (# frames)
        max_length = 1 + math.ceil(
            (max_duration * sample_rate - self.win_length) / self.hop_length
        )
        max_pad = 16 - (max_length % 16)
        self.max_length = max_length + max_pad

    def get_seq_len(self, seq_len):
        seq_len = (seq_len + self.hop_length - 1) // self.hop_length
        seq_len = (seq_len + self.frame_splicing - 1) // self.frame_splicing
        return seq_len

    @torch.no_grad()
    def forward(self, inp: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, seq_len = inp

        dtype = x.dtype

        seq_len = self.get_seq_len(seq_len)

        # dither
        if self.dither > 0 and not self.use_deterministic_dithering:
            x += self.dither * torch.randn_like(x)

        # do preemphasis
        # Ideally, we would mask immediately after this... Ugh :(
        if self.preemph is not None:
            x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]),
                          dim=1)

        # do stft
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                       win_length=self.win_length,
                       center=True, window=self.window.to(dtype=torch.float),return_complex=False)

        # get power spectrum
        x = x.pow(2).sum(-1)

        if self.dither > 0 and self.use_deterministic_dithering:
            x = x + self.dither ** 2
        # dot with filterbank energies
        x = torch.matmul(self.fb.to(x.dtype), x)

        # log features if required
        if self.log:
            x = torch.log(x + 1e-20)

        # frame splicing if required
        if self.frame_splicing > 1:
            seq = [x]
            for n in range(1, self.frame_splicing):
                tmp = torch.zeros_like(x)
                tmp[:, :, :-n] = x[:, :, n:]
                seq.append(tmp)
            x = torch.cat(seq, dim=1)[:, :, ::self.frame_splicing]

        # normalize if required
        constant = 1e-5
        if self.normalize == "per_feature":
            x_mean = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype,
                                 device=x.device)
            x_std = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype,
                                device=x.device)
            for i in range(x.shape[0]):
                x_mean[i, :] = x[i, :, :seq_len[i]].mean(dim=1)
                x_std[i, :] = x[i, :, :seq_len[i]].std(dim=1)
                # make sure x_std is not zero
                x_std += constant
            x = (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)
        elif self.normalize == "all_features":
            x_mean = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
            x_std = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
            for i in range(x.shape[0]):
                x_mean[i] = x[i, :, :seq_len[i].item()].mean()
                x_std[i] = x[i, :, :seq_len[i].item()].std()
                # make sure x_std is not zero
                x_std += constant
            x = (x - x_mean.view(-1, 1, 1)) / x_std.view(-1, 1, 1)
        else:
            x = x

        # Hmmm... They don't do any masking anymore. Seems concerning!

        # mask to zero any values beyond seq_len in batch, pad to multiple of `pad_to` (for efficiency)
        # max_len = x.size(-1)
        x = x[:, :, :seq_len.max()]   # rnnt loss requires lengths to match
        # mask = torch.arange(max_len).to(seq_len.dtype).to(x.device).expand(x.size(0),
        #                                                                   max_len) >= seq_len.unsqueeze(1)

        # x = x.masked_fill(mask.unsqueeze(1).to(device=x.device), 0)
        pad_to = self.pad_to
        if pad_to != 0:
            raise NotImplementedError()
        # if pad_to == "max":
        #    x = nn.functional.pad(x, (0, self.max_length - x.size(-1)))
        # elif pad_to > 0:
        #    pad_amt = x.size(-1) % pad_to
        #    if pad_amt != 0:
        #        x = nn.functional.pad(x, (0, pad_to - pad_amt))

        return x.to(dtype)

    @classmethod
    def from_config(cls, cfg, log=False):
        return cls(sample_rate=cfg['sample_rate'], window_size=cfg['window_size'],
                   window_stride=cfg['window_stride'], n_fft=cfg['n_fft'],
                   nfilt=cfg['features'], window=cfg['window'],
                   normalize=cfg['normalize'],
                   max_duration=cfg.get('max_duration', 16.7),
                   dither=cfg['dither'], pad_to=cfg.get("pad_to", 0),
                   frame_splicing=cfg.get("frame_splicing", 1), log=log)


class FeatureFactory(object):
    featurizers = {
        "logfbank": FilterbankFeatures,
        "fbank": FilterbankFeatures,
    }

    def __init__(self):
        pass

    @classmethod
    def from_config(cls, cfg):
        feat_type = cfg.get('feat_type', "logspect")
        featurizer = cls.featurizers[feat_type]
        # return featurizer.from_config(cfg, log="log" in cfg['feat_type'])
        return featurizer.from_config(cfg, log="log" in feat_type)