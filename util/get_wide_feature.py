import numpy as np
import random
import torchaudio
import librosa
import torch
from torchaudio import transforms


def get_sex(data):
    sex = None
    for l in data.split('\n'):
        if l.startswith('#Sex:'):
            try:
                sex = l.split(': ')[1].strip()
            except:
                pass
    return sex
# Compare normalized strings.


def compare_strings(x, y):
    try:
        return str(x).strip().casefold() == str(y).strip().casefold()
    except AttributeError:  # For Python 2.x compatibility
        return str(x).strip().lower() == str(y).strip().lower()


def get_pregnancy_status_mod(data):
    is_pregnant = None
    for l in data.split('\n'):
        if l.startswith('#Pregnancy status:'):
            try:
                if compare_strings(l.split(': ')[1].strip(), 'True'):
                    is_pregnant = True
                else:
                    is_pregnant = False
                # is_pregnant = bool(l.split(': ')[1].strip())
            except:
                pass
    return is_pregnant


def get_age(data):
    age = None
    for l in data.split('\n'):
        if l.startswith('#Age:'):
            try:
                age = l.split(': ')[1].strip()
            except:
                pass
    return age


def get_features_mod(data):
   # Extract the age group, sex and the pregnancy status features
    age_group = get_age(data)
    age_list = ['Neonate', 'Infant', 'Child', 'Adolescent', 'Young Adult']
    is_pregnant = get_pregnancy_status_mod(data)
    if age_group not in ['Neonate', 'Infant', 'Child', 'Adolescent', 'Young Adult']:
        if is_pregnant:
            age = 'Young Adult'
        else:
            age = 'Child'
    else:
        age = age_group

    age_fea = np.zeros(5, dtype=int)
    age_fea[age_list.index(age)] = 1
    # Extract sex. Use one-hot encoding.
    sex = get_sex(data)
    sex_features = np.zeros(2, dtype=int)
    if compare_strings(sex, 'Female'):
        sex_features[0] = 1
    elif compare_strings(sex, 'Male'):
        sex_features[1] = 1
    preg_fea = np.zeros(2, dtype=int)
    if is_pregnant:
        preg_fea[0] = 1
    else:
        preg_fea[1] = 1

    wide_fea = np.append(age_fea, [sex_features, preg_fea])
    return wide_fea


class AudioUtil():
    # all the data augmentation
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)

    # ----------------------------
    # Since Resample applies to a single channel, we resample one channel at a time
    # ----------------------------
    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud

        if (sr == newsr):
            # Nothing to do
            return aud

        num_channels = sig.shape[0]
        # Resample first channel
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :])
        if (num_channels > 1):
            # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:, :])
            resig = torch.cat([resig, retwo])

        return ((resig, newsr))

    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr // 1000 * max_ms

        if sig_len > max_len:
            # random select the max_len
            cut_start = random.randint(0, sig_len-max_len)
            sig = sig[:, cut_start:max_len+cut_start]
        elif sig_len < max_len:
            # length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)
        return (sig, sr)

    # ---
    # raw audio augmentation
    @staticmethod
    def time_shift(aud, shift_limit):
        sig, sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)

    @staticmethod
    def pitch_shift(aud, shift_limit=[-4, 4]):
        sig, sr = aud
        pitch_shift = np.random.randint(shift_limit[0], shift_limit[1] + 1)
        sig_new = librosa.effects.pitch_shift(
            sig.numpy(), sr=sr, n_steps=pitch_shift)
        return (torch.Tensor(sig_new), sr)

    @staticmethod
    def time_stretch(aud, shift_limit=[0.9, 1.2]):
        sig, sr = aud
        stretch_time = random.uniform(shift_limit[0], shift_limit[1])
        sig_new = librosa.effects.time_stretch(sig.numpy(), rate=stretch_time)
        return (torch.Tensor(sig_new), sr)

    @staticmethod
    def add_noise(aud):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        wn = np.random.randn(sig_len)
        sig = sig + 0.005*wn
        return (sig, sr)

    @staticmethod
    def add_noise_snr(aud, snr=15, p=0.5):
        aug_apply = torch.distributions.Bernoulli(p).sample()
        if aug_apply.to(torch.bool):
            sig, sr = aud
            num_rows, sig_len = sig.shape
            p_sig = np.sum(abs(sig.numpy())**2)/sig_len
            p_noise = p_sig / 10 ** (snr/10)
            wn = np.random.randn(sig_len) * np.sqrt(p_noise)
            wn_new = np.tile(wn, (num_rows, 1))
            sig_new = sig.numpy() + wn_new
            sig_new = torch.Tensor(sig_new)
        else:
            sig_new, sr = aud
        return (sig_new, sr)

    @staticmethod
    def get_zrc(aud):
        sig, sr = aud
        sig_len = sig.shape
        # sig_len_act = np.nonzero(sig.numpy())[-1]
        zcr = librosa.zero_crossings(sig.numpy())
        # zcr_ratio = np.sum(zcr)/sig_len
        zcr = librosa.feature.zero_crossing_rate(
            sig.numpy(), frame_length=25, hop_length=10)
        zcr_mean = np.mean(np.squeeze(zcr))
        zcr_std = np.std(np.squeeze(zcr))
        return zcr_mean, zcr_std

    @staticmethod
    def get_spec(aud):
        sig, sr = aud
        spectral_centroids = librosa.feature.spectral_centroid(
            y=sig.numpy(), sr=16000, n_fft=200, hop_length=10)[0]
        spec_bw = librosa.feature.spectral_bandwidth(
            y=sig.numpy(), sr=16000, n_fft=200, hop_length=10)
        return np.mean(spectral_centroids) / 8000, np.std(spectral_centroids)/8000, np.mean(spec_bw)/8000, np.std(spec_bw)/8000

    # -------
    # generate a spectrogram
    # ----
    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=512, win_len=None, hop_len=None):
        sig, sr = aud
        top_db = 80

        win_length = int(round(win_len * sr / 1000))
        hop_length = int(round(hop_len * sr / 1000))
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, win_length=win_length,
                                         hop_length=hop_length, n_mels=n_mels, f_min=25, f_max=2000)(sig)

        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)

    # ----
    # Augment the spectrogram

    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1, p=0.7):
        aug_apply = torch.distributions.Bernoulli(p).sample()
        if aug_apply.to(torch.bool):
            _, n_mels, n_steps = spec.shape
            mask_value = spec.mean()
            aug_spec = spec

            freq_mask_param = max_mask_pct * n_mels

            for _ in range(n_freq_masks):
                aug_spec = transforms.FrequencyMasking(
                    freq_mask_param)(aug_spec, mask_value)

            time_mask_param = max_mask_pct * n_steps
            for _ in range(n_time_masks):
                aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec)
        else:
            aug_spec = spec
        return aug_spec


def hand_fea(aud):
    sig, sr = aud
    zcr_mean, zcr_std = AudioUtil.get_zrc(aud)
    spec_mean, spec_std, spbw_mean, spbw_std = AudioUtil.get_spec(aud)
    fea = np.asarray([zcr_mean, zcr_std, spec_mean,
                     spec_std, spbw_mean, spbw_std], dtype=np.float32)
    return fea
