import os
import numpy as np
import librosa
import soundfile as sf

# 数据增强文件
speed_factor1 = 1.1
speed_factor0 = 0.8
time_path1 = r'D:\Shilong\murmur\01_dataset\01_s1s2\trainset\time_stretch0.8'
time_path2 = r'D:\Shilong\murmur\01_dataset\01_s1s2\trainset\time_stretch1.1'
path = r'D:\Shilong\murmur\01_dataset\01_s1s2\trainset\present'
if not os.path.exists(time_path1):
    os.makedirs(time_path1)
if not os.path.exists(time_path2):
    os.makedirs(time_path2)

for root, dir, file in os.walk(path):
    for filename in file:
        print("processing "+filename)
        wav_path = os.path.join(root, filename)
        data, sr = librosa.load(wav_path, sr=4000)

        data_time_stretch = librosa.effects.time_stretch(
            data, rate=speed_factor1)
        sf.write(os.path.join(time_path1, filename+'_08.wav'), data, sr)

        data_time_stretch = librosa.effects.time_stretch(
            data, rate=speed_factor0)
        sf.write(os.path.join(time_path2, filename+'_11.wav'), data, sr)
