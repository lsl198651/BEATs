import os
import numpy as np
import librosa
import soundfile as sf

# 数据增强文件
speed_factor = 1.2
time_path = r'D:\Shilong\murmur\03_circor_states\trainset\volume2'
path = r'D:\Shilong\murmur\03_circor_states\trainset\present'
for root, dir, file in os.walk(path):
    for filename in file:
        print("processing "+filename)
        wav_path = os.path.join(root, filename)
        data, sr = librosa.load(wav_path, sr=None)
        data_time_stretch = librosa.effects.time_stretch(
            data, rate=speed_factor)
        sf.write(os.path.join(time_path, filename+'_.wav'), data, sr)
