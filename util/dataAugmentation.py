import os
import numpy as np
import librosa
import soundfile as sf
from BEATs_def import wav_reverse
from pydub import AudioSegment

# 数据增强文件
speed_factor1 = 1.2
speed_factor0 = 0.9
# 原始数据path
path = r'D:\Shilong\murmur\01_dataset\01_s1s2\trainset\time_stretch0.8'
# 时域伸缩后保存的path
time_path1 = r'D:\Shilong\murmur\01_dataset\01_s1s2\trainset\time_stretch0.9'
time_path2 = r'D:\Shilong\murmur\01_dataset\01_s1s2\trainset\time_stretch1.2'
# 反转后保存的path
reverse_path = r'D:\Shilong\murmur\01_dataset\01_s1s2\trainset\reverse0.8'

if not os.path.exists(time_path1):
    os.makedirs(time_path1)
if not os.path.exists(time_path2):
    os.makedirs(time_path2)
if not os.path.exists(reverse_path):
    os.makedirs(reverse_path)

for root, dir, file in os.walk(path):
    for filename in file:
        print("processing "+filename)
        wav_path = os.path.join(root, filename)

        # 时域拉伸
        # data, sr = librosa.load(wav_path, sr=4000)
        # data_time_stretch = librosa.effects.time_stretch(
        #     data, rate=speed_factor1)
        # sf.write(os.path.join(time_path1, filename+'_09.wav'), data, sr)

        # data_time_stretch = librosa.effects.time_stretch(
        #     data, rate=speed_factor0)
        # sf.write(os.path.join(time_path2, filename+'_12.wav'), data, sr)

        # 倒放
        wav = AudioSegment.from_file(wav_path, format="wav")
        backplay = wav.reverse()
        # 存为相关格式倒放文件
        reverse_name = filename.split(".")[0]+"_reverse"
        backplay.export(reverse_path+"\\"+reverse_name+'.wav', format="wav")
