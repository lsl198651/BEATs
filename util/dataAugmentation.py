import os
import numpy as np
import librosa
import soundfile as sf
from BEATs_def import wav_reverse, mkdir
from pydub import AudioSegment


def data_Auge():
    # 数据增强文件
    speed_factor0 = 1.2
    speed_factor1 = 1.1
    speed_factor2 = 0.9
    speed_factor3 = 0.8
    time_path0 = r'D:\Shilong\murmur\01_dataset\04_newDataset\trainset\time_stretch1.2'
    time_path1 = r'D:\Shilong\murmur\01_dataset\04_newDataset\trainset\time_stretch1.1'
    time_path2 = r'D:\Shilong\murmur\01_dataset\04_newDataset\trainset\time_stretch0.9'
    time_path3 = r'D:\Shilong\murmur\01_dataset\04_newDataset\trainset\time_stretch0.8'
    path = r'D:\Shilong\murmur\01_dataset\04_newDataset\trainset\present'
    if not os.path.exists(time_path0):
        os.makedirs(time_path0)
    if not os.path.exists(time_path1):
        os.makedirs(time_path1)
    if not os.path.exists(time_path2):
        os.makedirs(time_path2)
    if not os.path.exists(time_path3):
        os.makedirs(time_path3)

    for root, dir, file in os.walk(path):
        for filename in file:
            print("processing "+filename)
            wav_path = os.path.join(root, filename)
            data, sr = librosa.load(wav_path, sr=4000)

            data_time_stretch = librosa.effects.time_stretch(
                data, rate=speed_factor0)
            sf.write(os.path.join(time_path0, filename +
                     '_12.wav'), data_time_stretch, sr)

            data_time_stretch = librosa.effects.time_stretch(
                data, rate=speed_factor1)
            sf.write(os.path.join(time_path1, filename +
                     '_11.wav'), data_time_stretch, sr)

            data_time_stretch = librosa.effects.time_stretch(
                data, rate=speed_factor2)
            sf.write(os.path.join(time_path2, filename +
                     '_09.wav'), data_time_stretch, sr)

            data_time_stretch = librosa.effects.time_stretch(
                data, rate=speed_factor3)
            sf.write(os.path.join(time_path3, filename +
                     '_08.wav'), data_time_stretch, sr)

    # 反转后保存的path
    reverse_path0 = r'D:\Shilong\murmur\01_dataset\04_newDataset\trainset\reverse1.2'
    reverse_path1 = r'D:\Shilong\murmur\01_dataset\04_newDataset\trainset\reverse1.1'
    reverse_path2 = r'D:\Shilong\murmur\01_dataset\04_newDataset\trainset\reverse0.9'
    reverse_path3 = r'D:\Shilong\murmur\01_dataset\04_newDataset\trainset\reverse0.8'
    mkdir(reverse_path0)
    mkdir(reverse_path1)
    mkdir(reverse_path2)
    mkdir(reverse_path3)

    for root, dir, file in os.walk(time_path0):
        for filename in file:
            print("processing "+filename)
            wav_path = os.path.join(root, filename)

            # 倒放
            wav = AudioSegment.from_file(wav_path, format="wav")
            backplay = wav.reverse()
            # 存为相关格式倒放文件
            reverse_name = filename.split(".")[0]+"_reverse"
            backplay.export(reverse_path0+"\\" +
                            reverse_name+'.wav', format="wav")

    for root, dir, file in os.walk(time_path1):
        for filename in file:
            print("processing "+filename)
            wav_path = os.path.join(root, filename)

            # 倒放
            wav = AudioSegment.from_file(wav_path, format="wav")
            backplay = wav.reverse()
            # 存为相关格式倒放文件
            reverse_name = filename.split(".")[0]+"_reverse"
            backplay.export(reverse_path1+"\\" +
                            reverse_name+'.wav', format="wav")

    for root, dir, file in os.walk(time_path2):
        for filename in file:
            print("processing "+filename)
            wav_path = os.path.join(root, filename)

            # 倒放
            wav = AudioSegment.from_file(wav_path, format="wav")
            backplay = wav.reverse()
            # 存为相关格式倒放文件
            reverse_name = filename.split(".")[0]+"_reverse"
            backplay.export(reverse_path2+"\\" +
                            reverse_name+'.wav', format="wav")

    for root, dir, file in os.walk(time_path3):
        for filename in file:
            print("processing "+filename)
            wav_path = os.path.join(root, filename)

            # 倒放
            wav = AudioSegment.from_file(wav_path, format="wav")
            backplay = wav.reverse()
            # 存为相关格式倒放文件
            reverse_name = filename.split(".")[0]+"_reverse"
            backplay.export(reverse_path3+"\\" +
                            reverse_name+'.wav', format="wav")
