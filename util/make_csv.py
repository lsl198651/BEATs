from re import A
from helper_code import *
from typing import Optional
import pandas as pd
"""
    此文件用于生成csv文件，用于后续的训练和测试
    valid和test没有csv，不便于产生数据集
    """


def define_csv(data_folder, csv_path: Optional[str] = None):
    patient_id = ['Patient ID']
    murmur = ['Murmur']
    recording_locations = ['Recording locations:']
    Murmur_locations = ['Murmur locations']
    Systolic_murmur_timing = ['Systolic murmur timing']
    Diastolic_murmur_timing = ['Diastolic murmur timing']
    Age = ['Age']
    Sex = ['Sex']
    height = ['Height']
    weight = ['Weight']
    patient_files = find_patient_files(data_folder)
    for i in range(len(patient_files)):
        current_patient_data = load_patient_data(patient_files[i])
        pid = get_patient_id(current_patient_data)
        label = get_murmur(current_patient_data)
        location = get_locations(current_patient_data)
        murmur_loca = get_murmur_locations(current_patient_data)
        systolic = get_systolic_murmur_timing(current_patient_data)
        diastolic = get_diastolic_murmur_timing(current_patient_data)
        a = get_age(current_patient_data)
        s = get_sex(current_patient_data)
        h = get_height(current_patient_data)
        w = get_weight(current_patient_data)

        patient_id.append(pid)
        murmur.append(label)
        recording_locations.append(location)
        Murmur_locations.append(murmur_loca)
        Systolic_murmur_timing.append(systolic)
        Diastolic_murmur_timing.append(diastolic)
        Age.append(a)
        Sex.append(s)
        height.append(h)
        weight.append(w)
    info = zip(patient_id, recording_locations, Age, Sex, height, weight, murmur,
               Murmur_locations, Systolic_murmur_timing, Diastolic_murmur_timing)
    pd.DataFrame(info).to_csv(f"{csv_path}.csv", index=False, header=False)
    # print(pid, label, location, murmur_loca,
    #       systolic, diastolic, a, s, h, w)
    # ========================/ setup loader /========================== #


if __name__ == '__main__':
    data_folder = r'D:\Shilong\murmur\Dataset\PCGdataset\test_data'
    define_csv(data_folder, csv_path='test_data')
