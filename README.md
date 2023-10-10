# 关于 BEATs 模型的使用

## 主要文件说明

1. BEATs.py 官方 BEATs 模型
2. BEATs.py 个人定义的 function 文件，用于提取特征等操作
3. BEATs_try.py 主要程序运行文件，包括数据处理和模型训练
4. BEATs_def.py 定义部分功能函数
5. BEATs_try.ipynb 用于测试函数功能
6. BEATs_run.py 用于运行训练和测试模型

## 程序流程说明

### 文件操作

1. 将文件按照 Patient ID 创建文件夹并把听诊区 wav 文件和 tsv 文件存入文件夹
2. 在每个 Patient ID 文件下创建听诊区文件，按照 tsv 文件的分割点对 wav 文件进行分割为收缩期心跳和舒张期心跳存入听诊区文件夹(**file\*name: patient_ID_Poisition\_收缩/舒张\_num_Present\\Absent_timming.wav**)
3. load “Patient ID”，按照 8:2 将 patient ID 划分为 train-set/test-set 并存为 CSV >>absnet_train/test_id.csv, present_train/test_id.csv
4. 按照 CSV 将 wav 文件存入 train/test 文件夹 (**>>copy_wav**)[此步取消]
5. 遍历 train/test 文件夹，在 **get_wav_data**中对 wav 进行操作
   - 重采样至 16k
   - 设定每条 wav 数据长度为**3500**对长度不足 3500 的 wav 数据 padding,对于大于 3500 的特征裁剪,\[0:3500\]\( **\>\>get_mfcc_features** \)
   - 将裁剪后的 wav 数据保存为单独的.csv 文件并 append 到列表中，最后返回一个包含所有 wav 数据的数组
   - **保存 wav 返回的 array 数组和 label 为.npy ,下次直接读取**
6. 设置超参数，如 masking、调整训练和测试数据集大小，Absent\/Present 比例、lr 等细节。
7. 通过 MayDataset 和 dataloader 生成**train_loader，test_loader**
8. 训练模型 9.保存模型结果等信息

### 模型定义

模型定义位于 BEATs.py 的**_class BEATs_Pre_Train_itere3(nn.Module)_**
由于 **BEATs\.extract_features\(wav\)\[0\]** 已经在上文执行，故此处只定义 fine-tuning 部分，

- 首先添加 checkpoint,导入 BEATs 的配置
- 其次定于全连接层和 Dropout
  定义 forward 函数：

```python
def forward(self, x, padding_mask: torch.Tensor = None):
    # with torch.no_grad():
    x, _ = self.BEATs.extract_features(x, padding_mask)
    # dropout
    # with torch.enable_grad():
    y = self.last_Dropout(x)

    y = self.fc_layer(y)
    # add fc layer
    output = self.last_layer(y)
    # mean
    output = output.mean(dim=1)
    # sigmoid
    # output = torch.sigmoid(output)
    return output
```

用于训练，测试函数并进行训练

### 模型相关说明

- 冻结参数所使用的方法为\:
  ```python
  param.requires_grad = False
  ```
- 时域和频域 masking 方法为：

  ```python
  freqm = TT.FrequencyMasking(freqm)
  timem = TT.TimeMasking(timem)
  ```

- 学习率设为可调，如 cosine 函数

### 数据集生成：

1. make dataset
2. save as npy file
