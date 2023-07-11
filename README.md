# 关于 BEATs 模型的使用

## 主要文件说明

1. BEATs.py 官方 BEATs 模型
2. BEATs.py 个人定义的 function 文件，用于提取特征等操作
3. BEATs_try.py 主要程序运行文件，包括数据处理和模型训练
4. BEATs_try.ipynb 用于测试函数功能

## 程序流程说明

### 文件操作

1. 将文件按照 Patient ID 创建文件夹并把听诊区 wav 文件和 tsv 文件存入文件夹
2. 在每个 Patient ID 文件下创建听诊区文件，按照 tsv 文件的分割点对 wav 文件进行分割为收缩期心跳和舒张期心跳存入听诊区文件夹(**file*name: patient_ID_Poisition*收缩/舒张\_num.wav**)
3. load “Patient ID”，按照 8:2 将 patient ID 划分为 train-set/test-set 并存为 CSV >>absnet_train/test_id.csv, present_train/test_id.csv
4. 按照 CSV 将 wav 文件存入 train/test 文件夹 (**>>copy_wav**)
5. 遍历 train/test 文件夹，在**_get_mfcc_features_**中对 wav 进行操作
   - 重采样至 16k
   - 设定每条 wav 数据长度为**10000**对长度不足 10000 的 wav 数据 padding,对于大于 10000 的特征裁剪,\[0:10000\](**>>get_mfcc_features**)
   - 将裁剪后的 wav 数据保存为单独的.csv 文件并 append 到列表中，最后返回一个包含所有 wav 数据的数组
   - **保存返回的 array 数组为.npy 文件,下次直接读取**
6. 返回**train_data，test_data** 并产生 label（ Absent=1，Present=0）
7. 通过 MayDataset 产生**train-set,test-set**
8. 通过 dataloader 生成**train_loader，test_loader**

### 模型定义

模型定义位于 BEATs.py 的**_class BEATs_Pre_Train_itere3(nn.Module)_**
由于 **BEATs\.extract_features\(wav\)\[0\]** 已经在上文执行，故此处只定义 fine-tuning 部分，

- 首先添加 checkpoint,导入 BEATs 的配置
- 其次定于全连接层和 Dropout
  定义 forward 函数：

```python
def forward(self,x):
    with torch.no_grad():
        x=self.extract_features(x)[0]
    y=self.last_Dropout(x)
    # FC
    output=self.last_layer(y)
    # mean
    output=output.mean(dim=1)
    # sigmoid
    output=torch.sigmoid(output)
    return output
```

用于训练，测试函数并进行训练
