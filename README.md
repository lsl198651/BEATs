# 关于BEATs模型的使用
## 主要文件说明
1. BEATs.py 官方BEATs模型
2. BEATs.py 个人定义的function文件，用于提取特征等操作
3. BEATs_try.py 主要程序运行文件，包括数据处理和模型训练
4. BEATs_try.ipynb 用于测试函数功能

## 程序流程说明
### 文件操作
1. 将文件按照Patient ID创建文件夹并把听诊区wav文件和tsv文件存入文件夹
2. 在每个Patient ID文件下创建听诊区文件，按照tsv文件的分割点对wav文件进行分割为单一心跳wav(1[0]--4[1])存入听诊区文件夹(**file_name: patient_ID_Poisition_num.wav**)
3. load “Patient ID”，按照7：3 将patient ID划分w为train-set/test-set并存为CSV >>absnet_train/test_id.csv, present_train/test_id.csv
4. 按照CSV将wav文件存入train/test文件夹 (**>>copy_wav**)
5. 遍历train/test文件夹，在***get_mfcc_features***中对wav进行操作
    - 重采样至16k
    - 设定每条wav数据长度为**10000**对长度不足10000的wav数据padding,对于大于10000的特征裁剪,\[0:10000\](**>>get_mfcc_features**)
    - 将裁剪后的wav数据保存为单独的.csv文件并append到列表中，最后返回一个包含所有wav数据的数组
    - **保存返回的array数组为.npy文件,下次直接读取**
6. 返回**train_data，test_data**  并产生label（ Absent=1，Present=0）
7. 通过MayDataset产生**train-set,test-set**
8. 通过dataloader生成**train_loader，test_loader**
### 模型定义
模型定义位于BEATs.py 的***class BEATs_Pre_Train_itere3(nn.Module)***
由于 **BEATs\.extract\_features\(wav\)\[0\]** 已经在上文执行，故此处只定义fine-tuning部分，
- 首先添加checkpoint,导入BEATs的配置
- 其次定于全连接层和Dropout
定义forward函数：

```python
def forward(self,x):
        y=self.last_Dropout(x)
        # FC
        output=self.last_layer(y)
        # mean
        output=output.mean(dim=1)
        # sigmoid
        output=torch.sigmoid(output)
        return output
```
定于训练，测试函数并进行训练



