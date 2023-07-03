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
5. 遍历train/test文件夹，在***get_mfcc_features***中对wav
    - 重采样至16k
    - 使用**BEATs.extract_features(wav)[0]**提特征，并存为csv文件(**file_name: patientID_Poisition_num.wav.csv**)
    - 在***get_mel_features***读取每个csv,对不足24\*768的特征进行padding,对于大于24\*768的特征裁剪\[0:24,:\](**>>get_mfcc_features**)
    - 查询absent_id和present_id对每个file的ID添加Label(absent:1, present:0)
    - 返回**train_data,train_label,test_data,test_label**  (data和label存为.npy文件方便后续读取)
    - 通过MayDataset产生train-set和test-set
    - 通过dataloader生成train_loader，test_loader
### 模型定义
模型定义位于BEATs.py 的***class BEATs_Pre_Train_itere3(nn.Module)***
由于**BEATs.extract_features(wav)[0]**已经在上文执行，故此处只定义fine-tuning部分，
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



