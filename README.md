# MLKD2020-Classification-and-identification-of-musical-emotions

**音乐情感分类**
==============================================================

1 **运行环境**
--------------------------------------------------------------
* Windows 10 64-bit
* python >= 3.7.0

### 1.1 主要使用的库:

* PySide2==5.15.2
* metric-learn==0.6.2
* scikit-learn==0.23.2
* scipy==1.5.4
* librosa==0.8.0
* pydub==0.24.1

全部库依赖和版本详见 [requirement.txt](requirement.txt).

### 1.2 环境变量修改

pydub库只支持原生的wav格式的文件处理, librosa库无法直接读取mp3格式的文件, 均需要下载[FFmpeg](https://ffmpeg.org/), 将其安装目录下的bin文件夹添加到系统的Path环境变量中方可使用.

2 **代码文件说明**
--------------------------------------------------------------
* 音乐的重命名脚本为[code/rename.py](code/rename.py), 分割脚本为[code/split.py](code/split.py), 特征提取函数在脚本[code/Extract_feature.py](code/Extract_feature.py)中, 从数据集中提取特征并保存的脚本为[code/data_prepare.py](code/data_prepare.py), 几乎所有分类器都包含于脚本[code/classifier.py](code/classifier.py)中, 高斯混合模型在脚本[code/GMM.py](code/GMM.py)中实现, 流形正则化模型在脚本[code/MR.py](code/MR.py)中.

* [code](./code)文件夹中的[code/run.py](code/run.py)可以测试除S3VM和MR以外的算法分类精度, S3VM使用[code/S3VM.py](code/S3VM.py)测试

* 用户界面(GUI)由[main/main.py](main/main.py) 和 [main/knn_classifier.py](main/knn_classifier.py)两个脚本实现,  使用方法参照第 4 节.

3 **数据集**
--------------------------------------------------------------
### 3.1 原始数据集
* 本项目采用mp3格式的音频数据, 每30s截取片段作为特征提取数据, 对数据进行情感标注.项目采用的有标签音乐片段共857个, 对全部数据特征提取后的特征数组和标签存储在npy格式文件中, 这些文件存在[data/multi/numpy](data/multi/numpy)文件夹下.其中所有音乐片段总大小为982M, 保存时域数据的'data.npy'文件大小为3.16GB, 单独给出[here](https://jbox.sjtu.edu.cn/l/Y0TeIM).直接分离训练集和测试集的npy格式数据在[data/multi/labeled](data/multi/labeled)文件夹下.

### 3.2 数据集扩展
* 若想扩展有标签数据集, 请根据命名规则:序号-情感-歌名.mp3, 对文件进行重命名, 可将同一情感数据放在一个文件夹下, 用提供的'./rename.py'对其进行批量重命名.分割之后进行特征提取并保存在npy文件中.若想扩展无标签数据集, 可直接分割并提取特征保存.

* PS:将脚本中的文件夹路径改为你所要的路径.

4 **用户程序使用方法**
--------------------------------------------------------------
### 4.1 启动
* 如果需要测试的文件在默认文件夹, 即运行目录下的./test, 请命令行调用  
```bash
cd ./main
python main.py -path-test ./test
```
* 如果需要测试的mp3文件不在默认文件夹, 请输入
```bash
cd ./main
python main.py -path-test path_to_test_folder
```

### 4.2 训练
* 先选择是否使用无标签数据集(默认不使用).  
* 单击 ```train the model now!``` 按钮, 开始训练. 

### 4.3 查询相似音乐和情绪成分分析
* 选择 ```Find Similar Music``` 选项卡,  
* 在第一个输出框输入需要查询的文件(文件需提前放入测试文件夹中, 默认 ./test, 如 2.1 所述.)  注意不要输入空格.
>* 如果查询.mp3文件, 输入文件名即可. 注意, 文件名不能是纯整型数, 如果文件名完全由整数组成, 请确保输入扩展名.
>* 如果查询预处理好的测试集中的音乐, 请输入一个整数, 即该音乐的特征在feature.npy文件中的序号. 例子:
>>* 预处理测试集中的第10首音乐:输入 10  
>>* 处理名为"Music.mp3"的音乐, 输入 Music 或 Music.mp3
>>* 处理名为"10.mp3"的文件, 输入 10.mp3
* 在第二个输入框输入需要考虑的邻居数目, 必须为正整数.  
* 单击 ```Start Query Now!``` 按钮, 开始查询. 

### 4.4 随机查询训练数据集中某种情绪的音乐
* 选择 ```Get Music with giving emotion``` 选项卡,  
* 在下拉菜单中选择要查询的情绪. 仅支持上文所述的6种情绪. 
* 在输入框中输入需要返回的片段数目, 必须为正整数. 
* 单击 ```Start Query Now!``` 按钮, 开始查询. 

### 4.5 实现方式
* 只使用有标签数据的情况下, 采用kNN算法, 度量采用LSMN生成的马氏度量.  
* 联合使用无标签数据时, 通过GMM算法生成伪标签, 再使用同样的kNN算法.  
* 情绪成分分析的结果是根据邻居们到测试数据的距离, 取负数, 计算softmax得到的. 

### 4.6 数据集扩展
* 如果需要扩展用户程序的数据集, 请按照 3.2 节的方法处理数据, 并以相同的文件名保存在train_labeled, train_unlabel, test文件夹中. 
