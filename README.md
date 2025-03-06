# SB-SAM

### 介绍
**A score-based small atomic model with application to stress calculation**

SB-SAM 是一个基于 Score Matching 的分子动力学模拟框架，旨在用于应力计算。

### 文章链接
[请补充文章链接]()


### 数据生成
本项目采用分子动力学方法，模拟 NVT 系综下的粒子分布，并将其作为训练样本。

数据生成代码存储于 ./md 目录

相关理论可参考：《Modeling Materials: Continuum, Atomistic and Multiscale Techniques》


### 模型训练
本模型基于 PyTorch 框架，参数配置文件存储于 ./config/paramsxx.yml

可通过以下命令运行训练过程：

```python
python main.py --config=./config/paramsxx.yml
