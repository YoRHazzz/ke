# KE

## 介绍

PyTorch实现知识图谱嵌入
已经实现了的模型：TransE

## 安装教程

1. 安装 CUDA和cuDNN，例如 CUDA=11.3, cuDNN=8.2.1.32
2. 安装对应版本的PyTorch，安装命令见https://pytorch.org/get-started/previous-versions/
2. 安装 tqdm pandas setproctitle
3. 示例

```bash
# 安装 CUDA=11.3 cudnn=8.2.1.32
# 下载本仓库到本地
git clone https://gitee.com/YoRHazzz/ke.git
# 可选项，创建虚拟环境
conda create -n ke python=3.8
conda activate ke
# 安装第三方库
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install tqdm pandas setproctitle
# 用默认超参数训练
python TransE.py
```

5. 模型保存为 **ckpt/checkpoint.tar**，日志输出到 **log/Year-Month-Day.txt**。

## 实验结果

####  Link Prediction task on FB15K
| Source/Metric | Hits@1 (filter) | Hits@3 (filter) |  Hits@10 (filter)    | MR(filter) |  MRR (filter)    |
| ------------- | --------------- | --------------- | ---- | ---------------- | ---- |
|      TransE paper[[1]](#references)      |                 |                 | 47.1 | 125 |      |
| KE (Same HyperParameter) | 17.287 | 31.553 | 47.731 | 140.599 | 0.276 |
| KE (Better HyperParameter) | 38.793 | 70.424 | 84.012 | 37.108 | 0.565 |
|  | **Hits@1 (raw)** | **Hits@3 (raw)** | **Hits@10 (raw)** | **MR(raw)** | **MRR (raw)** |
| TransE paper[[1]](#references) |  |  | 34.9 | 243 | |
| KE (Same HyperParameter) | 8.410 | 18.670 | 34.693 | 260.572 | 0.170 |
| KE (Better HyperParameter) | 10.906 | 29.619 | 52.721 | 196.072 | 0.244 |

以上实验的超参数设置如下，这些脚本都保存在examples文件夹中：

```bash
# Same HyperParameter (filter)
python TransE.py --NORM=1 --MARGIN=1 --VECTOR_LENGTH=50 --LEARNING_RATE=0.01 --EPOCHS=1000 --VALIDATE_FREQUENCY=50 --FILTER_FLAG=True --USE_GPU=True --GPU_INDEX=0 --DATASET_PATH=./benchmarks/FB15K --CHECKPOINT_PATH=./ckpt/checkpoint.tar --TRAIN_BATCH_SIZE=50 --VALID_BATCH_SIZE=64 --TEST_BATCH_SIZE=64 --TARGET_METRIC=h10 --TARGET_SCORE=None --SEED=1234 --PROC_TITLE=Same_Hyperparameter --LOG=True --NUM_WORKERS=0
# Better HyperParameter (filter)
python TransE.py --NORM=1 --MARGIN=3 --VECTOR_LENGTH=200 --LEARNING_RATE=1 --EPOCHS=2000 --VALIDATE_FREQUENCY=50 --FILTER_FLAG=True --USE_GPU=True --GPU_INDEX=0 --DATASET_PATH=./benchmarks/FB15K --CHECKPOINT_PATH=./ckpt/checkpoint.tar --TRAIN_BATCH_SIZE=2048 --VALID_BATCH_SIZE=64 --TEST_BATCH_SIZE=64 --TARGET_METRIC=h10 --TARGET_SCORE=None --SEED=1234 --PROC_TITLE=Better_Hyperparameter --LOG=True --NUM_WORKERS=0
```

默认启用Filter模式，训练速度更慢但是模型效果更好。关闭Filter模式启用Raw模式的参数为：
```bash
--FILTER_FLAG=False
```

**请注意：FB15K已经过时并被弃用，此处使用FB15K仅为与原论文对比。现在更推荐使用FB15K237，参数如下：**

```bash
--DATASET_PATH=./benchmarks/FB15K-237.2
```

## 参考文献

[1] [Bordes et al., "Translating embeddings for modeling multi- relational data," in Adv. Neural Inf. Process. Syst., 2013](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf)