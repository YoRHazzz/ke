# KE

## Description
A PyTorch implementation of Knowledge Graph Embedding
Implemented these models: TransE

## Installation

1.  Install CUDA and cuDNN, for example CUDA=11.3, cuDNN=8.2.1.32
2.  Install the appropriate version of PyTorch, see https://pytorch.org/get-started/previous-versions/
2.  Install tqdm pandas setproctitle
3.  Quick Start

```bash
# Install CUDA=11.3 cuDNN=8.2.1.32
git clone https://gitee.com/YoRHazzz/ke.git
# Optional: Create a virtual environment
conda create -n ke python=3.8
conda actInstalling third-party Librariesivate ke
# Install Libraries
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install tqdm pandas setproctitle
# Train with default hyperparameters
python TransE.py
```

5. The model is saved as **ckpt/checkpoint.tar**, and logs are output to **log/Year-Month-Day.txt**.

## Experiments

####  Link Prediction task on FB15K

| Source/Metric                  | Hits@1 (filter)  | Hits@3 (filter)  | Hits@10 (filter)  | MR(filter)  | MRR (filter)  |
| ------------------------------ | ---------------- | ---------------- | ----------------- | ----------- | ------------- |
| TransE paper[[1]](#references) |                  |                  | 47.1              | 125         |               |
| KE (Same HyperParameter)       |                  |                  |                   |             |               |
| KE (Better HyperParameter)     | 38.793           | 70.424           | 84.012            | 37.108      | 0.565         |
|                                | **Hits@1 (raw)** | **Hits@3 (raw)** | **Hits@10 (raw)** | **MR(raw)** | **MRR (raw)** |
| TransE paper[[1]](#references) |                  |                  | 34.9              | 243         |               |
| KE (Same HyperParameter)       |                  |                  |                   |             |               |
| KE (Better HyperParameter)     | 10.906           | 29.619           | 52.721            | 196.072     | 0.244         |

All hyperparameter Settings are as follows：

```bash
# Same HyperParameter (filter)
python TransE.py --NORM=1 --MARGIN=1 --VECTOR_LENGTH=50 --LEARNING_RATE=0.01 --EPOCHS=1000 --VALIDATE_FREQUENCY=50 --FILTER_FLAG=True --USE_GPU=True --GPU_INDEX=0 --DATASET_PATH=./benchmarks/FB15K --CHECKPOINT_PATH=./ckpt/checkpoint.tar --TRAIN_BATCH_SIZE=64 --VALID_BATCH_SIZE=64 --TEST_BATCH_SIZE=64 --TARGET_METRIC=h10 --TARGET_SCORE=None --SEED=1234 --PROC_TITLE=Same_Hyperparameter --LOG=True
# Better HyperParameter (filter)
python TransE.py --NORM=1 --MARGIN=3 --VECTOR_LENGTH=200 --LEARNING_RATE=1 --EPOCHS=2000 --VALIDATE_FREQUENCY=50 --FILTER_FLAG=True --USE_GPU=True --GPU_INDEX=0 --DATASET_PATH=./benchmarks/FB15K --CHECKPOINT_PATH=./ckpt/checkpoint.tar --TRAIN_BATCH_SIZE=2048 --VALID_BATCH_SIZE=64 --TEST_BATCH_SIZE=64 --TARGET_METRIC=h10 --TARGET_SCORE=None --SEED=1234 --PROC_TITLE=Better_Hyperparameter --LOG=True
```

The Filter mode is enabled by default. Hence, the training speed is slower but the model effect is better. To disable filter and enable the raw mode, set the argument as follows:

```bash
--FILTER_FLAG=False
```

**Also note that: FB15K is deprecated. FB15K is used here only for comparison with the original paper. Now it is recommended to use FB15K237. Set the argument as follows: **

```bash
--DATASET_PATH=./benchmarks/FB15K-237.2
```

## Reference

[1] [Bordes et al., "Translating embeddings for modeling multi- relational data," in Adv. Neural Inf. Process. Syst., 2013](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf)
