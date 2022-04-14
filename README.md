# KE

#### 介绍
PyTorch实现知识图谱嵌入

已经实现了的模型：TransE

#### 安装教程

1. 安装 CUDA==11.3 cudnn=8.2.1.32
2. 示例

```bash
git clone https://gitee.com/YoRHazzz/ke.git
conda create -n ke python=3.8
conda activate ke
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install tqdm pandas
cd examples
python FB15K_TransE.py
```

