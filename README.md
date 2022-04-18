# KE

#### 介绍
PyTorch实现知识图谱嵌入
已经实现了的模型：TransE

#### 安装教程

1. 安装 CUDA和cudnn，例如example cuda==11.3, cudnn==8.2.1.32
2. 安装对应版本的PyTorch，安装命令见https://pytorch.org/get-started/previous-versions/
2. 安装 tqdm pandas setproctitle
3. 示例

```bash
# Install CUDA=11.3 cudnn=8.2.1.32
git clone https://gitee.com/YoRHazzz/ke.git
# 可选项，创建虚拟环境
conda create -n ke python=3.8
conda activate ke

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install tqdm pandas setproctitle
python FB15K237_TransE.py
```

