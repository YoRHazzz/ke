# KE

#### Description
A PyTorch implementation of Knowledge Graph Embedding
Implemented these models: TransE

#### Installation

1.  Install CUDA and cudnn, for example cuda==11.3, cudnn==8.2.1.32
2.  Install the appropriate version of PyTorch, see https://pytorch.org/get-started/previous-versions/
2.  Install tqdm pandas setproctitle
3.  Quick Start

```bash
# Install CUDA=11.3 cudnn=8.2.1.32
git clone https://gitee.com/YoRHazzz/ke.git
# Optional: Create a virtual environment
conda create -n ke python=3.8
conda activate ke

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install tqdm pandas setproctitle
python FB15K237_TransE.py
```

