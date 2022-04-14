import torch
from torch import nn
import os


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.one_const = nn.Parameter(torch.tensor([1]), requires_grad=False)
        self.zero_const = nn.Parameter(torch.tensor([0]), requires_grad=False)

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(os.path.join(path)))
