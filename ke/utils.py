import torch
import random
import numpy as np


def fix_random(seed=1234):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_proc_title(title:str):
    try:
        import setproctitle
        setproctitle.setproctitle(title)
    except ImportError:
        print("Set process title failed, but it doesn't matter.")
        print("If you want to use this feature, try pip install setproctitle")


