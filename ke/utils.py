import torch
import random
import numpy as np
import argparse
import os


def fix_random(seed=1234):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_proc_title(title: str):
    try:
        import setproctitle
        setproctitle.setproctitle(title)
    except ImportError:
        print("Set process title failed, but it doesn't matter.")
        print("If you want to use this feature, try \npip install setproctitle\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--NORM", default=1)
    parser.add_argument("--MARGIN", default=5.0)
    parser.add_argument("--VECTOR_LENGTH", default=200)
    parser.add_argument("--LEARNING_RATE", default=1.)
    parser.add_argument("--EPOCHS", default=1000)
    parser.add_argument("--TRAIN_BATCH_SIZE", default=4096)
    parser.add_argument("--VALID_BATCH_SIZE", default=64)
    parser.add_argument("--TEST_BATCH_SIZE", default=64)
    parser.add_argument("--VALIDATE_FREQUENCY", default=50)
    parser.add_argument("--TARGET_METRIC", default="h10")
    parser.add_argument("--TARGET_SCORE", default=None)
    parser.add_argument("--FILTER_FLAG", default=True)
    parser.add_argument("--USE_GPU", default=True)
    parser.add_argument("--GPU_INDEX", default=0)
    parser.add_argument("--SEED", default=1234)
    parser.add_argument("--PROC_TITLE", default="TransE")
    parser.add_argument("--DATASET_PATH", default=os.path.join("benchmarks", "FB15K-237.2"))
    parser.add_argument("--CHECKPOINT_PATH", default=os.path.join("ckpt", "FB15K-237.2_TransE.checkpoint.tar"))

    return parser.parse_args()
