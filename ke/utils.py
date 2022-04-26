import argparse
import logging
import os
import random

import numpy as np
import torch


def fix_random(seed=1234):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_proc_title(title: str):
    if title is None:
        return
    try:
        import setproctitle
        setproctitle.setproctitle(title)
    except ImportError:
        print("Set process title failed, but it doesn't matter.")
        print("If you want to use this feature, try \npip install setproctitle\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--NORM", default=1, type=int)
    parser.add_argument("--MARGIN", default=3.0, type=float)
    parser.add_argument("--VECTOR_LENGTH", default=200, type=int)
    parser.add_argument("--LEARNING_RATE", default=0.0005, type=float)
    parser.add_argument("--EPOCHS", default=2000, type=int)
    parser.add_argument("--TRAIN_BATCH_SIZE", default=2048, type=int)
    parser.add_argument("--VALID_BATCH_SIZE", default=64, type=int)
    parser.add_argument("--TEST_BATCH_SIZE", default=64, type=int)
    parser.add_argument("--VALIDATE_FREQUENCY", default=100, type=int)
    parser.add_argument("--TARGET_METRIC", default="h10", type=str)
    parser.add_argument("--TARGET_SCORE", default=None, type=_none_or_float)
    parser.add_argument("--FILTER_FLAG", default=True, type=_bool)
    parser.add_argument("--USE_GPU", default=True, type=_bool)
    parser.add_argument("--GPU_INDEX", default=0, type=int)
    parser.add_argument("--SEED", default=1234, type=int)
    parser.add_argument("--PROC_TITLE", default=None, type=_none_or_str)
    parser.add_argument("--DATASET_PATH", default=os.path.join("benchmarks", "FB15K"), type=str)
    parser.add_argument("--CHECKPOINT_PATH", default=os.path.join("ckpt", "checkpoint.tar"), type=str)
    parser.add_argument("--LOG", default=True, type=_bool)
    parser.add_argument("--NUM_WORKERS", default=0, type=int)

    return parser.parse_args()


def _bool(s):
    return s == "True" or s == "1"


def _none_or_float(s):
    return None if s == "None" else float(s)


def _none_or_str(s):
    return None if s == "None" else s


def get_logger(logfile_name="log.txt"):
    if not os.path.isdir("log"):
        os.mkdir("log")

    log_path = os.path.join("log", logfile_name)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fmt = "%(asctime)-15s %(message)s"
    datefmt = "%H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt)
    fh.setFormatter(formatter)

    logger = logging.getLogger(logfile_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    return logger
