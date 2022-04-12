import os
import time
import numpy as np

from ke.data import KGMapping, KGDataset
from ke import fix_random
import torch
from torch.utils.data import DataLoader
from tqdm import trange

EPOCHS = 10000
MARGIN = 1.0
VECTOR_LENGTH = 50
LEARNING_RATE = 0.1
TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
FILTER_FLAG = True
USE_GPU = True

if __name__ == "__main__":
    fix_random()
    FB15K_path = os.path.join("..", "benchmarks", "FB15K")
    train_path = os.path.join(FB15K_path, "train.txt")
    fb15k_mapping = KGMapping(FB15K_path)

    fb15k_train_dataset = KGDataset(train_path, fb15k_mapping, filter_flag=True)
    fb15k_train_dataloader = DataLoader(fb15k_train_dataset, TRAIN_BATCH_SIZE)
    broken_range = set(range(fb15k_train_dataset.n_entity))

    device = torch.device('cuda') if USE_GPU else torch.device('cpu')

    for i in trange(EPOCHS, desc='Train Epochs', mininterval=0.5, unit='epochs'):
        for heads, relations, tails, head_or_tail, negative_entities in fb15k_train_dataloader:
            current_batch_size = len(heads)
            # head_or_tail = torch.randint(high=2, size=(current_batch_size,))
            # negative_entities = torch.randint(high=fb15k_train_dataset.n_entity, size=(current_batch_size,))
            # heads, relations, tails = heads.to(device), relations.to(device), tails.to(device)
            # head_or_tail, negative_entities = head_or_tail.to(device), negative_entities.to(device)

            positive_triplets = torch.stack((heads, relations, tails), dim=1)

            negative_heads = torch.where(head_or_tail == 1, negative_entities, heads)
            negative_tails = torch.where(head_or_tail == 0, negative_entities, tails)
            negative_triplets = torch.stack((negative_heads, relations, negative_tails), dim=1)
        fb15k_train_dataset.regenerate_head_or_tail()
