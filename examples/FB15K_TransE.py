import os

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


if __name__ == "__main__":
    fix_random()
    FB15K_path = os.path.join("..", "benchmarks", "FB15K")
    train_path = os.path.join(FB15K_path, "train.txt")
    fb15k_mapping = KGMapping(FB15K_path)

    fb15k_train_dataset = KGDataset(train_path, fb15k_mapping, neg_sample_flag=True, filter_flag=True)
    fb15k_train_dataloader = DataLoader(fb15k_train_dataset, TRAIN_BATCH_SIZE)
    broken_range = set(range(fb15k_train_dataset.n_entity))

    for i in trange(EPOCHS, desc='Train Epochs', mininterval=0.5, unit='epochs'):
        for head, relation, tail, broken_head, broken_relation, broken_tail in fb15k_train_dataloader:
            positive_triplets = torch.stack((head, relation, tail), dim=1)
            negative_triplets = torch.stack((broken_head, broken_relation, broken_tail), dim=1)
            current_batch_size = len(head)
