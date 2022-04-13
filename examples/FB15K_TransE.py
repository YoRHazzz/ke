import os
import time
import numpy as np

from ke.data import KGMapping, KGDataset
from ke import fix_random
import torch
from torch.utils.data import DataLoader
from tqdm import trange
from ke.model import TransE
from torch import nn
from torch import optim

EPOCHS = 10000
MARGIN = 1.0
NORM = 2
VECTOR_LENGTH = 50
LEARNING_RATE = 0.1
TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
FILTER_FLAG = False
USE_GPU = False

CHECK_NEGATIVE = False

if __name__ == "__main__":
    fix_random()
    FB15K_path = os.path.join("..", "benchmarks", "FB15K")
    train_path = os.path.join(FB15K_path, "train.txt")
    fb15k_mapping = KGMapping(FB15K_path)

    fb15k_train_dataset = KGDataset(train_path, fb15k_mapping, filter_flag=FILTER_FLAG)
    fb15k_train_dataloader = DataLoader(fb15k_train_dataset, TRAIN_BATCH_SIZE)

    device = torch.device('cuda') if USE_GPU else torch.device('cpu')

    transe = TransE(fb15k_train_dataset.n_entity, fb15k_train_dataset.n_relation, VECTOR_LENGTH, p_norm=NORM)
    transe = transe.to(device)
    loss_fn = nn.MarginRankingLoss(margin=MARGIN, reduction='none')
    optimizer = optim.SGD(transe.parameters(), lr=LEARNING_RATE)

    pb = trange(EPOCHS, desc='Train Epochs', maxinterval=1, unit='epochs')
    postfix = {"epoch_total_loss": 0, "batch_loss": 0}
    for epoch in pb:
        epoch_total_loss = torch.tensor(0.)
        epoch_total_loss = epoch_total_loss.to(device)

        for data in fb15k_train_dataloader:
            current_batch_size = len(data[0])
            if FILTER_FLAG:
                heads, relations, tails, head_or_tail, negative_entities = data
            else:
                heads, relations, tails = data
                head_or_tail = torch.randint(high=2, size=(current_batch_size,))
                negative_entities = torch.randint(high=fb15k_train_dataset.n_entity, size=(current_batch_size,))

            heads, relations, tails = heads.to(device), relations.to(device), tails.to(device)
            head_or_tail, negative_entities = head_or_tail.to(device), negative_entities.to(device)

            positive_triplets = torch.stack((heads, relations, tails), dim=1)

            negative_heads = torch.where(head_or_tail == 0, negative_entities, heads)
            negative_tails = torch.where(head_or_tail == 1, negative_entities, tails)
            negative_triplets = torch.stack((negative_heads, relations, negative_tails), dim=1)
            if CHECK_NEGATIVE:
                for i in range(current_batch_size):
                    h, r, t = heads[i].item(), relations[i].item(), tails[i].item()
                    bh, bt = negative_heads[i].item(), negative_tails[i].item()
                    try:
                        assert t in fb15k_mapping.t_of_hr[(h, r)]
                        assert h in fb15k_mapping.h_of_rt[(r, t)]
                        assert (bh in fb15k_mapping.h_of_rt[(r, t)]) ^ (bt in fb15k_mapping.t_of_hr[(h, r)])
                        assert bh == h or bt == t
                    except:
                        print(h, r, t, bh, bt)

            optimizer.zero_grad()
            positive_distance = transe(positive_triplets)
            negative_distance = transe(negative_triplets)
            loss = loss_fn(positive_distance, negative_distance, torch.tensor([-1]))
            loss.mean().backward()
            optimizer.step()
            postfix["batch_loss"] = loss.mean().cpu().item()
            pb.set_postfix(postfix)
            epoch_total_loss += loss.sum()
        postfix["epoch_total_loss"] = epoch_total_loss.cpu().item()
        pb.set_postfix(postfix)
    fb15k_train_dataset.regenerate_head_or_tail()
