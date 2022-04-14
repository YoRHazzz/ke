import random
import numpy as np

from torch.utils.data import Dataset
from .KGMapping import KGMapping


class KGDataset(Dataset):
    def __init__(self, file_path, kg_mapping: KGMapping, filter_flag=False):
        super(KGDataset, self).__init__()
        self.kg_mapping = kg_mapping
        self.filter_flag = filter_flag

        entity2id, relation2id = self.kg_mapping.entity2id, self.kg_mapping.relation2id
        self.data = []
        with open(file_path, "r") as f:
            for line in f:
                h, r, t = line[:-1].split("\t")
                h, r, t = entity2id[h], relation2id[r], entity2id[t]
                self.data.append((h, r, t))

        self.n_entity = kg_mapping.n_entity
        self.n_relation = kg_mapping.n_relation
        self.n_triplet = len(self.data)
        self.head_or_tail = np.random.randint(low=0, high=2, size=(self.n_triplet,))

    def __len__(self):
        return self.n_triplet

    def __getitem__(self, index):
        if self.filter_flag:
            h, r, t = self.data[index]
            negative_index = random.randint(0, self.n_entity - 1)
            if self.head_or_tail[index] == 0:
                while negative_index in self.kg_mapping.h_of_rt[(r, t)]:
                    negative_index = random.randint(0, self.n_entity - 1)
            else:
                while negative_index in self.kg_mapping.t_of_hr[(h, r)]:
                    negative_index = random.randint(0, self.n_entity - 1)
            return *self.data[index], self.head_or_tail[index], negative_index
        else:
            return self.data[index]

    def regenerate_head_or_tail(self):
        self.head_or_tail = np.random.randint(low=0, high=2, size=(self.n_triplet,))
