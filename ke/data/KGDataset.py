import random

from torch.utils.data import Dataset
from .KGMapping import KGMapping


class KGDataset(Dataset):
    def __init__(self, file_path, kg_mapping: KGMapping, neg_sample_flag=False, filter_flag=False):
        self.kg_mapping = kg_mapping
        self.neg_sample_flag = neg_sample_flag
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
        self.entity_set = set(range(self.n_entity))

    def __len__(self):
        return self.n_triplet

    def __getitem__(self, index):
        if not self.neg_sample_flag:
            return self.data[index]
        else:
            h, r, t = self.data[index]
            head_or_tail = random.randint(0, 1)
            if head_or_tail == 0:
                negative_head = self.generate_negative(self.kg_mapping.h_of_rt[(r, t)])
                corrupted_triplet = (negative_head, r, t)
            else:
                negative_tail = self.generate_negative(self.kg_mapping.t_of_hr[(h, r)])
                corrupted_triplet = (h, r, negative_tail)
            return *self.data[index], *corrupted_triplet

    def generate_negative(self, illegal_set):
        negative_index = random.randint(0, self.n_entity - 1)
        if self.filter_flag:
            while negative_index in illegal_set:
                negative_index = random.randint(0, self.n_entity - 1)
        return negative_index
