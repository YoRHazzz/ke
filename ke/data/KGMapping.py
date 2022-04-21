import os
from collections import Counter
import pandas as pd
import pickle


class KGMapping(object):
    def __init__(self, kg_path):
        assert os.path.exists(kg_path) and not os.path.isfile(kg_path)

        self.train_path = os.path.join(kg_path, "train.txt")
        self.valid_path = os.path.join(kg_path, "valid.txt")
        self.test_path = os.path.join(kg_path, "test.txt")
        assert os.path.exists(self.train_path)
        assert os.path.exists(self.valid_path)
        assert os.path.exists(self.test_path)

        mapping_path = os.path.join(kg_path, "mapping")
        if not os.path.exists(mapping_path):
            os.mkdir(mapping_path)

        self.entity2id_path = os.path.join(mapping_path, "entity2id.txt")
        self.relation2id_path = os.path.join(mapping_path, "relation2id.txt")
        self.h_of_rt_path = os.path.join(mapping_path, "h_of_rt.pkl")
        self.t_of_hr_path = os.path.join(mapping_path, "t_of_hr.pkl")
        self.entity2id = {}
        self.relation2id = {}
        if not os.path.exists(self.entity2id_path) or not os.path.exists(self.relation2id_path):
            entity_counter = Counter()
            relation_counter = Counter()
            for file_path in [self.train_path, self.valid_path, self.test_path]:
                with open(file_path, "r") as f:
                    for line in f:
                        h, r, t = line[:-1].split("\t")
                        entity_counter.update([h, t])
                        relation_counter.update([r])

            for idx, (entity, _) in enumerate(entity_counter.most_common()):
                self.entity2id[entity] = idx
            for idx, (relation, _) in enumerate(relation_counter.most_common()):
                self.relation2id[relation] = idx

            if os.path.exists(self.t_of_hr_path):
                os.remove(self.t_of_hr_path)
            if os.path.exists(self.h_of_rt_path):
                os.remove(self.h_of_rt_path)
            pd.DataFrame({'entity': self.entity2id.keys(), 'id': self.entity2id.values()}) \
                .to_csv(self.entity2id_path, index=False, sep='\t', header=False)
            pd.DataFrame({'relation': self.relation2id.keys(), 'id': self.relation2id.values()}) \
                .to_csv(self.relation2id_path, index=False, sep='\t', header=False)
        else:
            df = pd.read_csv(self.entity2id_path, sep='\t', header=None)
            for _, (entity, idx) in df.iterrows():
                self.entity2id[entity] = idx
            df = pd.read_csv(self.relation2id_path, sep='\t', header=None)
            for _, (relation, idx) in df.iterrows():
                self.relation2id[relation] = idx
            del df
        self.n_entity = len(self.entity2id)
        self.n_relation = len(self.relation2id)

        self.t_of_hr = {}
        self.h_of_rt = {}
        if not os.path.exists(self.h_of_rt_path) or not os.path.exists(self.t_of_hr_path):
            for file_path in [self.train_path, self.valid_path, self.test_path]:
                with open(file_path, "r") as f:
                    for line in f:
                        h, r, t = line[:-1].split("\t")
                        h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                        if (h, r) not in self.t_of_hr:
                            self.t_of_hr[(h, r)] = set()
                        self.t_of_hr[(h, r)].add(t)
                        if (r, t) not in self.h_of_rt:
                            self.h_of_rt[(r, t)] = set()
                        self.h_of_rt[(r, t)].add(h)
            with open(self.t_of_hr_path, 'wb') as f:
                pickle.dump(self.t_of_hr, f)
            with open(self.h_of_rt_path, 'wb') as f:
                pickle.dump(self.h_of_rt, f)
        else:
            with open(self.t_of_hr_path, 'rb') as f:
                self.t_of_hr = pickle.load(f)
            with open(self.h_of_rt_path, 'rb') as f:
                self.h_of_rt = pickle.load(f)
