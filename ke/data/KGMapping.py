import os
import pickle
import shutil
from collections import Counter

import pandas as pd


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
        self.rt2h_path = os.path.join(mapping_path, "rt2h.pkl")
        self.hr2t_path = os.path.join(mapping_path, "hr2t.pkl")
        self.h2t_path = os.path.join(mapping_path, "h2t.pkl")
        self.t2h_path = os.path.join(mapping_path, "t2h.pkl")
        self.entity2id = {}
        self.relation2id = {}
        if not os.path.exists(self.entity2id_path) or not os.path.exists(self.relation2id_path):
            shutil.rmtree(mapping_path)
            os.mkdir(mapping_path)
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

            if os.path.exists(self.hr2t_path):
                os.remove(self.hr2t_path)
            if os.path.exists(self.rt2h_path):
                os.remove(self.rt2h_path)
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

        self.hr2t = {}
        self.rt2h = {}
        self.h2t = {}
        self.t2h = {}
        if not os.path.exists(self.rt2h_path) or not os.path.exists(self.hr2t_path) \
                or not os.path.exists(self.h2t_path) or not os.path.exists(self.t2h_path):
            for file_path in [self.train_path, self.valid_path, self.test_path]:
                with open(file_path, "r") as f:
                    for line in f:
                        h, r, t = line[:-1].split("\t")
                        h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                        if (h, r) not in self.hr2t:
                            self.hr2t[(h, r)] = set()
                        self.hr2t[(h, r)].add(t)
                        if (r, t) not in self.rt2h:
                            self.rt2h[(r, t)] = set()
                        self.rt2h[(r, t)].add(h)
                        if h not in self.h2t:
                            self.h2t[h] = set()
                        self.h2t[h].add(t)
                        if t not in self.t2h:
                            self.t2h[t] = set()
                        self.t2h[t].add(h)
            with open(self.hr2t_path, 'wb') as f:
                pickle.dump(self.hr2t, f)
            with open(self.rt2h_path, 'wb') as f:
                pickle.dump(self.rt2h, f)
            with open(self.h2t_path, 'wb') as f:
                pickle.dump(self.h2t, f)
            with open(self.t2h_path, 'wb') as f:
                pickle.dump(self.t2h, f)
        else:
            with open(self.hr2t_path, 'rb') as f:
                self.hr2t = pickle.load(f)
            with open(self.rt2h_path, 'rb') as f:
                self.rt2h = pickle.load(f)
            with open(self.h2t_path, 'rb') as f:
                self.h2t = pickle.load(f)
            with open(self.t2h_path, 'rb') as f:
                self.t2h = pickle.load(f)
