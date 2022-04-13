import math

from torch import nn
import torch.nn.functional as F


class TransE(nn.Module):
    def __init__(self, n_entities, n_relations, vector_length, p_norm=2):
        super(TransE, self).__init__()
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.vector_length = vector_length
        self.p_norm = p_norm

        self.entity_embeddings = nn.Embedding(self.n_entities, vector_length)
        self.relation_embeddings = nn.Embedding(self.n_relations, vector_length)
        initial_range = 6. / math.sqrt(self.vector_length)
        self.entity_embeddings.weight.data.uniform_(-initial_range, initial_range)
        self.relation_embeddings.weight.data.uniform_(-initial_range, initial_range)
        self.relation_embeddings.weight.data = F.normalize(self.relation_embeddings.weight.data, p=self.p_norm)

    def forward(self, triplets):
        self.entity_embeddings.weight.data = F.normalize(self.entity_embeddings.weight.data, p=self.p_norm)
        assert triplets.size()[1] == 3
        heads, relations, tails = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        return (self.entity_embeddings(heads) + self.relation_embeddings(relations) - self.entity_embeddings(tails))\
            .norm(p=self.p_norm, dim=1)
