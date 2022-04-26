import torch
from torch import nn
from torch.utils.data import DataLoader


class Tester(object):
    def __init__(self, model: nn.Module, data_loader: DataLoader, device: torch.device, filter_flag: bool):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.filter_flag = filter_flag

    def link_prediction(self):
        self.model.eval()
        n_entity = self.data_loader.dataset.n_entity
        hits_at_1 = 0.
        hits_at_3 = 0.
        hits_at_10 = 0.
        mr_score = 0.
        mrr_score = 0.

        entity_ids = torch.arange(n_entity, device=self.device).unsqueeze(0)
        for heads, relations, tails in self.data_loader:
            current_batch_size = len(heads)
            heads, relations, tails = heads.to(self.device), relations.to(self.device), tails.to(self.device)
            if self.filter_flag:
                tails_predictions_mask = torch.zeros(size=(current_batch_size, n_entity)).bool()
                heads_predictions_mask = torch.zeros(size=(current_batch_size, n_entity)).bool()
                hr2t = self.data_loader.dataset.kg_mapping.hr2t
                rt2h = self.data_loader.dataset.kg_mapping.rt2h
                for i in range(current_batch_size):
                    h, r, t = heads[i].item(), relations[i].item(), tails[i].item()
                    mask_cols = list(hr2t[(h, r)])
                    tails_predictions_mask[i, mask_cols] = True
                    tails_predictions_mask[i, tails[i]] = False
                    mask_cols = list(rt2h[(r, t)])
                    heads_predictions_mask[i, mask_cols] = True
                    heads_predictions_mask[i, heads[i]] = False

            all_entities = entity_ids.repeat(current_batch_size, 1)
            repeat_heads = heads.reshape(-1, 1).repeat(1, n_entity)
            repeat_relations = relations.reshape(-1, 1).repeat(1, n_entity)
            repeat_tails = tails.reshape(-1, 1).repeat(1, n_entity)

            triplets = torch.stack((repeat_heads, repeat_relations, all_entities), dim=2).reshape(-1, 3)
            tails_predictions = self.model(triplets).reshape(current_batch_size, -1)
            if self.filter_flag:
                tails_predictions[tails_predictions_mask] = torch.inf
            triplets = torch.stack((all_entities, repeat_relations, repeat_tails), dim=2).reshape(-1, 3)
            heads_predictions = self.model(triplets).reshape(current_batch_size, -1)
            if self.filter_flag:
                heads_predictions[heads_predictions_mask] = torch.inf

            predictions = torch.cat((tails_predictions, heads_predictions), dim=0)
            ground_truth_entity_id = torch.cat((tails.reshape(-1, 1), heads.reshape(-1, 1)))

            rank = (predictions.argsort() == ground_truth_entity_id).nonzero()[:, 1].float().add(1.0)
            hits_at_1 += (rank <= 1).sum().item()
            hits_at_3 += (rank <= 3).sum().item()
            hits_at_10 += (rank <= 10).sum().item()
            mr_score += rank.sum().item()
            mrr_score += (1. / rank).sum().item()

        example_count = 2 * self.data_loader.dataset.n_triplet
        hits_at_1 = (hits_at_1 / example_count) * 100
        hits_at_3 = (hits_at_3 / example_count) * 100
        hits_at_10 = (hits_at_10 / example_count) * 100
        mr_score = (mr_score / example_count)
        mrr_score = (mrr_score / example_count)
        return hits_at_1, hits_at_3, hits_at_10, mr_score, mrr_score
