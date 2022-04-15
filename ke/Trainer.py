from tqdm import trange
import sys
import torch
from ke.model import BaseModel
from torch.utils.data import DataLoader


class Trainer(object):
    def __init__(self, model: BaseModel, train_dataloader: DataLoader, optimizer, device: torch.device,
                 epochs: int, validation_frequency, validator, checkpoint_path,
                 target_metric="h10", target_score=None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.validation_frequency = validation_frequency
        self.validator = validator
        self.checkpoint_path = checkpoint_path
        self.target_metric = target_metric
        self.target_score = target_score

    def run(self):
        p_bar = trange(self.epochs, desc='Train Epochs', mininterval=1, unit='epochs', file=sys.stdout,
                       colour="blue")
        postfix = {"loss_sum": '?', "loss_mean": '?'}
        epoch = 0
        n_entity = self.train_dataloader.dataset.n_entity

        best_metric_score = 0.
        metric_score_dict = {}
        model_limit = 0.

        for epoch in p_bar:
            self.model.train()
            loss_sum = torch.tensor(0.)
            loss_sum = loss_sum.to(self.device)

            for data in self.train_dataloader:
                current_batch_size = len(data[0])
                if self.train_dataloader.dataset.filter_flag:
                    heads, relations, tails, head_or_tail, negative_entities = data
                else:
                    heads, relations, tails = data
                    head_or_tail = torch.randint(high=2, size=(current_batch_size,))
                    negative_entities = torch.randint(high=n_entity, size=(current_batch_size,))

                heads, relations, tails = heads.to(self.device), relations.to(self.device), tails.to(self.device)
                head_or_tail, negative_entities = head_or_tail.to(self.device), negative_entities.to(self.device)

                positive_triplets = torch.stack((heads, relations, tails), dim=1)

                negative_heads = torch.where(head_or_tail == 0, negative_entities, heads)
                negative_tails = torch.where(head_or_tail == 1, negative_entities, tails)
                negative_triplets = torch.stack((negative_heads, relations, negative_tails), dim=1)

                self.optimizer.zero_grad()
                positive_distance = self.model(positive_triplets)
                negative_distance = self.model(negative_triplets)
                loss = self.model.loss_func(positive_distance, negative_distance)
                loss.mean().backward()
                self.optimizer.step()
                loss_sum += loss.sum()
                # finish one batch
                pass
            postfix["loss_sum"] = loss_sum.item()
            postfix["loss_mean"] = postfix["loss_sum"] / self.train_dataloader.dataset.n_triplet
            p_bar.set_postfix(postfix)
            # finish one epoch
            if epoch % self.validation_frequency == 0:
                hits_at_1, hits_at_3, hits_at_10, mr, mrr = self.validator.link_prediction()
                metric_score_dict["h1"], metric_score_dict["h3"], metric_score_dict["h10"], \
                metric_score_dict["mr"], metric_score_dict["mrr"] = hits_at_1, hits_at_3, hits_at_10, mr, mrr
                metric_score = metric_score_dict[self.target_metric]

                model_limit = (metric_score - best_metric_score) * ((self.epochs - epoch) / self.validation_frequency)
                model_limit += metric_score
                p_bar.write(f"validate --- "
                            f"h1: {hits_at_1:<6.3f}% | "
                            f"h3: {hits_at_3:<6.3f}% | "
                            f"h10: {hits_at_10:<6.3f}% | "
                            f"mr: {mr:<6.3f} | "
                            f"mrr: {mrr:<6.3f} | "
                            f"epoch: {epoch:<6} | "
                            f"model_limit: {model_limit:<6.3f}")
                if metric_score > best_metric_score:
                    self.model.save_checkpoint(self.checkpoint_path)
                    best_metric_score = metric_score

                # early return
                if self.target_score:
                    if self.target_metric == "h1" or self.target_metric == "h3" \
                           or self.target_metric == "h10" or self.target_metric == "mrr":
                        if metric_score > self.target_score:
                            print(f"metric_score is greater than target_min_score:{self.target_score}")
                            return epoch, best_metric_score, model_limit
                        if epoch != 0 and model_limit < self.target_score:
                            print(f"model_limit is lower than target_min_score:{self.target_score}")
                            return epoch, best_metric_score, model_limit
                    if self.target_metric == "mr":
                        if metric_score < self.target_score:
                            print(f"metric_score is lower than target_max_score:{self.target_score}")
                            return epoch, best_metric_score, model_limit
                        if epoch != 0 and model_limit > self.target_score:
                            print(f"model_limit is greater than target_max_score:{self.target_score}")
                            return epoch, best_metric_score, model_limit

            self.train_dataloader.dataset.regenerate_head_or_tail()
        # finish training
        p_bar.close()
        return epoch, best_metric_score, model_limit