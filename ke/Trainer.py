import os.path

from tqdm import trange
import sys
import torch
from ke.model import BaseModel
from torch.utils.data import DataLoader


class Trainer(object):
    def __init__(self, model: BaseModel, train_dataloader: DataLoader, optimizer, device: torch.device,
                 epochs: int, validation_frequency, validator, checkpoint_path,
                 target_metric="h10", target_score=None, log_write_func=None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.validation_frequency = validation_frequency
        self.validator = validator
        self.checkpoint_path = checkpoint_path
        self.target_metric = target_metric.lower()
        self.target_score = target_score
        self.log_write_func = log_write_func

    def run(self, rerun_from=None):
        if rerun_from and os.path.exists(rerun_from):
            self.model.load_checkpoint(rerun_from)
        else:
            self.model.save_checkpoint(self.checkpoint_path)

        p_bar = trange(self.epochs, desc='Train Epochs', mininterval=1, unit='epochs', file=sys.stdout,
                       colour="blue", ncols=100)
        message = f"epoch  |   h@1   |   h@3   |   h@10   |   mrr  |    mr    | " \
                  f"predict limit {self.target_metric} on validation set"
        p_bar.write(message)
        if self.log_write_func is not None:
            self.log_write_func(message)

        postfix = {"loss_sum": '?', "loss_mean": '?'}
        epoch = 0
        n_entity = self.train_dataloader.dataset.n_entity

        best_metric_score = 0.
        metric_score_dict = {}

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
                # one batch finished
                pass
            postfix["loss_sum"] = loss_sum.item()
            postfix["loss_mean"] = postfix["loss_sum"] / self.train_dataloader.dataset.n_triplet
            p_bar.set_postfix(postfix)
            # one epoch finished
            if epoch % self.validation_frequency == 0 or epoch == self.epochs - 1:
                hits_at_1, hits_at_3, hits_at_10, mr, mrr = self.validator.link_prediction()
                metric_score_dict["h1"], metric_score_dict["h3"], metric_score_dict["h10"], \
                    metric_score_dict["mr"], metric_score_dict["mrr"] = hits_at_1, hits_at_3, hits_at_10, mr, mrr
                metric_score = metric_score_dict[self.target_metric]

                predict_limit = (metric_score - best_metric_score) * ((self.epochs - epoch) / self.validation_frequency)
                predict_limit += metric_score
                message = f"{epoch:<6} | " \
                          f"{hits_at_1:<6.3f}% | " \
                          f"{hits_at_3:<6.3f}% | " \
                          f"{hits_at_10:<6.3f}%  | " \
                          f"{mrr:<6.4f} | " \
                          f"{mr:<8.3f} | " \
                          f"{predict_limit:<6.3f}"
                p_bar.write(message)
                if self.log_write_func is not None:
                    self.log_write_func(message)

                if metric_score > best_metric_score:
                    self.model.save_checkpoint(self.checkpoint_path)
                    best_metric_score = metric_score

                higher_better = {"h1", "h3", "h10", "mrr"}
                lower_better = {"mr"}
                # early return
                if self.target_score:
                    if self.target_metric in higher_better:
                        if metric_score > self.target_score:
                            message = "-" * 20 + f" Reach the goal in the epoch {epoch} " + "-" * 20
                            p_bar.write(message)
                            if self.log_write_func is not None:
                                self.log_write_func(message)
                            return epoch, best_metric_score, postfix["loss_sum"], postfix["loss_mean"]
                        if epoch != 0 and predict_limit < self.target_score:
                            message = "-" * 20 + f" Never Reach the goal:{self.target_score} " + "-" * 20
                            p_bar.write(message)
                            if self.log_write_func is not None:
                                self.log_write_func(message)
                            return epoch, best_metric_score, postfix["loss_sum"], postfix["loss_mean"]
                    if self.target_metric in lower_better:
                        if metric_score < self.target_score:
                            message = "-" * 20 + f" Reach the goal in the epoch {epoch} " + "-" * 20
                            p_bar.write(message)
                            if self.log_write_func is not None:
                                self.log_write_func(message)
                            return epoch, best_metric_score, postfix["loss_sum"], postfix["loss_mean"]
                        if epoch != 0 and predict_limit > self.target_score:
                            message = "-" * 20 + f" Never Reach the goal:{self.target_score} " + "-" * 20
                            p_bar.write(message)
                            if self.log_write_func is not None:
                                self.log_write_func(message)
                            return epoch, best_metric_score, postfix["loss_sum"], postfix["loss_mean"]

            self.train_dataloader.dataset.regenerate_head_or_tail()
        # train finished
        p_bar.close()
        return epoch, best_metric_score, postfix["loss_sum"], postfix["loss_mean"]
