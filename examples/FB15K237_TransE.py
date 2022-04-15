import os
import sys

from ke import fix_random, Tester, Trainer
from ke.data import KGMapping, KGDataset
from ke.model import TransE

import torch
from torch.utils.data import DataLoader
from torch import optim

sys.path.append(os.path.dirname(sys.path[0]))
NORM = 1
MARGIN = 5.0
VECTOR_LENGTH = 200
LEARNING_RATE = 1
TARGET_METRIC = 'h10'  # hit@10
TARGET_SCORE = None  # 70%
EPOCHS = 5000
TRAIN_BATCH_SIZE = 4096
VALID_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
VALIDATION_FREQUENCY = 100
FILTER_FLAG = True
USE_GPU = True
DATASET_PATH = os.path.join("..", "benchmarks", "FB15K-237.2")
if not os.path.isdir("ckpt"):
    os.mkdir("ckpt")
CHECKPOINT_PATH = os.path.join("ckpt", "FB15K-237.2_TransE.checkpoint.tar")
SEED = 1234

if __name__ == "__main__":
    print(f"MARGIN:{MARGIN}, NORM:{NORM}, VECTOR_LENGTH:{VECTOR_LENGTH}, LEARNING_RATE:{LEARNING_RATE}\n"
          f"EPOCHS:{EPOCHS}, VALIDATE_FREQUENCY:{VALIDATION_FREQUENCY}, FILTER_FLAG:{FILTER_FLAG}\n"
          f"USE_GPU:{USE_GPU}, SEED:{SEED}, DATASET_PATH:{DATASET_PATH}, CHECKPOINT_PATH:{CHECKPOINT_PATH}\n"
          f"TRAIN_BATCH_SIZE:{TRAIN_BATCH_SIZE}, VALID_BATCH_SIZE:{VALID_BATCH_SIZE}, "
          f"TEST_BATCH_SIZE:{TEST_BATCH_SIZE}\n"
          f"TARGET_METRIC:{TARGET_METRIC}, TARGET_SCORE:{TARGET_SCORE}\n")

    fix_random(SEED)
    FB15K_path = DATASET_PATH
    train_path = os.path.join(FB15K_path, "train.txt")
    valid_path = os.path.join(FB15K_path, "valid.txt")
    test_path = os.path.join(FB15K_path, "test.txt")
    checkpoint_path = CHECKPOINT_PATH

    print("preparing knowledge graph data...", end='')
    fb15k_mapping = KGMapping(FB15K_path)
    n_entity = fb15k_mapping.n_entity
    n_relation = fb15k_mapping.n_relation

    fb15k_train_dataset = KGDataset(train_path, fb15k_mapping, filter_flag=FILTER_FLAG)
    fb15k_train_dataloader = DataLoader(fb15k_train_dataset, TRAIN_BATCH_SIZE)
    fb15k_valid_dataset = KGDataset(valid_path, fb15k_mapping)
    fb15k_valid_dataloader = DataLoader(fb15k_valid_dataset, VALID_BATCH_SIZE)
    fb15k_test_dataset = KGDataset(test_path, fb15k_mapping)
    fb15k_test_dataloader = DataLoader(fb15k_test_dataset, TEST_BATCH_SIZE)
    print("done")
    print(f"entity_count:{n_entity}, n_relation_count:{n_relation}, "
          f"train_triplets_count:{fb15k_train_dataset.n_triplet}, "
          f"valid_triplets_count:{fb15k_valid_dataset.n_triplet}, "
          f"test_triplets_count:{fb15k_test_dataset.n_triplet}\n")

    device = torch.device('cuda') if USE_GPU else torch.device('cpu')

    print("preparing model...", end='')
    transe = TransE(n_entity, n_relation, VECTOR_LENGTH, p_norm=NORM, margin=MARGIN)
    transe = transe.to(device)
    optimizer = optim.SGD(transe.parameters(), lr=LEARNING_RATE)
    validator = Tester(model=transe, data_loader=fb15k_valid_dataloader, device=device,
                       filter_flag=FILTER_FLAG)
    tester = Tester(model=transe, data_loader=fb15k_test_dataloader, device=device,
                    filter_flag=FILTER_FLAG)
    trainer = Trainer(model=transe, train_dataloader=fb15k_train_dataloader, optimizer=optimizer,
                      device=device, epochs=EPOCHS, validation_frequency=VALIDATION_FREQUENCY,
                      validator=validator, checkpoint_path=checkpoint_path,
                      target_metric=TARGET_METRIC, target_score=TARGET_SCORE)

    print("done")
    print(transe)

    print("-" * 20, "start training epochs", "-" * 20)
    exit_epoch, best_metric_score = trainer.run()
    print(f"exit epoch: {exit_epoch}, best {TARGET_METRIC} on validation: {best_metric_score}")
    transe.load_checkpoint(checkpoint_path)
    hits_at_1, hits_at_3, hits_at_10, mr, mrr = tester.link_prediction()
    print(f"test --- "
          f"h1: {hits_at_1:<6.3f}% | "
          f"h3: {hits_at_3:<6.3f}% | "
          f"h10: {hits_at_10:<6.3f}% | "
          f"mr: {mr:<6.3f} | "
          f"mrr: {mrr:<6.3f} | ")
