import os

from ke import fix_random, Tester, Trainer
from ke.data import KGMapping, KGDataset
from ke.model import TransE

import torch
from torch.utils.data import DataLoader
from torch import optim

NORM_LIST = [1, 2]
MARGIN_LIST = [1.0, 1.5, 2.0, 3.0]
VECTOR_LENGTH_LIST = [50, 100, 150, 200]
LEARNING_RATE_LIST = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
TARGET_SCORE = 50.
EPOCHS = 5000
TRAIN_BATCH_SIZE = 1024
VALID_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
VALIDATION_FREQUENCY = 100
FILTER_FLAG = True
USE_GPU = True
DATASET_PATH = os.path.join("..", "benchmarks", "FB15K")
CHECKPOINT_PATH = os.path.join("ckpt", "FB15K_TransE.checkpoint.tar")
SEED = 1234

if __name__ == "__main__":
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
    fb15k_valid_dataset = KGDataset(valid_path, fb15k_mapping, filter_flag=False)
    fb15k_valid_dataloader = DataLoader(fb15k_valid_dataset, VALID_BATCH_SIZE)
    fb15k_test_dataset = KGDataset(test_path, fb15k_mapping, filter_flag=False)
    fb15k_test_dataloader = DataLoader(fb15k_test_dataset, TEST_BATCH_SIZE)
    print("done")
    print(f"entity_count:{n_entity}, n_relation_count:{n_relation}, "
          f"train_triplets_count:{fb15k_train_dataset.n_triplet}, "
          f"valid_triplets_count:{fb15k_valid_dataset.n_triplet}, "
          f"test_triplets_count:{fb15k_test_dataset.n_triplet}")

    device = torch.device('cuda') if USE_GPU else torch.device('cpu')

    result = {}
    for MARGIN in MARGIN_LIST:
        for NORM in NORM_LIST:
            for VECTOR_LENGTH in VECTOR_LENGTH_LIST:
                for LEARNING_RATE in LEARNING_RATE_LIST:
                    print(f"MARGIN:{MARGIN}, NORM:{NORM}, VECTOR_LENGTH:{VECTOR_LENGTH}, LEARNING_RATE:{LEARNING_RATE}")
                    print("preparing model...")
                    transe = TransE(n_entity, n_relation, VECTOR_LENGTH, p_norm=NORM, margin=MARGIN)
                    transe = transe.to(device)
                    optimizer = optim.SGD(transe.parameters(), lr=LEARNING_RATE)
                    validator = Tester(model=transe, data_loader=fb15k_valid_dataloader, device=device,
                                       filter_flag=FILTER_FLAG)
                    tester = Tester(model=transe, data_loader=fb15k_test_dataloader, device=device,
                                    filter_flag=FILTER_FLAG)
                    trainer = Trainer(model=transe, train_dataloader=fb15k_train_dataloader, optimizer=optimizer,
                                      device=device, epochs=EPOCHS, validation_frequency=VALIDATION_FREQUENCY,
                                      validator=validator,
                                      checkpoint_path=checkpoint_path, target_score=TARGET_SCORE)

                    print(transe)
                    print("-" * 20, "start training epochs", "-" * 20)
                    epoch, best_hits_at_10, model_limit = trainer.run()
                    transe.load_checkpoint(checkpoint_path)
                    hits_at_1, hits_at_3, hits_at_10, mr, mrr = tester.link_prediction()
                    print(f"test --- "
                          f"h1: {hits_at_1:<6.3f}% | "
                          f"h3: {hits_at_3:<6.3f}% | "
                          f"h10: {hits_at_10:<6.3f}% | "
                          f"mr: {mr:<6.3f} | "
                          f"mrr: {mrr:<6.3f} | ")
                    result[(MARGIN, NORM, VECTOR_LENGTH, LEARNING_RATE)] = (epoch, best_hits_at_10, model_limit)

    for MARGIN, NORM, VECTOR_LENGTH, LEARNING_RATE in result:
        epoch, best_hits_at_10, model_limit = result[(MARGIN, NORM, VECTOR_LENGTH, LEARNING_RATE)]
        print(f"MARGIN = {MARGIN} "
              f"NORM = {NORM} "
              f"VECTOR_LENGTH = {VECTOR_LENGTH} "
              f"LEARNING_RATE = {LEARNING_RATE} "
              f"epoch = {epoch} "
              f"best_hits_at_10 = {best_hits_at_10} "
              f"model_limit = {model_limit}")
