import os
import datetime

from ke import fix_random, set_proc_title, parse_args, get_logger, Tester, Trainer
from ke.data import KGMapping, KGDataset
from ke.model import TransE

import torch
from torch.utils.data import DataLoader
from torch import optim

args = parse_args()
NORM = args.NORM
MARGIN = args.MARGIN
VECTOR_LENGTH = args.VECTOR_LENGTH
LEARNING_RATE = args.LEARNING_RATE
EPOCHS = args.EPOCHS
VALIDATE_FREQUENCY = args.VALIDATE_FREQUENCY
FILTER_FLAG = args.FILTER_FLAG
USE_GPU = args.USE_GPU
GPU_INDEX = args.GPU_INDEX
DATASET_PATH = args.DATASET_PATH
CHECKPOINT_PATH = args.CHECKPOINT_PATH
TRAIN_BATCH_SIZE = args.TRAIN_BATCH_SIZE
VALID_BATCH_SIZE = args.VALID_BATCH_SIZE
TEST_BATCH_SIZE = args.TEST_BATCH_SIZE
TARGET_METRIC = args.TARGET_METRIC
TARGET_SCORE = args.TARGET_SCORE
SEED = args.SEED
PROC_TITLE = args.PROC_TITLE
LOG = args.LOG
NUM_WORKERS = args.NUM_WORKERS

if __name__ == "__main__":
    now = datetime.datetime.now()
    date = f"{now.year}-{now.month}-{now.day}"
    logger = get_logger(date+".txt") if LOG else None
    message = f"\nMARGIN:{MARGIN}, NORM:{NORM}, VECTOR_LENGTH:{VECTOR_LENGTH}, LEARNING_RATE:{LEARNING_RATE}\n" \
              f"EPOCHS:{EPOCHS}, VALIDATE_FREQUENCY:{VALIDATE_FREQUENCY}, FILTER_FLAG:{FILTER_FLAG}\n" \
              f"USE_GPU:{USE_GPU}, GPU_INDEX:{GPU_INDEX}, SEED:{SEED}, PROC_TITLE:{PROC_TITLE}\n" \
              f"DATASET_PATH:{DATASET_PATH}, CHECKPOINT_PATH:{CHECKPOINT_PATH}\n" \
              f"TRAIN_BATCH_SIZE:{TRAIN_BATCH_SIZE}, VALID_BATCH_SIZE:{VALID_BATCH_SIZE}, " \
              f"TEST_BATCH_SIZE:{TEST_BATCH_SIZE}\n" \
              f"TARGET_METRIC:{TARGET_METRIC}, TARGET_SCORE:{TARGET_SCORE}\n" \
              f"LOG:{LOG}, NUM_WORKERS:{NUM_WORKERS}\n"
    print(message)
    if LOG:
        logger.info("-"*20+f" {date} "+"-"*20)
        logger.info(message)

    if not os.path.isdir("ckpt"):
        os.mkdir("ckpt")
    fix_random(SEED)
    set_proc_title(PROC_TITLE)
    FB15K_path = DATASET_PATH
    train_path = os.path.join(FB15K_path, "train.txt")
    valid_path = os.path.join(FB15K_path, "valid.txt")
    test_path = os.path.join(FB15K_path, "test.txt")
    checkpoint_path = CHECKPOINT_PATH

    print("preparing knowledge graph data...", end='')
    if LOG:
        logger.info("preparing knowledge graph data...")

    fb15k_mapping = KGMapping(FB15K_path)
    n_entity = fb15k_mapping.n_entity
    n_relation = fb15k_mapping.n_relation

    fb15k_train_dataset = KGDataset(train_path, fb15k_mapping, filter_flag=FILTER_FLAG)
    fb15k_train_dataloader = DataLoader(fb15k_train_dataset, TRAIN_BATCH_SIZE, num_workers=NUM_WORKERS)
    fb15k_valid_dataset = KGDataset(valid_path, fb15k_mapping)
    fb15k_valid_dataloader = DataLoader(fb15k_valid_dataset, VALID_BATCH_SIZE)
    fb15k_test_dataset = KGDataset(test_path, fb15k_mapping)
    fb15k_test_dataloader = DataLoader(fb15k_test_dataset, TEST_BATCH_SIZE)

    print("done")
    if LOG:
        logger.info("done")

    message = f"entity_count:{n_entity}, n_relation_count:{n_relation}\n" \
              f"train_triplets_count:{fb15k_train_dataset.n_triplet}, " \
              f"valid_triplets_count:{fb15k_valid_dataset.n_triplet}, " \
              f"test_triplets_count:{fb15k_test_dataset.n_triplet}\n"
    print(message)
    if LOG:
        logger.info(message)

    if USE_GPU:
        assert GPU_INDEX < torch.cuda.device_count()
    device = torch.device('cuda:' + str(GPU_INDEX)) if USE_GPU else torch.device('cpu')

    print("preparing model...", end='')
    if LOG:
        logger.info("preparing model...")

    transe = TransE(n_entity, n_relation, VECTOR_LENGTH, p_norm=NORM, margin=MARGIN)
    transe = transe.to(device)
    optimizer = optim.SGD(transe.parameters(), lr=LEARNING_RATE)
    # optimizer = optim.Adam(transe.parameters(), lr=LEARNING_RATE)
    validator = Tester(model=transe, data_loader=fb15k_valid_dataloader, device=device,
                       filter_flag=FILTER_FLAG)
    tester = Tester(model=transe, data_loader=fb15k_test_dataloader, device=device,
                    filter_flag=FILTER_FLAG)
    trainer = Trainer(model=transe, train_dataloader=fb15k_train_dataloader, optimizer=optimizer,
                      device=device, epochs=EPOCHS, validation_frequency=VALIDATE_FREQUENCY,
                      validator=validator, checkpoint_path=checkpoint_path,
                      target_metric=TARGET_METRIC, target_score=TARGET_SCORE,
                      log_write_func=logger.info if LOG else None)

    print("done")
    if LOG:
        logger.info("done")
    print(transe)
    if LOG:
        logger.info(transe)

    print("-" * 20, "start training epochs", "-" * 20)
    if LOG:
        logger.info("-" * 20 + " start training epochs " + "-" * 20)

    exit_epoch, best_metric_score, loss_sum, loss_mean = trainer.run()
    message = f"exit epoch: {exit_epoch}, loss_sum: {loss_sum}, loss_min:{loss_mean} \n" \
              f"best {TARGET_METRIC} on validation: {best_metric_score}"
    print(message)
    if LOG:
        logger.info(message)

    transe.load_checkpoint(checkpoint_path)
    hits_at_1, hits_at_3, hits_at_10, mr, mrr = tester.link_prediction()

    message = f"test --- " \
              f"h1: {hits_at_1:<6.3f}% | " \
              f"h3: {hits_at_3:<6.3f}% | " \
              f"h10: {hits_at_10:<6.3f}% | " \
              f"mr: {mr:<6.3f} | " \
              f"mrr: {mrr:<6.3f} | "
    print(message)
    if LOG:
        logger.info(message)
        logger.info("-" * 50)
