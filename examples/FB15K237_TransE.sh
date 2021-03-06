python TransE.py \
--NORM=1 \
--MARGIN=3 \
--VECTOR_LENGTH=200 \
--LEARNING_RATE=1 \
--EPOCHS=10000 \
--VALIDATE_FREQUENCY=50 \
--FILTER_FLAG=True \
--USE_GPU=True \
--GPU_INDEX=0 \
--DATASET_PATH=./benchmarks/FB15K-237.2 \
--CHECKPOINT_PATH=./ckpt/checkpoint.tar \
--TRAIN_BATCH_SIZE=2048 \
--VALID_BATCH_SIZE=64 \
--TEST_BATCH_SIZE=64 \
--TARGET_METRIC=h10 \
--TARGET_SCORE=None \
--SEED=1234 \
--PROC_TITLE=TransE \
--LOG=True \
--NUM_WORKERS=0
