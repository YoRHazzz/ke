python TransE.py \
--NORM=1 \
--MARGIN=1 \
--VECTOR_LENGTH=50 \
--LEARNING_RATE=0.01 \
--EPOCHS=1000 \
--VALIDATE_FREQUENCY=50 \
--FILTER_FLAG=True \
--USE_GPU=True \
--GPU_INDEX=0 \
--DATASET_PATH=./benchmarks/FB15K \
--CHECKPOINT_PATH=./ckpt/checkpoint.tar \
--TRAIN_BATCH_SIZE=50 \
--VALID_BATCH_SIZE=64 \
--TEST_BATCH_SIZE=64 \
--TARGET_METRIC=h10 \
--TARGET_SCORE=None \
--SEED=1234 \
--PROC_TITLE=Same_Hyperparameter \
--LOG=True \
--NUM_WORKERS=0
