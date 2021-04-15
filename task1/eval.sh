export CUDA_VISIBLE_DEVICES=0
DATADIR=../../deft_corpus/data/deft_files
MODELDIR=./save/checkpoint-1000-0.8611
PER_GPU_BATCH_EVAL=48
BLOCKSIZE=384


# -m torch.distributed.launch --nproc_per_node=$PER_NODE_GPU
python  eval.py \
        --data_dir=$DATADIR \
        --evaluate_dir=$MODELDIR \
        --block_size=$BLOCKSIZE \
        --eval_batch_size=$PER_GPU_BATCH_EVAL \
        --seed=2233 \
