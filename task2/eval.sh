export CUDA_VISIBLE_DEVICES=3
DATADIR=../../deft_corpus/data/deft_files
OUTDIR=./save-1
MODELDIR=./save-1/checkpoint-2000-9.4999
PER_GPU_BATCH_EVAL=8
BLOCKSIZE=512


# -m torch.distributed.launch --nproc_per_node=$PER_NODE_GPU
python  eval.py \
        --data_dir=$DATADIR \
        --output_dir=$OUTDIR \
        --pretrain_dir=$MODELDIR \
        --block_size=$BLOCKSIZE \
        --eval_batch_size=$PER_GPU_BATCH_EVAL \
        --seed=2233 \
