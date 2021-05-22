export CUDA_VISIBLE_DEVICES=3
DATADIR=../../deft_corpus/data/deft_files
OUTDIR=./save-aug1
MODELDIR=./save-aug1/checkpoint-500-0.4666
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
