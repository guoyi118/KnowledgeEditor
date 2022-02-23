python scripts/train_T5_augmented_qdmr.py \
    --gpus 4 \
    --accelerator ddp \
    --num_workers 32 \
    --batch_size 32 \
    --max_steps 200000 \
    --divergences cr \
    --precision 32 \
    --lr 5e-3 \
    --max_length 128\
