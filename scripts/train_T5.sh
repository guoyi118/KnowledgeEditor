python scripts/train_T5.py \
    --gpus 2 \
    --accelerator ddp \
    --num_workers 32 \
    --batch_size 64 \
    --max_steps 200000 \
    --divergences kl \
