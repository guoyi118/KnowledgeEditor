#/bin/bash
python scripts/train_bart_seq2seq_augmented_kilt.py \
    --gpus 1 \
    --accelerator ddp \
    --num_workers 32 \
    --batch_size 64 \
    --max_steps 200000 \
    --divergences both \
