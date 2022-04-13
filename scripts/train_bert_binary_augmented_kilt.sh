#/bin/bash
python -m scripts.train_bert_binary_augmented_kilt \
    --gpus 1 \
    --accelerator ddp \
    --num_workers 32 \
    --batch_size 2 \
    --max_steps 50000 \
    --divergences both \
    --use_views \
    2>&1 | tee models/bert_binary_augmented_fever/log.txt
