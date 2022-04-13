#/bin/bash
python -m scripts.train_bart_seq2seq_augmented_kilt \
    --gpus 1 \
    --accelerator ddp \
    --num_workers 32 \
    --batch_size 2 \
    --max_steps 200000 \
    --divergences both \
    --train_data_path src/datasets/structured_zeroshot-train-new_annotated_final.jsonl \
    --use_views \
    2>&1 | tee models/bart_seq2seq_augmented_structured_zeroshot/log.txt
