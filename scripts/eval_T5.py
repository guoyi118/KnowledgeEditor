import argparse
import logging
import os
import pickle
from copy import deepcopy

import torch
from tqdm.auto import tqdm
import sys
sys.path.append('/root/KnowledgeEditor')

from src.data.seq2seq_kilt import Seq2SeqKILT
from src.models.bart_seq2seq_augmented_kilt import BartSeq2SeqAugmented
from src.models.bart_seq2seq_kilt import BartSeq2Seq
from src.models.T5_seq2seq_kilt import T5Seq2Seq

from src.utils import batch_it, shuffle_it, normalize, label_smoothed_nll_loss

checkpoint = '/root/KnowledgeEditor/models/bart_seq2seq_structured_zeroshot/T5-new_loss_version_12/checkpoints/model-epoch=04-valid_acc=0.9990.ckpt'
device = 'cuda'
datapath = '/root/sparqling-queries/data/break/logical-forms-fixed/dev_data_df.jsonl'

model = T5Seq2Seq.load_from_checkpoint(checkpoint).to(device)

val_dataset0 = Seq2SeqKILT(
    tokenizer=model.tokenizer,
    data_path=datapath,
    max_length=model.hparams.max_length,
)

trg = [e["trg"] for e in val_dataset0]

all_guess = {}
all_rephrases = {}
all_alts = {}

iter_ = tqdm(val_dataset0)

for j, d0 in iter_:
    # all_alts[j] = d0[0]["alt"]
    print(d0)
    tmodel = deepcopy(model)

    batch = {
        k: v.to(tmodel.device)
        for k, v in val_dataset0.collate_fn([d0]).items()
        if isinstance(v, torch.Tensor)
    }

    print(batch)


#         logits = tmodel(batch)
#         _, loss = label_smoothed_nll_loss(
#             logits.log_softmax(-1),
#             batch["trg_input_ids"][:, 1:],
#             epsilon=0,
#             ignore_index=model.tokenizer.pad_token_id,
#         )
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

#     all_rephrases[j] = tmodel.sample(d0["view"])

#     all_guess_batch = []
#     for i, d1 in enumerate(batch_it(tqdm(val_dataset1), args.batch_size)):
#         all_guess_batch += tmodel.sample(
#             [e["src"] for e in d1], num_return_sequences=5
#         )

#     all_guess[j] = all_guess_batch

#     iter_.set_postfix(
#         succ=sum(
#             normalize(all_alts[k]) == normalize(v[k][0])
#             for k, v in all_guess.items()
#         )
#         / len(all_guess),
#         retain=sum(
#             (
#                 sum(a == b for a, b in zip(preds[:k], [e[0] for e in v[:k]]))
#                 + sum(
#                     a == b
#                     for a, b in zip(preds[k + 1 :], [e[0] for e in v[k + 1 :]])
#                 )
#             )
#             / (len(v) - 1)
#             for k, v in all_guess.items()
#         )
#         / len(all_guess),
#         equiv=sum(
#             sum(e[0] == all_guess[k][k][0] for e in v) / len(v)
#             for k, v in all_rephrases.items()
#         )
#         / len(all_rephrases),
#     )

# filename = os.path.join(
#     args.output_path,
#     f"all_guess-{args.from_idx}-{args.to_idx}-baseline-{args.layer}.pkl",
# )
# logging.info("Saving {}".format(filename))
# with open(filename, "wb") as f:
#     pickle.dump(all_guess, f)

# filename = os.path.join(
#     args.output_path,
#     f"all_rephrases-{args.from_idx}-{args.to_idx}-baseline-{args.layer}.pkl",
# )
# logging.info("Saving {}".format(filename))
# with open(filename, "wb") as f:
#     pickle.dump(all_rephrases, f)

