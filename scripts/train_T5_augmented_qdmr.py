import os
from argparse import ArgumentParser
from pprint import pprint
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
import sys
sys.path.append('/root/KnowledgeEditor')

from src.models.T5_seq2seq_augmented_qdmr import T5Seq2SeqAugmented

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--dirpath",
        type=str,
        default="models/T5_seq2seq_augmented_qdmr",
    )
    parser.add_argument("--save_top_k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser = T5Seq2SeqAugmented.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args, _ = parser.parse_known_args()
    pprint(args.__dict__)

    seed_everything(seed=args.seed)

    logger = TensorBoardLogger(args.dirpath, name=None)

    callbacks = [
        ModelCheckpoint(
            monitor="valid_acc",
            mode="max",
            dirpath=os.path.join(logger.log_dir, "checkpoints"),
            save_top_k=args.save_top_k,
            filename="model-{epoch:02d}-{valid_acc:.4f}-{valid_flipped:4f}",
        ),
        LearningRateMonitor(
            logging_interval="step",
        ),
    ]

    trainer = Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks, gradient_clip_val=0.5,plugins=DDPPlugin(find_unused_parameters=True))

    model = T5Seq2SeqAugmented(**vars(args))

    trainer.fit(model)

