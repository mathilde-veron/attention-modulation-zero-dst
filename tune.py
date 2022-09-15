import argparse
from pathlib import Path

import pytorch_lightning as pl

from dataset import SlotEncoding, MultiWOZ, Ontology
from model import SumbtLL, BERTTokenizer
from utils import Config


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True, help="Experiment directory where config.yml is located")
parser.add_argument('--gpus', type=int, required=False, default=2, help="Number of gpus to train on")
args = parser.parse_args()
args.dir = Path(args.dir)

config = Config(args.dir / 'config.yml')

# ------------------ TRAINING ------------------
print("Loading ontology... ", end="", flush=True)
ontology_name = f'ontology_{config.train.ontology_subset}.json'
slot_encoding = SlotEncoding(config, ontology_name)
ontology = Ontology(ontology_name, config, BERTTokenizer, slot_encoding)
print("OK")

print("Loading model... ", end="", flush=True)
model = SumbtLL(config, slot_encoding, ontology)
print("OK")

print("Loading training data... ", end="", flush=True)
dataset = MultiWOZ(config, slot_encoding, BERTTokenizer)
dataset_training = dataset[config.train.data_subset]
train_loader = dataset_training.get_loader(
    split='train',
    batch_size=config.train.batch_size,
    num_workers=config.train.num_workers,
    shuffle=True
)
val_loader = dataset_training.get_loader(
    split='dev',
    batch_size=config.train.batch_size,
    num_workers=config.train.num_workers,
    shuffle=True
)
print("OK")

save_dir = args.dir / 'tuning_logs'
save_dir.mkdir(parents=True, exist_ok=True)

trainer = pl.Trainer(default_root_dir=save_dir)

# Run learning rate finder
lr_finder = trainer.tuner.lr_find(model, train_loader, val_loader)

# Plot lr vs. losss and save it
fig = lr_finder.plot(suggest=True)
fig.savefig(save_dir / 'lr_tuning_results.png')

# Pick point based on plot to get the suggested lr
new_lr = lr_finder.suggestion()
print('Suggested learning rate:', new_lr)
