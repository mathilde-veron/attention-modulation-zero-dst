import argparse
from ast import literal_eval
from pathlib import Path
from re import findall

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd

from dataset import SlotEncoding, MultiWOZ, Ontology
from model import SumbtLL, BERTTokenizer
from utils import Config


def save_parameters(my_model: SumbtLL, save_dir: Path):
    with (save_dir / 'trainable_params.txt').open(mode='w') as f:
        print('--- TRAINABLE PARAMS ---', file=f)
        print('Name: Param', file=f)
        total_params = 0
        for name, parameter in my_model.named_parameters():
            if not parameter.requires_grad:
                continue
            param = parameter.numel()
            print(f'{name}: {param}', file=f)
            total_params += param
        print(f"Total Trainable Params: {total_params}", file=f)
    return total_params


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True, help="Experiment directory where config.yml is located")
parser.add_argument('--gpus', type=int, required=False, default=2, help="Number of gpus to train on")
parser.add_argument("--resume", type=literal_eval, required=False, help="Whether to resume training from a checkpoint")
args = parser.parse_args()
args.dir = Path(args.dir)

config = Config(args.dir / 'config.yml')

# ------------------ TRAINING ------------------
print(f"Loading ontology {config.train.ontology_subset}... ", end="", flush=True)
ontology_name = f'ontology_{config.train.ontology_subset}.json'
slot_encoding = SlotEncoding(config, ontology_name)
ontology = Ontology(ontology_name, config, BERTTokenizer, slot_encoding)
print("OK")

# debug
debug = False
if debug:
    print('--- Slot encoding ---')
    print('int2type:')
    print(slot_encoding.int2type)
    print('int2value dict lengths:')
    print([len(v) for _, v in slot_encoding.int2value.items()])
    print('---------------------')

print("Loading model... ", end="", flush=True)
model = SumbtLL(config, slot_encoding, ontology)
save_parameters(model, args.dir)
print("OK")

print(f"Loading {config.train.data_subset} training data... ", end="", flush=True)
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
    shuffle=False
)
print("OK")

# Only save the best model according to the validation metric
validation_metric = 'joint_acc'
checkpoint_callback = ModelCheckpoint(
    monitor=f"dev_{validation_metric}",
    mode="min" if validation_metric == "loss" else "max",
    save_top_k=1
)

early_stop_callback = EarlyStopping(monitor="dev_loss", min_delta=0.000001, patience=15, verbose=False, mode="min")

# resume training
if args.resume:
    ckpt_dir = Path(args.dir / 'logs' / 'checkpoints')
    ckpt_paths = [p for p in ckpt_dir.iterdir()]
    last_epoch = 0
    for p in ckpt_paths:
        ckpt_epoch = int(findall(r'epoch=([0-9]+)-step=[0-9]+.ckpt', p.name)[0])
        if ckpt_epoch > last_epoch:
            last_epoch = ckpt_epoch
            ckpt_path = p

# The checkpoints and logging files are automatically saved in save_dir
logger = TensorBoardLogger(save_dir=args.dir, name=None, version='logs')
lr_monitor = LearningRateMonitor(logging_interval='step')
trainer = pl.Trainer(
    gpus=args.gpus,
    max_epochs=config.train.epochs,
    num_sanity_val_steps=0,
    logger=logger,
    checkpoint_callback=True,
    callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
    resume_from_checkpoint=None if not args.resume else ckpt_path
)

slot_encoding.save(save_dir=args.dir)

trainer.fit(model, train_loader, val_loader)

# ------------------ EVALUATION ------------------
# Load the best checkpoint
print("Loading best checkpoint...   ", end="", flush=True)
best_ckpt = trainer.checkpoint_callback.best_model_path
model = SumbtLL.load_from_checkpoint(
    checkpoint_path=best_ckpt,
    map_location=model.device,
    config=model.cfg,
    slot_encoding=model.slot_encoding,
    ontology=model.ontology
)
print("OK")

# Evaluate the best checkpoint on each test split
for subset in config.eval.subsets:
    print('\n')
    print(f"Evaluate the trained model on {subset} data")

    # Update the slot encoding and extend the ontology the model has access to if needed
    if subset != config.train.ontology_subset:
        print(f"Extend current ontology with {subset} ontology...   ", end="", flush=True)
        slot_encoding.update_from_ontology(f'ontology_{subset}.json')
        model.update_ontology(f'ontology_{subset}.json')
        print('OK')

    # Load evaluation data
    print("Loading evaluation data...   ", end="", flush=True)
    dataset_eval = dataset_training if subset == config.train.data_subset else dataset[subset]
    eval_loader = dataset_eval.get_loader(
        split='test',
        batch_size=config.eval.batch_size,
        num_workers=config.train.num_workers,
        shuffle=False
    )
    print("OK")

    # Get evaluation results
    results = trainer.test(model, eval_loader, verbose=False)[0]
    print('Evaluation results:', results['performance'])

    # Save evaluation results in a csv
    filename = f"all_accuracies_{subset}_test.csv"
    suffix = subset
    if config.eval.attention_modulation:
        attn_modulation_model = config.eval.attention_modulation.domain_model
        if attn_modulation_model:
            filename = filename.replace('.csv', f'_{attn_modulation_model}_attn_modulation.csv')
            suffix = suffix + f'_{attn_modulation_model}_attn_modulation'
            on_domain_name_type = config.eval.attention_modulation.on_domain_name_type
            if on_domain_name_type:
                filename = filename.replace('.csv', '_on_domain_name_type.csv')
                suffix = suffix + '_on_domain_name_type'
    pd.DataFrame(results['performance']).to_csv(Path(logger.log_dir) / filename)
    target_types = slot_encoding.get_target_slots()
    df_preds = pd.DataFrame(results['predictions'], index=results['guids'], columns=target_types)
    df_preds.to_csv(Path(logger.log_dir) / f'preds_test_{suffix}.csv')
    df_targets = pd.DataFrame(results['targets'], index=results['guids'], columns=target_types)
    df_targets.to_csv(Path(logger.log_dir) / f'targets_test_{suffix}.csv')
