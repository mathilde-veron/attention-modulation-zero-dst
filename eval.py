import argparse
from pathlib import Path

import pytorch_lightning as pl
import pandas as pd
from re import findall

from dataset import SlotEncoding, MultiWOZ, Ontology
from model import SumbtLL, BERTTokenizer
from utils import Config

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True, help="Experiment directory where config.yml is located")
parser.add_argument('--ckpt_name', type=str, required=False, help="Name of the checkpoint to evaluate")
args = parser.parse_args()
args.dir = Path(args.dir)

config = Config(args.dir / "config.yml")

print("Loading ontology... ", end="", flush=True)
ontology_name = f'ontology_{config.train.ontology_subset}.json'
slot_encoding = SlotEncoding(config, ontology_name, save_dir=args.dir)
ontology = Ontology(ontology_name, config, BERTTokenizer, slot_encoding)
print("OK")

print("Loading dataset... ", end="", flush=True)
dataset = MultiWOZ(config, slot_encoding, BERTTokenizer)
print("OK")

# Load the best checkpoint
print("Loading best checkpoint...   ", end="", flush=True)
ckpt_dir = Path(args.dir / 'logs' / 'checkpoints')
if args.ckpt_name:
    best_ckpt = ckpt_dir / args.ckpt_name
else:
    ckpt_paths = [p for p in ckpt_dir.iterdir()]
    last_epoch = 0
    for p in ckpt_paths:
        ckpt_epoch = int(findall(r'epoch=([0-9]+)-step=[0-9]+.ckpt', p.name)[0])
        if ckpt_epoch > last_epoch:
            last_epoch = ckpt_epoch
            best_ckpt = p
model = SumbtLL.load_from_checkpoint(
    checkpoint_path=best_ckpt,
    dataset=dataset,
    config=config,
    slot_encoding=slot_encoding,
    ontology=ontology
)
print("OK")

trainer = pl.Trainer()

log_dir = Path(args.dir / 'logs')
for subset in config.eval.subsets:
    print('\n')
    print(f"Evaluate the trained model on {subset} data")

    filename = f"all_accuracies_{subset}_test.csv"
    if config.eval.attention_modulation:
        attn_modulation_model = config.eval.attention_modulation.domain_model
        if attn_modulation_model:
            filename = filename.replace('.csv', f'_{attn_modulation_model}_attn_modulation.csv')
            on_domain_name_type = config.eval.attention_modulation.on_domain_name_type
            if on_domain_name_type:
                filename = filename.replace('.csv', '_on_domain_name_type.csv')
    if Path(log_dir / filename).exists():
        print(f'Skip evaluation. {filename} already exists (delete file to re-run evaluation on {subset}).')
        continue

    # Update the slot encoding and extend the ontology the model has access to if needed
    if subset != config.train.ontology_subset:
        print(f"Extend current ontology with {subset} ontology...   ", end="", flush=True)
        slot_encoding.update_from_ontology(f'ontology_{subset}.json')
        model.update_ontology(f'ontology_{subset}.json')
        print('OK')

    # Load evaluation data
    print("Loading evaluation data...   ", end="", flush=True)
    dataset_eval = dataset[subset]
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
    pd.DataFrame(results['performance']).to_csv(log_dir / filename)

    target_types = slot_encoding.get_target_slots()
    suffix = subset
    if config.eval.attention_modulation:
        attn_modulation_model = config.eval.attention_modulation.domain_model
        if attn_modulation_model:
            suffix = suffix + f'_{attn_modulation_model}_attn_modulation'
            on_domain_name_type = config.eval.attention_modulation.on_domain_name_type
            if on_domain_name_type:
                suffix = suffix + '_on_domain_name_type'
    df_preds = pd.DataFrame(results['predictions'], index=results['guids'], columns=target_types)
    df_preds.to_csv(log_dir / f'preds_test_{suffix}.csv')
    df_targets = pd.DataFrame(results['targets'], index=results['guids'], columns=target_types)
    df_targets.to_csv(log_dir / f'targets_test_{suffix}.csv')
