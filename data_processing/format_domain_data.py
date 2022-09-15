from argparse import ArgumentParser
from copy import deepcopy
import json
from pathlib import Path
import sys

import pandas as pd


parser = ArgumentParser()
parser.add_argument('sce', type=str, help='Source directory path where the MultiWOZ 2.1 json files are.')
parser.add_argument('-d', '--domains', nargs='+', type=str,
                    default=['attraction', 'hotel', 'restaurant', 'taxi', 'train'],
                    help="Domains to consider, else the uttenrance is annotated with 'none'.")
args = parser.parse_args()

data_dir = Path(args.sce)
save_dir = data_dir / 'domain'
save_dir.mkdir(exist_ok=True)

# multi_domain_output = sys.stdout
multi_domain_output = open(save_dir / 'multi_domain_turns.txt', 'w')
annotated_base = {'dialog_id': None, 'turn_idx': None, 'user_utt': None, 'syst_utt': None, 'domain': None}

with open(data_dir / 'valListFile.txt') as f_dev:
    dev_dialog_ids = f_dev.read().splitlines()
with open(data_dir / 'testListFile.txt') as f_test:
    test_dialog_ids = f_test.read().splitlines()

dialogs = json.load(open(data_dir / 'data.json'))

all_annotated_train, all_annotated_dev, all_annotated_test = [], [], []
multi_domain_turns = 0
for d_id, dialog in dialogs.items():
    t_idx = 0
    annotated = deepcopy(annotated_base)
    for turn in dialog['log']:
        # User turn
        if len(turn['metadata']) == 0:
            if 'dialog_act' not in turn:
                print(f'WARNING no dialog act in dialog {d_id} at turn {t_idx}')
                continue
            # get the domain associated to the current turn
            turn_domains = set()
            for act, _ in turn['dialog_act'].items():
                domain = act.split('-')[0].lower()
                domain = domain if domain in args.domains else 'none'
                turn_domains.add(domain)
            # don't add turn to data that can be associated to multiple domains
            if len(turn_domains) > 1:
                multi_domain_turns += 1
                print(f"\t{d_id}, \t{t_idx}, \t{turn['text']}, \tdomains: {turn_domains}", file=multi_domain_output)
                continue
            turn_domain = 'none' if not turn_domains else list(turn_domains)[0]
            # add annotated turn to data
            if turn['text'] != '':
                annotated['dialog_id'] = d_id
                annotated['turn_idx'] = t_idx
                annotated['user_utt'] = turn['text']
                annotated['domain'] = turn_domain
                if d_id in dev_dialog_ids:
                    all_annotated_dev.append(annotated)
                elif d_id in test_dialog_ids:
                    all_annotated_test.append(annotated)
                else:
                    all_annotated_train.append(annotated)

        # System turn
        else:
            annotated = deepcopy(annotated_base)
            t_idx += 1
            annotated['syst_utt'] = turn['text']

num_turns_train, num_turns_dev, num_turns_test = len(all_annotated_train), len(all_annotated_dev), len(all_annotated_test)
num_tot_turns = num_turns_train + num_turns_dev + num_turns_test
multi_domains_prop = multi_domain_turns * 100 / (multi_domain_turns + num_tot_turns)
print(f'Annotated turns in train: {num_turns_train}')
print(f'Annotated turns in dev: {num_turns_dev}')
print(f'Annotated turns in test: {num_turns_test}')
print(f'Total: {num_tot_turns}')
print(f'Multi domains turns: {multi_domain_turns} ({int(multi_domains_prop)}%)')

if multi_domain_output != sys.stdout:
    multi_domain_output.close()

for split, data in [('train', all_annotated_train), ('dev', all_annotated_dev), ('test', all_annotated_test)]:
    df = pd.DataFrame(data)
    df.rename(
        {'dialog_id': '# Dialogue ID',
         'turn_idx': 'Turn Index',
         'user_utt': 'User Utterance',
         'syst_utt': 'System Response',
         'domain': 'Domain'}
    )

    df.to_csv(save_dir / f'{split}.tsv', encoding='utf-8', sep='\t')
