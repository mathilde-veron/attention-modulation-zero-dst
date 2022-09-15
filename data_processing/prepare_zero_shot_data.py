from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Tuple

import csv

from utils import read_tsv, load_ontology, save_json


def save_tsv(data: Dict[Any, List[Dict]], dst: Path, headers):
    num_turns = sum([len(t) for _, t in data.items()])
    print(f'Write {dst.name} with {len(data)} dialogues and {num_turns} turns')
    with open(str(dst), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter='\t')
        writer.writeheader()
        for dialog_id, turns in data.items():
            writer.writerows(turns)


def remove_domain(ontology: Dict[str, List[str]], unknown_domain: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    ont_known, ont_unknown = {}, {}
    for s_type, s_values in ontology.items():
        if s_type.split('-')[0] == unknown_domain:
            ont_unknown[s_type] = s_values
        else:
            ont_known[s_type] = s_values
    return ont_known, ont_unknown


def split_data_domain(sce_dir: Path, dst_dir: Path, unknown_domain: str, test_data_mandatory: bool = False):
    print(f'Split train, dev and test in known and unknown for LL in {dst_dir}')

    assert unknown_domain in ['attraction', 'bus', 'hospital', 'hotel', 'restaurant', 'taxi', 'train']
    if test_data_mandatory:
        assert unknown_domain not in ['hospital', 'bus']

    # load original ontology
    ontology = load_ontology(sce_dir / "ontology.json", test_data_mandatory, verbose=True)

    ontology_known, ontology_unknown = remove_domain(ontology, unknown_domain)
    save_json(ontology_known, dst_dir / 'ontology_known.json', verbose=True)
    save_json(ontology_unknown, dst_dir / 'ontology_unknown.json', verbose=True)

    s_types = list(ontology_unknown.keys())
    for corpus in ['train', 'dev', 'test']:
        print(f'Split original {corpus} corpus in known and unknown... ', end="", flush=True)
        known, unknown = {}, {}
        path = sce_dir / f'{corpus}.tsv'
        data = read_tsv(path)
        headers = data[list(data.keys())[0]][0].keys()
        for dialogue_id, turns in data.items():
            appended = False
            for turn in turns:
                for s_type in s_types:
                    if turn[s_type] in ontology_unknown[s_type]:
                        unknown[dialogue_id] = turns
                        appended = True
                        break
                if appended:
                    break
            if not appended:
                known[dialogue_id] = turns

        print(f'--- Save {corpus} generated corpus ---')
        for name, data in [('known', known), ('unknown', unknown)]:
            save_tsv(data, dst_dir / f'{corpus}_{name}.tsv', headers)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('sce', type=str, help='Source directory path where the SUMBT tsv files and the ontology are.')
    parser.add_argument('dst', type=str, help='Destination directory path, create it if it does not exist.')
    parser.add_argument('-d', '--domains', nargs='+', type=str,
                        default=['attraction', 'hotel', 'restaurant', 'taxi', 'train'],
                        help="Domains to generate zero-shot data (one sub-directory per domain).")
    args = parser.parse_args()

    sce_path = Path(args.sce)
    dst_path = Path(args.dst)

    dst_path.mkdir(exist_ok=True)

    for domain in args.domains:
        split_data_domain(sce_path, dst_path, unknown_domain=domain, test_data_mandatory=True)
