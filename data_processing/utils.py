import csv
import json
from logging import Logger
from pathlib import Path
from re import match
from typing import Any, Dict, List

import pandas as pd


def read_tsv(path: Path, logger: Logger = None) -> Dict[Any, List[Dict]]:
    """
    Reads the tsv file corresponding to path and returns the associated data per dialogue ID
    """
    data = {}
    with open(str(path), "r", encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row['# Dialogue ID'] not in data:
                data[row['# Dialogue ID']] = [row]
            else:
                data[row['# Dialogue ID']].append(row)
    num_turns = sum([len(t) for _, t in data.items()])

    info = f"Load {path.name} with {len(data)} dialogues and {num_turns} turns"
    if logger:
        logger.info(info)
    else:
        print(info)

    return data


def load_ontology(file: Path,
                  test_data_mandatory: bool = False,
                  logger: Logger = None,
                  verbose: bool = False
                  ) -> Dict:
    if verbose:
        if logger:
            logger.info(f'--- Load ontology {file.name} ---')
        else:
            print(f'Load ontology {file.name}')
    with open(str(file), "r") as f:
        ontology = json.load(f)

    ontology = format_ontology(ontology, test_data_mandatory, logger, verbose)

    return ontology


def extract_ontology(
        sce: Path,
        test_data_mandatory: bool = False,
        logger: Logger = None,
        verbose: bool = False
) -> Dict:

    data = pd.read_csv(sce, sep='\t')

    ontology = {}
    for col_name in list(data):
        unused_col = ['# Dialogue ID', 'Turn Index', 'User Utterance', 'System Response']
        if col_name in unused_col or match(r'Unnamed: [0-9]+', col_name):
            continue
        ontology[col_name] = list(set(data[col_name].tolist()))

    ontology = format_ontology(ontology, test_data_mandatory, logger, verbose)

    return ontology


def format_ontology(
        ontology: Dict[str, List[str]],
        test_data_mandatory: bool = False,
        logger: Logger = None,
        verbose: bool = False
) -> Dict[str, List[str]]:

    if test_data_mandatory:
        info = "Remove slot-values from ontology for domains without test data (bus and hospital domain)"
        if logger:
            logger.info(info)
        else:
            print(info)
        slot_type_to_remove = [t for t in ontology.keys() if t[:4] == 'bus-' or t[:9] == 'hospital-']
        for slot_type in slot_type_to_remove:
            ontology[slot_type] = []

    for s_type, s_values in ontology.items():
        if "do not care" in s_values:
            ontology[s_type].remove("do not care")
        if "none" in s_values:
            ontology[s_type].remove("none")
        if verbose:
            if logger:
                logger.info(f'{s_type}: \t{len(s_values)}')
            else:
                print(f'{s_type}: \t{len(s_values)}')

    for slot_type in list(ontology):
        ontology[slot_type] = list(set(ontology[slot_type]))

    return ontology


def extend_ontology(ont1: Dict, ont2: Dict) -> Dict:
    if not ont1:
        return ont2
    else:
        ont = ont1.copy()
        for s_type, s_values in ont2.items():
            if s_type in ont:
                ont[s_type] = list(set(ont[s_type] + s_values))
            else:
                ont[s_type] = s_values
        return ont


def save_json(data: Dict, dst: Path, logger: Logger = None, verbose=False):
    info = f'Write {dst.name}'
    if logger:
        logger.info(info)
    else:
        print(info)

    if verbose:
        # print number of values per slot type
        for s_type, s_values in data.items():
            logger.info(f'{s_type}: \t{len(s_values)} \t(preview: {s_values[:5]})')

    with open(str(dst), 'w') as f:
        json.dump(data, f, indent=4)
