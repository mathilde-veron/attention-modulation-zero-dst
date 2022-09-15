import multiprocessing
from pathlib import Path
from pickle import dump, load
from random import shuffle
from re import findall
from typing import Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Union

import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase, BatchEncoding

from utils import Config

TokenizerBuilder = Callable[[Config], PreTrainedTokenizerBase]


def read_multiwoz_tsv(file: Path, target_slot_types: List[str]) -> pd.DataFrame:
    """
    Read a tsv file from the MultiWOZ dataset
    :param file: Path
        path to the path
    :param target_slot_types: List[str]
        List of target slot types to consider
    :return: pd.Dataframe
        MultiWOZ data grouped by dialog so that a dialog could not be split when loading a batch
    """
    columns = ['# Dialogue ID', 'Turn Index', 'User Utterance', 'System Response']
    data_row = pd.read_csv(file, sep='\t', encoding='utf-8', usecols=columns.extend(target_slot_types))
    data_row = data_row.rename(columns={'# Dialogue ID': 'dialogue_id',
                                        'Turn Index': 'turn_idx',
                                        'User Utterance': 'text_a',
                                        'System Response': 'text_b'})

    def get_guid(split_type: str, dialogue_id: str, turn_idx: Union[int, str]) -> str:
        return f"{split_type}-{dialogue_id}-{turn_idx}"

    data_row['guid'] = data_row.apply(lambda x: get_guid(file.name, x.dialogue_id, x.turn_idx), axis=1)
    data_row['slots'] = data_row[target_slot_types].apply(lambda x: x.to_dict(), axis=1)
    data_row = data_row[['dialogue_id', 'guid', 'text_a', 'text_b', 'slots']]

    data_row = data_row.groupby('dialogue_id').agg(lambda x: x.tolist())

    return data_row


def read_ontology(file: Union[Path, str], domains_to_rm: List[str] = None) -> Dict[str, List[str]]:
    """
        Read a json ontology
        :param file: Path
            path to the path
        :param domains_to_rm: List[str]
            List of domains which
        :return: Dict[str, List[str]]
            Dict with the domain-slot-type as dict key and a list of associated slot-values as dict value
        """
    with open(str(file)) as f:
        ontology = json.load(f)

    for slot_type in list(ontology):
        if domains_to_rm and any([slot_type[:len(domain)] == domain for domain in domains_to_rm]):
            del ontology[slot_type]
        else:
            ontology[slot_type].append('none')
            ontology[slot_type].append('do not care')

            ontology[slot_type] = list(set(ontology[slot_type]))
            # The values have to be sorted to insure reproducibility
            ontology[slot_type].sort()
            shuffle(ontology[slot_type])

    return ontology


def read_csv(file: Union[Path, str], target_slot_types: List[str]):
    df = pd.read_csv(file, sep=',', encoding='utf-8')
    df.set_index('domain-slot', inplace=True)
    data = df.to_dict(orient='index')
    data = {k: v for k, v in data.items() if k in target_slot_types}
    return data


class SlotEncoding:
    """
    Encode single turn slots annotations as list of integers.
    The encoding depends on the initialization and can be updated on-the-fly when encoding new slot-types/values
    """

    def __init__(self, config: Config, ontology_name: str, save_dir: Path = None):
        self.data_dir = Path(config.data.dir)
        self.domains_to_rm = config.data.domains_to_rm
        self.ontology = read_ontology(self.data_dir / ontology_name, self.domains_to_rm)

        # slot-type conversion dicts
        self.type2int: Dict[str, int] = {}
        self.load_type2int(save_dir)
        self.int2type: Dict[int, str] = {i: slot_type for slot_type, i in self.type2int.items()}

        # slot-value conversion dicts
        # Their encoding depend on the associated slot-type since some values are shared across slot-types
        self.value2int: Dict[str, Dict[str, int]] = {}
        self.load_value2int(save_dir)
        self.int2value: Dict[int, Dict[int, str]] = {i_type: {i_value: value for value, i_value in self.value2int[slot_type].items()}
                                                     for slot_type, i_type in self.type2int.items()}

    def _update_from_slots(self, slots: Dict[str, str]):
        """
        Update conversion dict attributes if slots contains new slot-types or new slot-values
        """
        for slot_type, slot_value in slots.items():
            if slot_type not in self.type2int:
                type_int = len(self.type2int)
                self.type2int[slot_type] = type_int
                self.int2type[type_int] = slot_type
                self.value2int[slot_type] = {slot_value: 0}
                self.int2value[type_int] = {0: slot_value}
            elif slot_value not in self.value2int[slot_type]:
                type_int = self.type2int[slot_type]
                value_int = len(self.value2int[slot_type])
                self.value2int[slot_type][slot_value] = value_int
                self.int2value[type_int][value_int] = slot_value

    def update_from_ontology(self, ontology_name: str):
        new_ontology = read_ontology(self.data_dir / ontology_name, self.domains_to_rm)

        for slot_type, slot_values in new_ontology.items():
            if slot_type not in self.type2int:
                type_int = len(self.type2int)
                self.type2int[slot_type] = type_int
                self.int2type[type_int] = slot_type
                self.value2int[slot_type] = {v: i for i, v in enumerate(slot_values)}
                self.int2value[type_int] = {i: v for i, v in enumerate(slot_values)}
            else:
                for value in slot_values:
                    if value not in self.value2int[slot_type]:
                        type_int = self.type2int[slot_type]
                        value_int = len(self.value2int[slot_type])
                        self.value2int[slot_type][value] = value_int
                        self.int2value[type_int][value_int] = value

    def save(self, save_dir: Path):
        with open(Path(save_dir) / 'slot_encoding_type2int.pkl', 'wb') as f:
            dump(self.type2int, f)
        with open(Path(save_dir) / 'slot_encoding_value2int.pkl', 'wb') as f:
            dump(self.value2int, f)

    def load_type2int(self, save_dir: Path = None):
        if save_dir:
            with open(Path(save_dir) / 'slot_encoding_type2int.pkl', 'rb') as f:
                self.type2int = load(f)
        else:
            self.type2int = {slot_type: i for i, slot_type in enumerate(self.ontology.keys())}

    def load_value2int(self, save_dir: Path = None):
        if save_dir:
            with open(Path(save_dir) / 'slot_encoding_value2int.pkl', 'rb') as f:
                self.value2int = load(f)
        else:
            self.value2int = {slot_type: {value: i for i, value in enumerate(slot_values)}
                              for slot_type, slot_values in self.ontology.items()}

    def get_target_slots(self) -> List[str]:
        return list([self.get_type_name(i) for i in range(self.get_num_types())])

    def get_target_domains(self) -> List[str]:
        return list(set([t.split('-')[0] for t in self.type2int.keys()]))

    def get_domain_type_code(self, domain: str) -> List[int]:
        return [i for t, i in self.type2int.items() if t[:len(domain)] == domain]

    # def get_ontology(self) -> Dict[str, List[str]]:
    #     return self.ontology

    def get_type_name(self, type_code: int) -> str:
        return self.int2type[type_code]

    def get_value_name(self, type_code: int, value_code: int) -> str:
        assert type_code in self.int2value.keys(), f'{type_code} slot type index not in the slot encoding'
        assert value_code in self.int2value[type_code].keys(), f'{value_code} slot value index not in the slot encoding of slot type index {type_code} ({self.get_type_name(type_code)})'
        return self.int2value[type_code][value_code]

    def get_type_code(self, type_name: str) -> int:
        return self.type2int[type_name]

    def get_value_code(self, type_name, value_name) -> int:
        return self.value2int[type_name][value_name]

    def get_num_types(self):
        return len(self.type2int)

    def get_num_values(self, slot_type: Union[int, str]):
        assert isinstance(slot_type, (int, str))
        if isinstance(slot_type, int):
            return len(self.int2value[slot_type])
        else:
            return len(self.value2int[slot_type])

    def codify_slots(self, slots: Dict[str, str], update: bool = False) -> List[int]:
        """
        Encode single turn slot annotations
        :param slots: Dict[str, str]
            single turn (user + system) annotations with the slot-types as dict keys and the associated slot-value as
            dict value corresponding to the turn.
        :param update: bool
            whether to update the conversion dicts on-the-fly if slots contains new values
        :return: List[int]
            encoded annotations, each element corresponds to an encoded slot-values, the index denoting the encoded
            slot-type
        """
        if update:
            self._update_from_slots(slots)

        encoded_slots = [-1] * len(self.type2int)
        for slot_type, slot_value in slots.items():
            assert slot_type in self.type2int, f"Slot type '{slot_type}' is not in the provided ontology"
            assert slot_value in self.value2int[slot_type], f"Slot value '{slot_value}' of slot type '{slot_type}' is not in the provided ontology"
            encoded_slots[self.type2int[slot_type]] = self.value2int[slot_type][slot_value]

        return encoded_slots

    def decodify_slots(self, slot_codes) -> Dict[str, str]:
        return {self.int2type[type_code]: self.int2value[type_code][value_code]
                for type_code, value_code in enumerate(slot_codes)}


class Ontology(Dict):
    """
    Ontology used by SumbtLL model during training and testing.
    Dict with the domain-slot-type as dict key and a list of associated slot-values as dict value.
    """

    def __init__(
            self,
            ontology_name: str,
            config: Config,
            tokenizer: TokenizerBuilder,
            slot_encoding: SlotEncoding
    ):
        self.max_label_length = config.data.max_label_length
        self.root = Path(config.data.dir)
        self.domains_to_rm = config.data.domains_to_rm
        self.use_slot_desc = config.train.use_slot_desc
        self.domain_name_type_queries = config.train.domain_name_type_queries
        self.domain_name_type_attn = config.train.domain_name_type_attn
        self.new_domains = []

        super().__init__(read_ontology(self.root / ontology_name, self.domains_to_rm))

        self.tokenizer = tokenizer(config)
        self.slot_encoding = slot_encoding
        self.slot_info = {}
        if self.domain_name_type_queries or self.domain_name_type_attn:
            self.tokenized_names: List[BatchEncoding] = []
            self.tokenized_domains: List[BatchEncoding] = []
            self.tokenized_value_types: List[BatchEncoding] = []
            self.tokenized_types = None
        else:
            self.tokenized_types: List[BatchEncoding] = []
        self.tokenized_values: List[List[BatchEncoding]] = []

        self.update_slot_info()

        self.update_tokenized()

    def update_slot_info(self):
        if self.use_slot_desc or self.domain_name_type_queries or self.domain_name_type_attn:
            self.slot_info = read_csv(
                self.root.parents[1] / ('slot_description.csv' if self.use_slot_desc else 'domain_name_type.csv'),
                self.slot_encoding.get_target_slots()
            )

    def get_num_types(self) -> int:
        return len(self.keys())

    def get_domains(self) -> List[str]:
        return list(set([t.split('-')[0] for t in self.keys()]))

    def get_new_domains(self) -> List[str]:
        return self.new_domains

    def extend(self, ontology_name: str):
        """
        Extend the current ontology with a new ontology
        :param ontology_name: str
            name of the new ontology
        """
        domains_before = self.get_domains()
        new_ontology = read_ontology(self.root / ontology_name, self.domains_to_rm)
        for slot_type, slot_values in new_ontology.items():
            if slot_type not in self:
                self[slot_type] = slot_values
            else:
                self[slot_type] = list(set(self[slot_type]).union(slot_values))

        self.update_slot_info()
        self.update_tokenized()

        self.new_domains.extend(list(set(self.get_domains()).difference(domains_before)))

    def update_tokenized(self):
        get_domain_name_type_tokens = self.domain_name_type_queries or self.domain_name_type_attn
        num_type_tokenized = len(self.tokenized_types if not get_domain_name_type_tokens else self.tokenized_names)
        num_type_encoded = self.slot_encoding.get_num_types()
        if num_type_encoded > num_type_tokenized:
            if not get_domain_name_type_tokens:
                self.tokenized_types.extend([BatchEncoding()] * (num_type_encoded - num_type_tokenized))
            else:
                self.tokenized_names.extend([BatchEncoding()] * (num_type_encoded - num_type_tokenized))
                self.tokenized_domains.extend([BatchEncoding()] * (num_type_encoded - num_type_tokenized))
                self.tokenized_value_types.extend([BatchEncoding()] * (num_type_encoded - num_type_tokenized))
            self.tokenized_values.extend([[]] * (num_type_encoded - num_type_tokenized))

        for slot_type, slot_values in self.items():
            type_int = self.slot_encoding.get_type_code(slot_type)
            if not get_domain_name_type_tokens:
                if not self.tokenized_types[type_int]:
                    slot_str = slot_type if not self.use_slot_desc else self.slot_info[slot_type]['description']
                    self.tokenized_types[type_int] = self._tokenize_label(slot_str)
            else:
                if not self.tokenized_names[type_int]:
                    self.tokenized_names[type_int] = self._tokenize_label(self.slot_info[slot_type]['name'])
                    self.tokenized_domains[type_int] = self._tokenize_label(self.slot_info[slot_type]['domain'])
                    self.tokenized_value_types[type_int] = self._tokenize_label(self.slot_info[slot_type]['value_type'])

            # fixme: bug if the list is not copied, all lists of self.tokenized_values are inexplicably extended if
            #  num_value_encoded > num_value_tokenized
            self.tokenized_values[type_int] = self.tokenized_values[type_int].copy()
            num_value_tokenized = len(self.tokenized_values[type_int])
            num_value_encoded = self.slot_encoding.get_num_values(slot_type)
            if num_value_encoded > num_value_tokenized:
                self.tokenized_values[type_int].extend([BatchEncoding()] * (num_value_encoded - num_value_tokenized))
            for value in slot_values:
                value_int = self.slot_encoding.get_value_code(slot_type, value)
                assert len(self.tokenized_values[type_int]) > value_int, \
                    f"{slot_type}: {type_int}, {value}: {value_int} >= {len(self.tokenized_values[type_int])}, {self.slot_encoding.value2int[slot_type]}"
                if not self.tokenized_values[type_int][value_int]:
                    self.tokenized_values[type_int][value_int] = self._tokenize_label(value)

    def format_tokenized(self, original_tokenized):
        tokenized = {'input_ids': [], 'attention_mask': [], 'token_type_ids': []}
        for tk in original_tokenized:
            tokenized['input_ids'].append(tk['input_ids'])
            tokenized['attention_mask'].append(tk['attention_mask'])
            tokenized['token_type_ids'].append(tk['token_type_ids'])
        tokenized['input_ids'] = torch.stack(tokenized['input_ids']).view(-1, self.max_label_length)
        tokenized['attention_mask'] = torch.stack(tokenized['attention_mask']).view(-1, self.max_label_length)
        tokenized['token_type_ids'] = torch.stack(tokenized['token_type_ids']).view(-1, self.max_label_length)
        return tokenized

    def get_tokenized(self) -> NamedTuple:
        """
        Get and format the tokenized slot labels needed by SUMBT_LL
        :return: NamedTuple
            tokenized slot labels in a format adapted to BERT models
            each tensor index corresponds to the integer of an encoded slot type/value
            each list index of the tokenized slot values correspond to the integer of an encoded slot type
        """
        Tokenized = NamedTuple('Tokenized', [('types', Dict[str, torch.Tensor]),
                                             ('names', Dict[str, torch.Tensor]),
                                             ('domains', Dict[str, torch.Tensor]),
                                             ('value_types', Dict[str, torch.Tensor]),
                                             ('values', List[Dict[str, torch.Tensor]])])

        # tokenize slot types
        if not self.domain_name_type_queries and not self.domain_name_type_attn:
            tokenized_types = self.format_tokenized(self.tokenized_types)
            tokenized_names, tokenized_domains, tokenized_value_types = None, None, None
        else:
            tokenized_names = self.format_tokenized(self.tokenized_names)
            tokenized_domains = self.format_tokenized(self.tokenized_domains)
            tokenized_value_types = self.format_tokenized(self.tokenized_value_types)
            tokenized_types = None

        # tokenize slot values
        tokenized_values = []
        for tk_values in self.tokenized_values:
            tokenized_values.append(self.format_tokenized(tk_values))

        return Tokenized(tokenized_types, tokenized_names, tokenized_domains, tokenized_value_types, tokenized_values)

    def _tokenize_label(self, label: str) -> BatchEncoding:
        tokenized = self.tokenizer(
            text=label,
            padding='max_length',
            truncation=True,
            max_length=self.max_label_length,
            return_tensors='pt')  # Return PyTorch torch.Tensor objects
        return tokenized


class MultiWOZSplit(Dataset):
    """
    A split of a MultiWOZ subset.
    Data are grouped by dialog.
    """
    def __init__(
            self,
            slot_encoding: SlotEncoding,
            guids: pd.Series,
            texts_a: pd.Series,
            texts_b: pd.Series,
            slots: pd.Series,
            update_encoding=False
    ):
        self.slot_encoding = slot_encoding
        self.update_encoding = update_encoding
        self.guids = guids        # pd.Series with elements of type List[str]
        self.texts_a = texts_a    # pd.Series with elements of type List[str]
        self.texts_b = texts_b    # pd.Series with elements of type List[str]
        self.slots = slots        # pd.Series with elements of type List[Dict[str]]

    def __len__(self):
        return len(self.guids)

    def __getitem__(self, idx):
        encoded_slots = [self.slot_encoding.codify_slots(slots, self.update_encoding)
                         for slots in self.slots.iloc[idx]]
        return (self.texts_a.iloc[idx],     # List[str]
                self.texts_b.iloc[idx],     # List[str]
                encoded_slots,              # List[List[int]]
                self.guids.iloc[idx])       # List[str]


class MultiWOZSubset:
    """
    A subset of the original MultiWOZ dataset.
    It can provide a pytorch DataLoader to load batches form a specific split in the correct format for BERT.
    """
    def __init__(self, config: Config, subset: str, slot_encoding: SlotEncoding, tokenizer: PreTrainedTokenizerBase):
        self.config = config
        self.subset = subset
        self.slot_encoding = slot_encoding
        self.tokenizer = tokenizer
        self.root = Path(config.data.dir)
        self.max_seq_length = config.data.max_seq_length
        self.max_turn_length = config.data.max_turn_length

    def _tokenize(self, texts_a: List[str], texts_b: List[str]):
        """
        Tokenize and encode each pair of texts_a and texts_b and add special tokens
        "[CLS] <text_a> [SEP] <text_b> [SEP]"
        """
        texts_tokenized = self.tokenizer(
            text=texts_a,
            text_pair=texts_b,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt')    # Return PyTorch torch.Tensor objects
        return texts_tokenized

    def _collate(self, batch: List[Tuple[List[str], List[str], List[List[int]], List[str]]]) -> Dict:
        # collate turns and pad dialogs so that all dialog has the same length in terms of turns
        texts_a, texts_b, all_value_labels, guids = [], [], [], []
        num_types = len(self.slot_encoding.type2int)
        for dialog in batch:
            num_turns = min(len(dialog[0]), self.max_turn_length - 1)
            # collate
            texts_a.extend(dialog[0][:num_turns])
            texts_b.extend(dialog[1][:num_turns])
            all_value_labels.extend(dialog[2][:num_turns])
            guids.extend(dialog[3][:num_turns])
            # pad
            for _ in range(num_turns, self.max_turn_length):
                texts_a.append('')
                texts_b.append('')
                all_value_labels.append([-1] * num_types)
                guids.append('')

        # tokenize the user's and the system's utterances
        texts_b = ['' if pd.isna(t) else t for t in texts_b]
        tokenized = self._tokenize(texts_a, texts_b)

        # reshape tensors
        tokenized['input_ids'] = tokenized['input_ids'].view(-1, self.max_turn_length, self.max_seq_length)
        tokenized['token_type_ids'] = tokenized['token_type_ids'].view(-1, self.max_turn_length, self.max_seq_length)
        if 'attention_mask' in tokenized:
            tokenized['attention_mask'] = tokenized['attention_mask'].view(-1, self.max_turn_length, self.max_seq_length)
        all_value_labels = torch.LongTensor(all_value_labels).view(-1, self.max_turn_length, num_types)

        return {'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'] if 'attention_mask' in tokenized else None,
                'token_type_ids': tokenized['token_type_ids'],
                'value_labels': all_value_labels,
                'guids': guids}

    def get_loader(
            self,
            split: str,
            batch_size: int = 32,
            shuffle: bool = False,
            num_workers: Optional[int] = None
    ) -> DataLoader:

        assert split in ['train', 'dev', 'test'], "Split should be either `train`, `dev` or `test`"
        data = read_multiwoz_tsv(
            self.root / f"{split}_{self.subset}.tsv",
            self.slot_encoding.get_target_slots()
        )
        dataset = MultiWOZSplit(
            self.slot_encoding,
            data.guid,
            data.text_a,
            data.text_b,
            data.slots,
            update_encoding=(split == 'test')
        )

        if num_workers is None:
            num_workers = multiprocessing.cpu_count() // 2

        return DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          collate_fn=self._collate,
                          shuffle=shuffle,
                          num_workers=num_workers)


class MultiWOZ:
    """
    MultiWOZ dataset to gather general variables.
    No data are loaded when initializing a MultiWOZ object.
    """
    def __init__(self, config: Config, slot_encoding: SlotEncoding, tokenizer: TokenizerBuilder):
        self.config = config
        self.slot_encoding = slot_encoding
        self.tokenizer = tokenizer(config)
        root = Path(config.data.dir)
        self.available_subsets = [findall(r'(?:(?:train)|(?:dev)|(?:test))_([a-z_]+)\.tsv', file.name)[0]
                                  for file in root.iterdir()
                                  if file.is_file() and file.suffix == '.tsv']

    def __contains__(self, subset: str) -> bool:
        return subset in self.available_subsets

    def __getitem__(self, subset: str) -> MultiWOZSubset:
        assert subset in self, f"Subset '{subset}' not in the list of available subsets {self.available_subsets}"
        return MultiWOZSubset(self.config, subset, self.slot_encoding, self.tokenizer)
