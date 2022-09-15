from pathlib import Path
from typing import Union

import pandas as pd


class Oracle:
    def __init__(self, path: Union[Path, str]):
        path = Path(path)
        references = pd.read_csv(path, sep='\t', encoding='utf-8', usecols=['dialog_id', 'turn_idx', 'domain'])

        def get_guid(split_type: str, dialogue_id: str, turn_idx: Union[int, str]) -> str:
            return f"{split_type}-{dialogue_id}-{turn_idx}"

        references['guid'] = references.apply(lambda x: get_guid(path.name, x.dialog_id, x.turn_idx), axis=1)
        references.set_index('guid', inplace=True)
        self.references = references[['domain']]

    def predict_domains(self, input_ids, input_guids):
        # if turn id of input not in references returns 'none'
        domains = []
        for input_guid in input_guids:
            input_guid_cleaned = input_guid.replace('_unknown', '').replace('_known', '')
            if input_guid == '' or input_guid_cleaned not in list(self.references.index.values):
                domains.append('none')
            else:
                domains.append(self.references.at[input_guid_cleaned, 'domain'])
        return domains
