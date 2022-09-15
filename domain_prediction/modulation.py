from typing import List

import torch
from transformers import BertTokenizer

from .oracle import Oracle


class Modulator:
    def __init__(self, config, new_domains: List[str] = None):
        self.beta = config.beta
        self.new_domains = new_domains
        if config.domain_model == 'oracle':
            self.model = Oracle(config.domain_references)
        else:
            self.model = None
        self.tokenizer = BertTokenizer.from_pretrained(config.tokenizer_name_or_path)
        self.special_tokens = self.get_special_tokens_ids()

    def get_special_tokens_ids(self, special_tokens: List[str] = ['[CLS]', '[SEP]', '[PAD]']):
        tokenized_tokens = []
        for token in special_tokens:
            tokenized = self.tokenizer(
                text=token,
                add_special_tokens=False,
                padding=False,
                return_tensors='np')  # Return Numpy np.ndarray objects
            tokenized_tokens.append(tokenized['input_ids'][0][0])
        return tokenized_tokens

    def domain2alpha(self, input_ids, domain_preds, slots: List[str]):
        alphas = []
        ds, ts, max_seq_length = input_ids.size()
        num_slots = len(slots)
        input_ids = input_ids.view(len(domain_preds), max_seq_length)
        for slot in slots:
            slot_domain = slot.split('-')[0]
            for turn_idx, domain_pred in enumerate(domain_preds):
                if domain_pred not in self.new_domains or domain_pred == 'none':
                    alphas.append([1]*max_seq_length)
                else:
                    turn_alphas = []
                    if slot_domain == domain_pred:
                        for token in input_ids[turn_idx, :].tolist():
                            if token in self.special_tokens:
                                turn_alphas.append(self.beta)
                            else:
                                turn_alphas.append(1)
                    else:
                        turn_alphas.extend([1]*max_seq_length)
                    alphas.append(turn_alphas)
        alphas = torch.LongTensor(alphas).view(ds*ts*num_slots, max_seq_length)
        # assert not torch.equal(alphas, torch.ones_like(alphas)), f"alphas tensor only has ones in it."
        return alphas

    def get_alphas(self, input_ids, guids, slots: List[str]):
        if not self.model:
            alphas = None
        else:
            domains = self.model.predict_domains(input_ids, guids)
            alphas = self.domain2alpha(input_ids, domains, slots)
        return alphas
