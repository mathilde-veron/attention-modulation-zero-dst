from typing import Any, Dict, Text, List, Optional
from pathlib import Path
from pickle import dump
import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.metrics import Accuracy
from transformers import AdamW, BertModel, PreTrainedTokenizerBase, BertTokenizer, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np

from dataset import SlotEncoding, Ontology
from metric import JointAccuracy
from utils import Config
from domain.modulation import Modulator


def BERTTokenizer(cfg: Config) -> PreTrainedTokenizerBase:
    """
    This is a wrapper of the tokenizer from Huggingface
    so there's no possibility of choosing the wrong one.
    """
    return BertTokenizer.from_pretrained(cfg.model.name_or_path)


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        self.scores = None

    def attention(self, q, k, v, d_k, mask=None, dropout=None, alphas=None):

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        if alphas != None:
            scores = alphas * scores

        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        self.scores = scores
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask=None, alphas=None):
        bs = q.size(0)  # here bs is equal to the number of unique turns by the number of slots to predict

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)  # dim 1 = sl max_seq_length
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)  # dim 1 = qs queries size (1|3)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        sl = k.size(1)  # max_seq_length
        qs = q.size(1)  # query size, number of queries in the query matrix (1|3)

        if alphas is not None:
            alphas = alphas.view(bs, qs*sl).repeat(1, self.h)
            alphas = alphas.view(bs, self.h, qs, sl)

        # transpose to get dimensions bs * h * sl * d_k
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout, alphas)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)  # [bs, qs, d_model]
        return output

    def get_scores(self):
        return self.scores


class SumbtLL(pl.LightningModule):
    def __init__(self, config: Config, slot_encoding: SlotEncoding, ontology: Ontology):
        super().__init__()
        self.cfg = config
        self.save_hyperparameters(self.cfg.train.to_dict())
        self.domain_name_type_queries = config.train.domain_name_type_queries
        self.domain_name_type_attn = config.train.domain_name_type_attn
        self.slot_encoding = slot_encoding
        self.ontology = ontology
        # specific to domain attention scores modulation
        self.new_domains = []
        self.attn_modulation_cfg = config.eval.attention_modulation
        self.modulate_attn = False
        if self.attn_modulation_cfg is not None:
            self.modulate_attn = self.attn_modulation_cfg.domain_model is not None
        self.domain_attn_modulator = None

        # Load utterance encoder
        self.utt_encoder = BertModel.from_pretrained(self.cfg.model.name_or_path)
        self.encoder_output_dim = self.utt_encoder.config.hidden_size
        self.hidden_dropout = self.utt_encoder.config.hidden_dropout_prob
        self.freeze_from = 0
        if self.cfg.train.freeze_utt_encoder_from_epoch is not None:
            self.freeze_from = self.cfg.train.freeze_utt_encoder_from_epoch
        if self.cfg.train.freeze_utt_encoder and self.freeze_from == 0:
            for param in self.utt_encoder.parameters():
                param.requires_grad = False

        # Load slot-value and slot-type encoder (not trainable)
        self.sv_encoder = BertModel.from_pretrained(self.cfg.model.name_or_path)
        for param in self.sv_encoder.parameters():
            param.requires_grad = False

        # Initialize slot types/values lookup tables bwt their integer labels and their values encoded by sv_encoder
        self.num_slots = len(ontology)
        self.type_lookup = None
        if self.domain_name_type_queries:
            self.name_lookup, self.domain_lookup, self.value_type_lookup = None, None, None
        self.value_lookup = nn.ModuleList()
        self.init_embeddings_lookup()

        # Debug
        debug = False
        if debug:
            print('--- type/value embeddings after init ---')
            print('shape:', self.type_lookup.weight.shape)
            for i in range(5):
                print(f"- Type slot '{self.slot_encoding.get_type_name(i)}' -")
                print('type emb:', self.type_lookup.weight[i].cpu().detach().tolist()[:5])
                print('associated values')
                print('shape:', self.value_lookup[i].weight.shape)
                for j in range(min(5, self.value_lookup[i].weight.shape[0])):
                    print(f"value '{self.slot_encoding.get_value_name(i, j)}' emb:",
                        self.value_lookup[i].weight[j].cpu().detach().tolist()[:5])

        # Attention layer
        # self.attention = nn.MultiheadAttention(self.encoder_output_dim, self.cfg.train.attention_heads)
        if not self.domain_name_type_attn:
            self.attention = MultiHeadAttention(self.cfg.train.attention_heads, self.encoder_output_dim, dropout=0)
        else:
            self.name_attn = MultiHeadAttention(self.cfg.train.attention_heads, self.encoder_output_dim, dropout=0)
            self.domain_attn = MultiHeadAttention(self.cfg.train.attention_heads, self.encoder_output_dim, dropout=0)
            self.value_type_attn = MultiHeadAttention(self.cfg.train.attention_heads, self.encoder_output_dim, dropout=0)

        if self.domain_name_type_queries or self.domain_name_type_attn:
            self.attn_aggregation = nn.Sequential(
                nn.Linear(self.encoder_output_dim*3, self.encoder_output_dim),
                nn.LeakyReLU()
            )

        # RNN belief tracker
        self.rnn = nn.GRU(
            input_size=self.encoder_output_dim,
            hidden_size=self.cfg.train.hidden_dim,
            num_layers=self.cfg.train.rnn_layers,
            dropout=self.hidden_dropout,
            batch_first=True
        )  # If True, then the input and output tensors are provided as (batch, seq, feature)
        self.init_parameter(self.rnn)
        self.rnn_init_linear = nn.Sequential(
            nn.Linear(self.encoder_output_dim, self.cfg.train.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.hidden_dropout)
        )

        self.linear = nn.Linear(self.cfg.train.hidden_dim, self.encoder_output_dim)
        self.layer_norm = nn.LayerNorm(self.encoder_output_dim)

        self.dropout = nn.Dropout(self.hidden_dropout)

        # Measure
        self.distance_metric = self.cfg.train.distance_metric
        if self.distance_metric == 'cosine':
            self.metric = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)
        elif self.distance_metric == "euclidean":
            self.metric = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)

        if self.cfg.train.rescale:
            # Define a loss for each slot-type to rescale their associated slot-value (unbalanced classes)
            weights = self.get_rescaling_weights()
            weights = [torch.Tensor(weights[i]).cuda() for i in range(self.num_slots)]
            self.type_loss = [torch.nn.CrossEntropyLoss(weight=weights[i], ignore_index=-1)
                              for i in range(self.num_slots)]
        else:
            self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

        self.train_acc = Accuracy()
        self.val_joint_acc = JointAccuracy(self.slot_encoding, eval=False)
        self.val_acc = Accuracy()
        self.test_joint_acc = JointAccuracy(self.slot_encoding, eval=True)

    @staticmethod
    def init_parameter(module):
        torch.nn.init.xavier_normal_(module.weight_ih_l0)
        torch.nn.init.xavier_normal_(module.weight_hh_l0)
        torch.nn.init.constant_(module.bias_ih_l0, 0.0)
        torch.nn.init.constant_(module.bias_hh_l0, 0.0)

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    @property
    def optimizer_grouped_parameters(self):
        param_optimizer = [(n, p) for n, p in self.named_parameters() if p.requires_grad]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
             'lr': self.hparams.learning_rate},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': self.hparams.learning_rate},
        ]
        return optimizer_grouped_parameters

    def dump_embeddings(self, save_dir):
        type_emb = {}
        for i in range(self.type_lookup.weight.shape[0]):
            label = self.slot_encoding.get_type_name(i)
            type_emb[label] = self.type_lookup.weight[i].cpu().detach().tolist()[:20]

        value_emb = {}
        for i in range(len(self.value_lookup)):
            type_label = self.slot_encoding.get_type_name(i)
            value_emb[type_label] = {}
            for j in range(self.value_lookup[i].weight.shape[0]):
                value_label = self.slot_encoding.get_value_name(i, j)
                value_emb[type_label][value_label] = self.value_lookup[i].weight[j].cpu().detach().tolist()[:20]

        with open(Path(save_dir) / 'type_embeddings.pkl', 'wb') as f:
            dump(type_emb, f)
            print(f"(!) type embeddings dumped at {Path(save_dir) / 'type_embeddings.pkl'}")
        with open(Path(save_dir) / 'value_embeddings.pkl', 'wb') as f:
            dump(value_emb, f)
            print(f"(!) value embeddings dumped at {Path(save_dir) / 'value_embeddings.pkl'}")

    def get_rescaling_weights(self) -> List[List[int]]:
        train_path = Path(self.cfg.data.dir) / f"train_{self.cfg.train.data_subset}.tsv"
        target_slot_types = self.slot_encoding.get_target_slots()
        data = pd.read_csv(train_path, sep='\t', encoding='utf-8', usecols=target_slot_types)

        weights = []
        num_turns = len(data.index)
        for type_idx, type_name in enumerate(target_slot_types):
            value_counts = data[type_name].value_counts()
            num_values = self.slot_encoding.get_num_values(type_name)
            ordered_value_names = [self.slot_encoding.get_value_name(type_idx, val_idx)
                                   for val_idx in range(num_values)]
            # fixme: a lot of values from the ontology are not in the training set
            value_counts = [1 if val_name not in value_counts else int(value_counts.loc[val_name])
                            for val_name in ordered_value_names]
            type_weights = np.array([num_turns / num_values] * num_values) / np.array(value_counts)
            weights.append(type_weights)

        return weights

    def update_ontology(self, ontology_name: str):
        self.ontology.extend(ontology_name)
        self.update_embeddings_lookup()
        self.num_slots = self.ontology.get_num_types()
        self.test_joint_acc.num_types = self.ontology.get_num_types()
        self.new_domains.extend(self.ontology.get_new_domains())

    def tokenized2embeddings(self, tokenized, i=0):
        output = self.sv_encoder(
            input_ids=tokenized['input_ids'][i:, :].to(self.device),
            token_type_ids=tokenized['token_type_ids'][i:, :].to(self.device),
            attention_mask=tokenized['attention_mask'][i:, :].to(self.device)
        )
        embeddings = output.last_hidden_state
        # get the output of the [CLS] token
        embeddings = embeddings[:, 0, :].detach()
        return embeddings

    def init_embeddings_lookup(self):
        self.sv_encoder.eval()

        tokenized = self.ontology.get_tokenized()

        if not self.domain_name_type_queries and not self.domain_name_type_attn:
            type_embeddings = self.tokenized2embeddings(tokenized.types)
            self.type_lookup = nn.Embedding.from_pretrained(type_embeddings, freeze=True)
        else:
            name_embeddings = self.tokenized2embeddings(tokenized.names)
            domain_embeddings = self.tokenized2embeddings(tokenized.domains)
            value_type_embeddings = self.tokenized2embeddings(tokenized.value_types)
            self.name_lookup = nn.Embedding.from_pretrained(name_embeddings, freeze=True)
            self.domain_lookup = nn.Embedding.from_pretrained(domain_embeddings, freeze=True)
            self.value_type_lookup = nn.Embedding.from_pretrained(value_type_embeddings, freeze=True)

        for tk_values in tokenized.values:
            value_embeddings = self.tokenized2embeddings(tk_values)
            self.value_lookup.append(nn.Embedding.from_pretrained(value_embeddings, freeze=True))

    def update_embeddings_lookup(self):
        self.sv_encoder.eval()

        tokenized = self.ontology.get_tokenized()

        get_domain_name_type_emb = self.domain_name_type_queries or self.domain_name_type_attn
        for elt in (['type'] if not get_domain_name_type_emb else ['name', 'domain', 'value_type']):
            # update slot types embeddings
            num_tokenized = len(self.ontology.tokenized_types) if not get_domain_name_type_emb else len(self.ontology.tokenized_names)
            num_embedding = getattr(self, f'{elt}_lookup').weight.shape[0]
            if num_tokenized > num_embedding:
                new_embeddings = self.tokenized2embeddings(getattr(tokenized, f'{elt}s'), num_embedding)
                old_embeddings = getattr(self, f'{elt}_lookup').weight
                lookup = nn.Embedding.from_pretrained(
                    torch.cat((old_embeddings, new_embeddings), dim=0),
                    freeze=True
                )
                setattr(self, f'{elt}_lookup', lookup)

        # update slot values embeddings
        for type_idx, tk_values in enumerate(tokenized.values):
            num_value_tokenized = len(self.ontology.tokenized_values[type_idx])
            num_value_embedding = 0 if type_idx >= num_embedding else self.value_lookup[type_idx].weight.shape[0]
            if num_value_tokenized > num_value_embedding:
                new_value_embeddings = self.tokenized2embeddings(tk_values, num_value_embedding)
                if type_idx >= num_embedding:
                    self.value_lookup.append(nn.Embedding.from_pretrained(new_value_embeddings, freeze=True))
                else:
                    old_value_embeddings = self.value_lookup[type_idx].weight
                    self.value_lookup[type_idx] = nn.Embedding.from_pretrained(
                        torch.cat((old_value_embeddings, new_value_embeddings), dim=0),
                        freeze=True
                    )

    def _get_dist_against_ontology(self, output):
        distances = []
        ds = output.size(1)  # dialog size
        ts = output.size(2)  # turn size
        for type_id in range(self.num_slots):
            # compute distance btw each slot values from the ontology and the model output
            value_embeddings = self.value_lookup[type_id].weight
            num_values = value_embeddings.size(0)

            _hid_label = value_embeddings.unsqueeze(0).unsqueeze(0).repeat(ds, ts, 1, 1).view(ds * ts * num_values, -1)
            _hidden = output[type_id, :, :, :].unsqueeze(2).repeat(1, 1, num_values, 1).view(ds * ts * num_values, -1)
            _dist = self.metric(_hid_label, _hidden).view(ds, ts, num_values)

            if self.distance_metric == "euclidean":
                _dist = -_dist
            distances.append(_dist)
        return distances

    def get_predictions(self, distances) -> List[torch.Tensor]:
        preds = []
        for type_id in range(self.num_slots):
            pred = torch.argmax(distances[type_id], 2, keepdim=True)
            preds.append(pred)
        return preds

    def _get_loss(self, distances, batch: Dict):
        loss = 0
        ds = distances[0].size(0)
        ts = distances[0].size(1)
        for type_id in range(self.num_slots):
            dist = distances[type_id].view(ds*ts, -1)
            targets = batch['value_labels'][:, :, type_id].view(-1)
            if self.cfg.train.rescale:
                type_loss = self.type_loss[type_id](dist, targets)
            else:
                type_loss = self.loss(dist, targets)
            loss += type_loss
        return loss

    def _log_loss(self, loss: torch.Tensor, stage: Text, prog_bar: bool = True):
        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=prog_bar
        )

    def forward(self, batch: Dict, modulate_attn=False):
        ds = batch['input_ids'].size(0)  # dialog size, number of dialogues
        ts = batch['input_ids'].size(1)  # turn size, number of turns in a dialogue (equal to max_turn_length)
        bs = ds * ts    # batch size

        # Encode the input utterances
        max_seq_length = self.cfg.data.max_seq_length
        encoder_output = self.utt_encoder(input_ids=batch['input_ids'].view(-1, max_seq_length),
                                          token_type_ids=batch['token_type_ids'].view(-1, max_seq_length),
                                          attention_mask=batch['attention_mask'].view(-1, max_seq_length))
        hidden = encoder_output.last_hidden_state   # [bs, max_seq_length, bert_hidden_size]

        hidden = torch.mul(hidden, batch['attention_mask'].view(-1, max_seq_length, 1).expand(hidden.size()).float())
        hidden = hidden.repeat(self.num_slots, 1, 1)  # [(num_slots*ds*ts), max_seq_length, bert_hidden_size]

        # Get the encoded slot-types
        if not self.domain_name_type_queries and not self.domain_name_type_attn:
            hid_slot = self.type_lookup.weight
            hid_slot = hid_slot.repeat(1, bs).view(bs * self.num_slots, -1)  # [(num_slots*ds*ts), bert_hidden_size]
            # hid_slot = hid_slot.unsqueeze(dim=1)  # [(num_slots*ds*ts), 1, bert_hidden_size]
        else:
            hid_slot_name = self.name_lookup.weight.repeat(1, bs).view(bs * self.num_slots, -1)
            hid_slot_domain = self.domain_lookup.weight.repeat(1, bs).view(bs * self.num_slots, -1)
            hid_slot_value_type = self.value_type_lookup.weight.repeat(1, bs).view(bs * self.num_slots, -1)
            if self.domain_name_type_queries:
                hid_slot_name = hid_slot_name.unsqueeze(dim=1)
                hid_slot_domain = hid_slot_domain.unsqueeze(dim=1)
                hid_slot_value_type = hid_slot_value_type.unsqueeze(dim=1)
                # concatenate the 3 to get [(num_slots*bs), 3, bert_hidden_size] dim
                hid_slot = torch.cat([hid_slot_name, hid_slot_domain, hid_slot_value_type], dim=1)

        # Apply attention to the utterance vector with the encoded slot-types as query
        """
        nn.MultiheadAttention forward(query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None)
        inputs:
        - query: (L, N, E) (L=target_seq_length, N=batch_size, E=embedding_size)
        - key: (S, N, E) (S=source_seq_length)
        - value: (S, N, E)
        - key_padding_mask: (N, S)
        - attn_mask: (L, S) or (N⋅num_heads,L,S) if 3D mask
        outputs:
        - attn_output: (L, N, E)
        - attn_output_weights: (N, L, S)
        """
        # mask = batch['attention_mask'].view(-1, max_seq_length).repeat(self.num_slots, 1) < 1
        # hidden, _ = self.attention(hid_slot, hidden, hidden, key_padding_mask=mask)  # [num_slots*bs, 1, bert_hidden_size]
        mask = batch['attention_mask'].view(-1, 1, max_seq_length).repeat(self.num_slots, 1, 1)
        if not self.domain_name_type_attn:
            attn_alphas = None
            if modulate_attn:
                attn_alphas = self.domain_attn_modulator.get_alphas(
                    batch['input_ids'],
                    batch['guids'],
                    self.slot_encoding.get_target_slots()
                )
                if self.domain_name_type_queries:
                    attn_alphas = attn_alphas.unsqueeze(dim=1)
                    if self.attn_modulation_cfg.on_domain_name_type:
                        attn_alphas = attn_alphas.repeat(1, 3, 1)
                    else:
                        default_alphas = torch.ones_like(attn_alphas)
                        attn_alphas = torch.cat([default_alphas, attn_alphas, default_alphas], dim=1)
            hidden = self.attention(hid_slot, hidden, hidden, mask=mask, alphas=attn_alphas)    # [num_slots*ds*ts, 1|3, bert_hidden_size]
        else:
            domain_attn_alphas, attn_alphas = None, None
            if modulate_attn:
                domain_attn_alphas = self.domain_attn_modulator.get_alphas(
                    batch['input_ids'],
                    batch['guids'],
                    self.slot_encoding.get_target_slots()
                )
            hidden_domain = self.domain_attn(hid_slot_domain, hidden, hidden, mask=mask, alphas=domain_attn_alphas)
            if self.attn_modulation_cfg.on_domain_name_type:
                attn_alphas = domain_attn_alphas
            hidden_value_type = self.value_type_attn(hid_slot_value_type, hidden, hidden, mask=mask, alphas=attn_alphas)
            hidden_name = self.name_attn(hid_slot_name, hidden, hidden, mask=mask, alphas=attn_alphas)
            hidden = torch.cat([hidden_name, hidden_domain, hidden_value_type], dim=1)

        if not self.domain_name_type_queries and not self.domain_name_type_attn:
            hidden = hidden.squeeze()  # [num_slots*ds*ts, bert_hidden_size]
        else:
            hidden = hidden.view(self.num_slots*bs, 1, self.encoder_output_dim*3)
            hidden = hidden.squeeze()   # [num_slots*ds*ts, bert_hidden_size*3]
            hidden = self.attn_aggregation(hidden)  # [num_slots*ds*ts, bert_hidden_size]
        hidden = hidden.view(self.num_slots, ds, ts, -1).view(-1, ts, self.encoder_output_dim)

        # RNN
        h = hidden[:, 0, :].unsqueeze(0).repeat(self.rnn.num_layers, 1, 1)
        h = self.rnn_init_linear(h)
        rnn_out, _ = self.rnn(hidden, h)
        rnn_out = self.layer_norm(self.linear(self.dropout(rnn_out)))

        output = rnn_out.view(self.num_slots, ds, ts, -1)

        return output

    def training_step(self, batch: Dict, batch_idx: int):
        """
        Minimize the distance bwt outputs and target slot-value's semantic vectors
        """
        output = self(batch)
        distances = self._get_dist_against_ontology(output)
        loss = self._get_loss(distances, batch)
        self._log_loss(loss, 'train')

        # log train accuracy
        preds = self.get_predictions(distances)
        preds = torch.cat(preds, 2)
        self.train_acc(preds, batch['value_labels'])
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)

        # save distances, predictions and targets to debug
        debug = False
        if debug and batch_idx == 0:
            for type_idx, dist in enumerate(distances):
                pd.DataFrame(dist.view(-1, len(self.slot_encoding.int2value[type_idx])).cpu().detach().numpy()).to_csv(
                    Path(self.logger.log_dir) / f'dist_train_batch_{batch_idx}_type_{type_idx}.csv')
            preds_list = preds.view(-1, self.num_slots).cpu().detach().tolist()
            targets_list = batch['value_labels'].view(-1, self.num_slots).cpu().detach().tolist()
            pd.DataFrame(preds_list).to_csv(Path(self.logger.log_dir) / f'preds_int_train_batch_{batch_idx}.csv')
            pd.DataFrame(targets_list).to_csv(Path(self.logger.log_dir) / f'targets_int_train_batch_{batch_idx}.csv')
            true_preds = [
                [self.slot_encoding.int2value[i][p] for ((i, p), l) in zip(enumerate(pred), label) if l != -1]
                for pred, label in zip(preds_list, targets_list)
            ]
            true_targets = [
                [self.slot_encoding.int2value[i][l] for (p, (i, l)) in zip(pred, enumerate(label)) if l != -1]
                for pred, label in zip(preds_list, targets_list)
            ]
            pd.DataFrame(true_preds).to_csv(Path(self.logger.log_dir) / f'preds_train_batch_{batch_idx}.csv')
            pd.DataFrame(true_targets).to_csv(Path(self.logger.log_dir) / f'targets_train_batch_{batch_idx}.csv')

        # if batch_idx == 0 and self.current_epoch == 1:
        #     self.logger.experiment.add_graph(self, batch)

        if self.freeze_from != 0:
            if self.current_epoch == self.freeze_from and batch_idx == 0:
                for param in self.utt_encoder.parameters():
                    param.requires_grad = False

        return loss

    def validation_step(self, batch: Dict, batch_idx: int):
        output = self(batch)
        distances = self._get_dist_against_ontology(output)
        preds = self.get_predictions(distances)

        # log dev joint accuracy
        self.val_joint_acc(preds, batch['value_labels'])
        self.log("dev_joint_acc", self.val_joint_acc, on_step=False, on_epoch=True)
        # log dev accuracy
        self.val_acc(torch.cat(preds, 2), batch['value_labels'])
        self.log("dev_acc", self.val_acc, on_step=False, on_epoch=True)

        # compute loss
        loss = self._get_loss(distances, batch)
        self._log_loss(loss, 'dev')

    def on_test_epoch_start(self) -> None:
        # initiate modulator and the model to predict the domain
        # if no domain model name is given, the modulator will let the attention scores unchanged (defaults)
        if self.modulate_attn:
            self.attn_modulation_cfg.tokenizer_name_or_path = self.cfg.model.name_or_path
            self.domain_attn_modulator = Modulator(
                self.attn_modulation_cfg,
                self.new_domains
            )

    def test_step(self, batch: Dict, batch_idx: int):
        output = self(batch, modulate_attn=self.modulate_attn)
        distances = self._get_dist_against_ontology(output)

        # Accumulate joint accuracy
        preds = self.get_predictions(distances)
        self.test_joint_acc(preds, batch['value_labels'], batch['guids'])

    def test_epoch_end(self, outputs: List[Any]) -> Dict[Text, Any]:
        perf = self.test_joint_acc.compute()
        return perf

    def configure_optimizers(self):
        optimizer = AdamW(self.optimizer_grouped_parameters, lr=self.hparams.learning_rate, correct_bias=False)
        # training_steps = self.num_training_steps
        training_steps = self.hparams.epochs
        warmup_steps = training_steps * 0.1
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=training_steps
        )

        return {'optimizer': optimizer,
                'lr_scheduler': lr_scheduler}
