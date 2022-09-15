import random
from pathlib import Path
from typing import Union, Text

import numpy as np
import pandas as pd
import torch
import yaml
from box import Box
from transformers import BertTokenizer, BertModel


class Config:
    """
    Dot-based access to configuration parameters saved in a YAML file.
    """
    def __init__(self, file: Union[Path, Text]):
        """
        Load the parameters from the YAML file.
        If no path are given in the YAML file for bert_checkpoint and seqeval, the corresponding objects will be load
        if used (needs an internet connection).
        """
        # get a Box object from the YAML file
        with open(str(file), 'r') as ymlfile:
            cfg = Box(yaml.safe_load(ymlfile), default_box=True, default_box_attr=None)

        # manually populate the current Config object with the Box object (since Box inheritance fails)
        for key in cfg.keys():
            setattr(self, key, getattr(cfg, key))

        # Correct types in train (ex. lr = 5e-5 is read as string)
        for float_var in ["learning_rate"]:
            val = getattr(self.train, float_var)
            if type(val) != float:
                setattr(self.train, float_var, float(val))

        # Some attributes could not be defined in config.yml
        self.train.num_workers = getattr(self.train, "num_workers", None)
        self.train.seed = getattr(self.train, "seed", None)
        self.train.rescale = getattr(self.train, "rescale", False)
        self.eval.batch_size = getattr(self.eval, "batch_size", self.train.batch_size)

        # Fix seed if specified
        if self.train.seed is not None:
            fix_seed(self.train.seed)

        # Check value of beta if defined
        if self.eval.attention_modulation:
            beta = self.eval.attention_modulation.beta
            if beta:
                assert beta >= 0, "beta should be between 0 and 1 (not included)"
                assert beta < 1, "beta should be between 0 and 1 (not included)"


def download_model(save_dir: Path, name: Text):
    """
    Load pretrained tokenizer and model from huggingface.co and save them locally
    :param save_dir: Path
        Directory where the files are saved.
    :param name: Text
        Name of the BERT checkpoint to download (e.g. 'bert-base-uncased')
        The list of possible names can be found at https://huggingface.co/transformers/pretrained_models.html
    """
    print(f'Creating directory {save_dir} ... ', end='', flush=True)
    save_dir.mkdir(parents=True, exist_ok=False)
    print('OK')

    print(f"Getting model {name} ... ", end='', flush=True)
    # Load pretrained tokenizer and model from huggingface.co
    tokenizer = BertTokenizer.from_pretrained(name)
    model = BertModel.from_pretrained(name)
    print('OK')

    print(f"Saving model {name} to {save_dir} ... ", end='', flush=True)
    # save the tokenizer and the model locally
    tokenizer.save_pretrained(str(save_dir))
    model.save_pretrained(str(save_dir))
    print('OK')


def as_path(path: Union[Path, Text]) -> Path:
    if not isinstance(path, Path):
        return Path(path)
    return path


def load_performance(file: Union[Path, Text]) -> pd.DataFrame:
    """
    Load a performance matrix saved as CSV with column and row headers
    """
    return pd.read_csv(file, sep=',', index_col=0)


def fix_seed(seed: int = 42):
    """
    Set a fixed seed for the experiment.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
