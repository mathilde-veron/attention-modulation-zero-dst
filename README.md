# Attention Modulation for Zero-Shot Cross-Domain Dialogue State Tracking
Companion repository for the paper "Attention Modulation for Zero-Shot Cross-Domain Dialogue State Tracking".

This project contains all the scripts and information needed to reproduce the experiments presented in the paper.

The code consists in the re-implementation of the SUMBT model ([paper](https://aclanthology.org/P19-1546/) and 
[repository](https://github.com/SKTBrain/SUMBT)) with Pytorch Lightning, plus the possibility to train two SUMBT 
variants described in the paper and the possibility to apply attention modulation on a trained model during 
inference.


## Citation

```bibtex
Paper accepted at CODI 2022 COLING Workshop. Waiting for publication.
```

## Installation

1) Create `conda` environment:

```shell
conda create -n sumbt python==3.8
conda activate sumbt
```

2) Install PyTorch (>= 1.7.1) following the instructions of the [docs](https://pytorch.org/get-started/locally/#start-locally)

3) Install dependencies:
```shell
pip install -r requirements.txt
```

## Requirements

### Configuration file

Each experiment needs a YAML configuration file including various paths and hyper-parameters.

`config_template.yml` gives a template for the configuration file.
To run an experiment we suggest you to create a specific experiment directory, copy the template and renamed it 
`config.yml`.

### Data

See instructions and documentation in [the readme specific for data](./data/readme.md).

### Encoder

The utterance encoder we used in the paper is `'bert-base-uncased'` from [huggingface.co](https://huggingface.co/models).
The name has to be set in the configuration file under `model.name_or_path`.
To run an experiment in offline mode you can download the model with `download_model` function defined in 
[utils.py](utils.py).


## Run the experiments

To run an experiment, modify the configuration file according to the variant you want to train:
* To fix BERT weights, set `train.freeze_utt_encoder` to `True` ;
* For the "triple query" variant, set `train.domain_name_type_queries` to `True` ;
* For the "triple attn." variant, set `train.domain_name_type_attn` to `True`.

All experiments were run using random seeds `0`, `100`, `200`, `300` and `400`.
You can set the random seed by modifying `train.seed` in the configuration file.

To train a model for zero-shot experiment set `dataset.path` to the path to the target zero-shot domain subdirectory 
(see [the readme specific for data](./data/readme.md)) and set `train.data_subset` and `ontology_subset` to `known` in 
the configuration file.

To evaluate a trained model on the target zero-shot domain set `eval.subsets` to `['unknown']`.

To evaluate a trained model and make it rely on attention modulation and a domain oracle, do the following changes in 
the configuration file:
```yaml
eval:
  # Params to modulate scores in the domain multi heads attention layer
  attention_modulation:
    # Which model to use for domain prediction
    # Let unfulfilled to not use attention scores modulation. Else oracle
    domain_model: 'oracle'
    # Path to the test file annotated with the domains for the oracle
    domain_references: 'path/to/domain/directory'
```

To run a zero-shot experiment (training + evaluation):
```shell script
python train.py --dir path/to/config/file/directory
```

To only evaluate a trained model :
```shell script
python eval.py --dir path/to/config/file/directory
```


## License

```
MIT License

Copyright (c) 2022 Université Paris-Saclay
Copyright (c) 2022 Laboratoire national de métrologie et d'essais (LNE)
Copyright (c) 2022 CNRS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```