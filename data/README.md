# Data

## MultiWOZ 2.0

Download MultiWOZ 2.0 data at the SUMBT format directly from 
[SUMBT repository](https://github.com/SKTBrain/SUMBT/tree/master/data/multiwoz).
You need especially `train.tsv`, `dev.tsv`, `test.tsv`, and `ontology.json`.

## Leave-one-out zero-shot data

To generate leave-one-out zero-shot data from MultiWOZ 2.0 data at the SUMBT format, run 
[prepare_zero_shot_data.py](../data_processing/prepare_zero_shot_data.py).
It will create a subdirectory for each target domain from the destination directory you gave as argument.
In each sub-directory are generated the following files: `train_<data_subset>.tsv`, `dev_<data_subset>.tsv`, 
`test_<data_subset>.tsv`, and `ontology_<ontology_subset>.tsv`, where `<data_subset>` and `<ontology_subset>` 
correspond to `known` or `unknown`.
`known` denotes data that are not related to the target domain.
`unknown` denotes data that related to the target domain.

For Leave-one-out zero-shot experiments, the model is trained on `train_known.tsv` and `dev_known.tsv` with 
`ontology_known.json` and is evaluated on `test_unknown.tsv` with `ontology_unknown.json`.

To configure a zero-shot experiment set `dataset.path` to the path to the target domain subdirectory in the 
configuration file.

## Information about the domain, name, and type of each slot

`domain_name_type.csv` and `slot_description.csv` files were created from the following work: 
[Leveraging Slot Descriptions for Zero-Shot Cross-Domain Dialogue State Tracking](https://aclanthology.org/2021.naacl-main.448).
This files have to be located in the parent directory of the data seen previously.

## Utterances annotated with the domain

The oracle we used for attention modulation needs for each utterance the associated reference domain.
The references were extracted with [format_domain_data.py](../data_processing/format_domain_data.py) script from 
[MultiWOZ 2.1 data](https://github.com/budzianowski/multiwoz/tree/master/data).
We had to use MultiWOZ 2.1 version because the 2.0 one is not annotated with domain on the user side.

To train sumbt and its variants we still used the 2.0 version because to the best of our knowledge there is no cleaned 
version of the ontology for the 2.1 version where the an entity can only be associated with one slot values with is 
essential for SUMBT variants.