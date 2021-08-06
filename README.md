## MedNLI Is Not Immune: Natural Language Inference Artifacts in the Clinical Domain

This repository contains the source code required to reproduce the analysis presented in the paper "MedNLI Is Not Immune: Natural Language Inference Artifacts in the Clinical Domain", appearing at [ACL-IJCNLP 2021](https://aclanthology.org/2021.acl-short.129/).

#### Data:

MedNLI can be downloaded from [PhysioNet](https://physionet.org/content/mednli/1.0.0/), though credentialed access is required.
After you have downloaded the data, put the resulting directory underneath the project root directory. Organization is as follows:

```
.
├── mednli
│   └── 1.0.0
│       ├── LICENSE.txt
│       ├── README.txt
│       ├── SHA256SUMS.txt
│       ├── index.html
│       ├── mli_dev_v1.jsonl
│       ├── mli_test_v1.jsonl
│       └── mli_train_v1.jsonl
```
----
### Set-up:

#### Conda environment:
`conda env create -f environment.yml`

`conda activate clinical_nli` 

#### scispaCy language model:
General usage is: `pip install <Model URL>`; `en_core_sci_sm` and `en_core_sci_lg` are both used in this pipeline:
- `pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_sm-0.4.0.tar.gz`
- `pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_lg-0.4.0.tar.gz`

#### fastText MIMIC-III embeddings:
Referenced in the [original MedNLI paper by Romanov and Shivade (2018)](https://arxiv.org/abs/1808.06752); available on the [associated repo](https://github.com/jgc128/mednli) or via:
- `wget https://mednli.blob.core.windows.net/shared/word_embeddings/wiki_en_mimic.fastText.no_clean.300d.pickled`

#### Configuration file:
`./example_cfg.ini`: Defines paths and task-specific hyper-parameters. 

----

### Shell and python scripts:

From the project root directory: 
`cd ./scripts && sh parse_embeds_aflite.sh`

Note: `parse_embeds_aflite.sh` has 4 boolean flags:
- `fastText`: parse MedNLI input files (JSON) and create fastText-formatted `.txt` files 
- `ftAllSubsets`: create a single fastText-formatted `.txt` file containing instances from all splits (eg, train, dev test). Useful for AFLite.
- `embeddings`: recovers embeddings for each instance in the corpus (language model is configurable)
- `aflite`: runs adversarial filtering algorithm `AfLite` adapted from [Sakaguchi et al. (2019)](https://arxiv.org/abs/1907.10641); yields `easy` and `difficult` partitions

#### To replicate reported results, after running `sh parse_embeds_aflite.sh` with all flags set to `True`, run:

- `sh ft_baseline.sh`: computes fastText baseline results; if `evalAflite` flag is set to `True`, also computes fastText results for AfLite *easy* and *difficult* partitions. 
- `sh lexical.sh`: computes ngram counts, PMI, and mean/median hypothesis length by label.
- `sh semantic.sh`: uses `scispaCy` to link named ents to UMLS; conducts statistical hypothesis testing re: heuristics. 

From the project root directory, `cd ./src/utils` and:
- `python get_hyp_len.py`: Computes hypothesis length for two versions of the corpus (multi-word entities merged and separate).
- `python get_partition_ids.py`: Creates 2 arrays with instance ids for the *easy* and *difficult* AfLite partitions.
  - instance ids will have the format `<split><numeric_id>`
  - underlying text can be recovered by joining against the `./mednli/fastText/mli_all_w_premise_v1_sep.txt` file. 

---- 

If you find this code useful in your research, please consider citing:

```
@inproceedings{herlihy-rudinger-2021-mednli,
    title = "{M}ed{NLI} Is Not Immune: {N}atural Language Inference Artifacts in the Clinical Domain",
    author = "Herlihy, Christine  and
      Rudinger, Rachel",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-short.129",
    doi = "10.18653/v1/2021.acl-short.129",
    pages = "1020--1027",
}
```
