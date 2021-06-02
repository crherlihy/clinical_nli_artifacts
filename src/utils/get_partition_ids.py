import os
import pandas as pd
import numpy as np
import argparse
import configparser
import hashlib

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Recover AfLite partition ids for each example')
    parser.add_argument('--config_file', type=str, default='./../../example_cfg.ini', nargs='?')
    parser.add_argument('--embeds_splits', type=str, default='test', nargs='?')
    parser.add_argument('--incl_premise', type=str, default="False", nargs='?')
    parser.add_argument('--combine_ents', type=str, default="True", nargs='?')
    args = parser.parse_args()
    print(os.getcwd())

    config = configparser.ConfigParser()
    config.read(args.config_file)
    DATA_DIR = config.get("paths", "DATA_DIR")
    AFLITE_DIR = os.path.join('..', config.get("paths", "AFLITE_DIR"))
    PARTITIONS_DIR = os.path.join(AFLITE_DIR, "partitions")

    INCL_PREMISE = eval(args.incl_premise)
    COMBINE_ENTS = eval(args.combine_ents)

    LANG_MODEL = config.get("embeddings", "AF_LANG_MODEL")

    lm_lookup = {"en_core_sci_sm": "sci_sm", "en_core_sci_md": "sci_md", "en_core_sci_lg": "sci_lg",
                 "fasttext_wiki_mimic": "ft_wiki_mimic", "clinical_bert": "clinical_bert"}

    LM_name = lm_lookup[LANG_MODEL]
    PREMISE_str = "w_premise" if INCL_PREMISE else "no_premise"
    ENTS_str = "comb_ents" if COMBINE_ENTS else "sep_ents"

    if not os.path.isdir(PARTITIONS_DIR):
        os.mkdir(PARTITIONS_DIR)

    for partition in ["easy", "difficult"]:

        with open(os.path.join(AFLITE_DIR, "mli_all_{}_{}_{}_{}.txt".format(partition, LM_name, PREMISE_str, ENTS_str)), 'r', encoding='utf-8') as f:
            ids = np.array([line.split()[0] for line in f.readlines()])
            np.save(os.path.join(PARTITIONS_DIR, "mednli_{}_{}_{}_{}_example_ids.npy".format(partition, LM_name, PREMISE_str, ENTS_str)), ids)



