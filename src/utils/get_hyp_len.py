import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from src.analysis.lexical import get_hyp_len_by_label
import pickle
import configparser
import numpy as np
import argparse

def plot_hist_sent_len(corpus: OrderedDict, labels: [str], plot_dir: str,
                       combine_ents: bool = True):

    len_stats = OrderedDict()

    sns.set()
    sns.set_context("notebook", font_scale=1.1, rc={"lines.linewidth": 1.5})

    for label in labels:
        print(label)
        print(len([v for v in corpus.values() if v['label'] == label]))
        lens = get_hyp_len_by_label(corpus, label)
        sns.kdeplot(lens, label=label, cumulative=False)
        len_stats[label] = {'mean': np.mean(lens),  'median': np.median(lens)}
    plt.xlabel("Number of tokens", weight='bold')
    plt.ylabel("Density", weight='bold')
    plt.legend()
    plt.title("Distribution of Hypothesis Length by Class ({})".format("ents merged" if combine_ents else "ents not merged"), weight='bold')
    plt.savefig(os.path.join(PLOT_DIR, "dist_hyp_len_{}.png".format("ents_merged" if combine_ents else "ents_not_merged")))
    plt.show()

    return len_stats

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Get hypothesis length for analysis')
    parser.add_argument('--config_file', type=str, default='./../../example_cfg.ini', nargs='?')

    config = configparser.ConfigParser()
    config.read(vars(parser.parse_args())['config_file'])
    PLOT_DIR = config.get("paths", "PLOT_DIR")
    SCISPACY_DIR = config.get("paths", "SCISPACY_DIR")
    SPLIT = "all"
    LM_str = "sci_lg"

    if not os.path.isdir(PLOT_DIR):
        os.mkdir(PLOT_DIR)

    res_df = pd.DataFrame()

    for merge_ents in [True, False]:

        corpus_str = os.path.join("..", SCISPACY_DIR, "corpora", "corpus_{}_{}_{}_{}.pkl".\
                format(SPLIT, LM_str, "no_premise", "sep_ents" if not merge_ents else "comb_ents"))

        with open(corpus_str, 'rb') as c:
            corpus = pickle.load(c)

        len_stats = plot_hist_sent_len(corpus=corpus, labels=['entailment', 'neutral', 'contradiction'],
                           plot_dir=PLOT_DIR, combine_ents=merge_ents)

        temp = pd.DataFrame.from_dict(len_stats).reset_index(0)
        temp['ents'] = "separate" if not merge_ents else "merged"

        if res_df.empty:
            res_df = temp
        else:
            res_df = res_df.append(temp)

    print(res_df.round(decimals=1))