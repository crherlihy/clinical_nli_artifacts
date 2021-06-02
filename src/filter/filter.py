import numpy as np
import os
import configparser
import jsonlines
# from src.utils.parse_input_data import create_nli_file
# from src.analysis.lexical import create_doc, create_corpus
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import argparse
import pickle


def aflite(X: np.array, y: np.array, save_dir: str, lm_name: str, prem_str: str, ent_str: str, n: int, m: int, k: int,
           tau: float):
    """
    implements: https://arxiv.org/abs/1907.10641 (Sakaguchi et al., 2019)
    :param X: pre-computed embeddings
    :param y: labels
    :param n: size of the ensemble
    :param m: training size for ensemble
    :param k: filtering cutoff
    :param tau: filtering threshold
    :return:
    """

    corpus_lookup = OrderedDict()
    print(np.mean(X, axis=1).shape)

    # We need a way to recover the text and label associated with each embedding
    for i, example in enumerate(np.mean(X, axis=1)):
        corpus_lookup[example] = i

    # D' = D
    X_filter = X
    y_filter = y

    # while |D'| > m do:
    while X_filter.shape[0] > m:

        # filtering phase

        E = OrderedDict()
        # for all e \in D' do
        # initialize the ensemble predictions E(e) = \emptyset
        for example in np.mean(X_filter, axis=1):
            E[example] = list()

        # for iteration 1:n do:
        for iteration in range(n):
            # random partition (T_i, V_i) of D' s.t. |T_i| = m
            X_train, X_test, y_train, y_test = train_test_split(X_filter, y_filter, train_size=m)

            # Train a linear classifier L on T_i
            clf = LogisticRegression(random_state=0, max_iter=5000).fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # forall e = (x, y) ∈ V_i do:
            for j, test_example in enumerate(np.mean(X_test, axis=1)):
                # Add L(x) to E(e)
                E[test_example].append(y_pred[j])

        # forall e = (x, y) ∈ D' do:
        scores = OrderedDict()
        for i, example in enumerate(np.mean(X_filter, axis=1)):
            key = example
            y_true = y_filter[i]
            # score(e) = |{p \in E(e) s.t. p = y}| / |E(e)|
            if len(E[key]) > 0:
                scores[i] = len([x for x in filter(lambda x: x == y_true, E[key])]) / len(E[key])
            else:
                scores[i] = 0

        # Select the top-k elements S \in D' s.t. score(e) >= tau
        S_ids = [k for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True) if v >= tau][:k]

        # D' = D' \ S
        X_filter = np.array([e for i, e in enumerate(X_filter) if i not in S_ids])
        y_filter = np.array([y for i, y in enumerate(y_filter) if i not in S_ids])

        print("X filter shape ", X_filter.shape)

        # if |S| < k then break
        if len(S_ids) < k or X_filter.shape[0] < m:
            with open(os.path.join(save_dir,
                                   'aflite_all_diff_ids_{}_{}_{}_{}_{}_{}_{}.npy'.format(lm_name, prem_str, ent_str, n,
                                                                                         m, k, tau)), 'wb') as f:
                np.save(f, np.array([corpus_lookup[e] for i, e in enumerate(np.mean(X_filter, axis=1))]))

            return X_filter, y_filter, corpus_lookup

    with open(os.path.join(save_dir,
                           'aflite_all_diff_ids_{}_{}_{}_{}_{}_{}_{}.npy'.format(lm_name, prem_str, ent_str, n, m, k,
                                                                                 tau)), 'wb') as f:
        np.save(f, np.array([corpus_lookup[e] for i, e in enumerate(np.mean(X_filter, axis=1))]))

    return X_filter, y_filter, corpus_lookup


def write_ft_file_from_corpus(corpus: OrderedDict, new_file: str, split: str, prem_str: str):
    ft_file = open(new_file, "w")

    for k, v in corpus.items():
        if prem_str == "no_premise":
            new_line = '{}{} __label__{} {}\n'.format(v['subset'], v['subset_id'], v['label'], v['doc'])
        else:
            new_line = '{}{} __label__{} {}\n'.format(v['subset'], v['subset_id'], v['label'], v['doc'])
        ft_file.write(new_line)

    ft_file.close()
    return


def partition_corpus(corpus: OrderedDict, X_filter: np.array, corpus_lookup: OrderedDict, aflite_dir: str, lm: str,
                     prem: str, ents: str, split: str):
    doc_ids = [corpus_lookup[np.mean(example)] for i, example in enumerate(X_filter)]

    easy_subset = OrderedDict()
    difficult_subset = OrderedDict()

    easy_file = os.path.join(aflite_dir, "mli_{}_easy_{}_{}_{}.txt".format(split, lm, prem, ents))
    difficult_file = os.path.join(aflite_dir, "mli_{}_difficult_{}_{}_{}.txt".format(split, lm, prem, ents))

    for k, v in corpus.items():
        if k in doc_ids:
            difficult_subset[k] = v
        else:
            easy_subset[k] = v

    for pair in zip([easy_subset, difficult_subset], [easy_file, difficult_file]):
        write_ft_file_from_corpus(corpus=pair[0], new_file=pair[1], split=split, prem_str=prem)

    return


def prepare_for_pretrained_model(model_file_dir: str, aflite_dir: str, lm: str, ents: str, prem: str,
                                 separator: str = "ñ"):
    all_file = open(os.path.join(model_file_dir, "mli_all_full_{}_{}_{}.txt".format(lm, prem, ents)), 'w')

    for partition in ["easy", "difficult"]:
        partition_file = open(os.path.join(model_file_dir, "mli_all_{}_{}_{}_{}.txt".format(partition, lm, prem, ents)),
                              'w')
        aflite_file = open(os.path.join(aflite_dir, "mli_all_{}_{}_{}_{}.txt".format(partition, lm, prem, ents)), 'r')

        for line in aflite_file.readlines():

            if prem == "w_premise":

                premise = " ".join(line.split(separator)[0].split()[2:])
                hypothesis = line.split(separator)[1]

                partition_file.write("{}\t{}".format(premise, hypothesis))  # " ".join([x for x in line.split()[2:]])))
                all_file.write("{}\t{}".format(premise, hypothesis))  # " ".join([x for x in line.split()[2:]])))

            else:

                partition_file.write(" \t{}\n".format(" ".join(line.split()[2:])))
                all_file.write(" \t{}\n".format(" ".join(line.split()[2:])))

        partition_file.close()
    all_file.close()

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Perform adversarial filtering to partition the full dataset into easy and difficult subsets')
    parser.add_argument('--config_file', type=str, default='./example_cfg.ini', nargs='?')
    parser.add_argument('--embeds_splits', type=str, default='test', nargs='?')
    parser.add_argument('--incl_premise', type=str, default="True", nargs='?')
    parser.add_argument('--combine_ents', type=str, default="False", nargs='?')
    parser.add_argument('--m', type=str, default="5620", nargs='?')
    parser.add_argument('--n', type=str, default="64", nargs='?')
    parser.add_argument('--k', type=str, default="500", nargs='?')
    parser.add_argument('--tau', type=str, default="0.75", nargs='?')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_file)
    DATA_DIR = config.get("paths", "DATA_DIR")
    AFLITE_DIR = config.get("paths", "AFLITE_DIR")
    AFLITE_SAVE_DIR = os.path.join(config.get("paths", "AFLITE_DIR"), "cluster_files")
    INFERSENT_DIR = config.get("paths", "INFERSENT_DIR")

    for dir_name in [AFLITE_DIR, AFLITE_SAVE_DIR, INFERSENT_DIR]:
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

    INCL_PREMISE = eval(args.incl_premise)
    COMBINE_ENTS = eval(args.combine_ents)
    SPLIT = args.embeds_splits
    LANG_MODEL = config.get("embeddings", "AF_LANG_MODEL")

    lm_lookup = {"en_core_sci_sm": "sci_sm", "en_core_sci_md": "sci_md", "en_core_sci_lg": "sci_lg",
                 "fasttext_wiki_mimic": "ft_wiki_mimic", "clinical_bert": "clinical_bert"}

    LM_name = lm_lookup[LANG_MODEL]
    PREMISE_str = "w_premise" if INCL_PREMISE else "no_premise"
    ENTS_str = "comb_ents" if COMBINE_ENTS else "sep_ents"

    print("LM {} prem str {} ents str {}".format(LM_name, PREMISE_str, ENTS_str))

    SCISPACY_CORPORA_DIR = os.path.join(".", config.get('paths', "SCISPACY_DIR"), "corpora")
    corpus_file = os.path.join(SCISPACY_CORPORA_DIR,
                               "corpus_{}_{}_{}_{}.pkl".format(SPLIT, "sci_lg", PREMISE_str, ENTS_str))

    EMBEDS_DIR = os.path.join(config.get("paths", "EMBEDS_DIR"), LM_name)

    embeds_file = os.path.join(EMBEDS_DIR, "embeds_{}_{}_{}_{}.npy".format(SPLIT, LM_name, PREMISE_str, ENTS_str))
    label_file = os.path.join(EMBEDS_DIR, "labels_{}_{}_{}_{}.npy".format(SPLIT, LM_name, PREMISE_str, ENTS_str))

    # AFLite hyperparameters
    m = int(args.m)
    n = int(args.n)
    k = int(args.k)
    tau = float(args.tau)

    embeds = np.load(embeds_file)
    labels = np.load(label_file)

    print("embeds and labels loaded")

    X_filter, y_filter, corpus_lookup = aflite(X=embeds, y=labels, save_dir=AFLITE_SAVE_DIR, lm_name=LM_name,
                                               prem_str=PREMISE_str, ent_str=ENTS_str, m=m, n=n, k=k, tau=tau)
    print(X_filter.shape, y_filter.shape, len(corpus_lookup.items()))

    with open(corpus_file, 'rb') as f:
        full_train_corpus = pickle.load(f)

    partition_corpus(corpus=full_train_corpus, X_filter=X_filter, corpus_lookup=corpus_lookup, aflite_dir=AFLITE_DIR,
                     lm=LM_name, prem=PREMISE_str, ents=ENTS_str, split=SPLIT)

    f.close()

    prepare_for_pretrained_model(model_file_dir=INFERSENT_DIR, aflite_dir=AFLITE_DIR, lm=LM_name, ents=ENTS_str,
                                 prem=PREMISE_str)

    # print("m={}, n = {}, k = {}, tau = {}".format(m,n, k, tau))
