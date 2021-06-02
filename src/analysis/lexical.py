import os
import spacy
import scispacy
import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from math import log2
from scispacy.linking import EntityLinker
import argparse
import configparser


def create_doc(hypothesis: str, nlp, combine_ent_tokens: bool = True):
    doc = nlp(hypothesis)

    if combine_ent_tokens:

        with doc.retokenize() as retokenizer:
            for ent in doc.ents:
                retokenizer.merge(doc[ent.start:ent.end])

        ents = [ent.text for ent in doc.ents]

        new_doc = ""
        for token in doc:
            if token.text in ents:
                token = "_".join(w for w in token.text.split())
            new_doc += " {} ".format(token if type(token) == str else token.text)

        doc = nlp(new_doc)
    return doc


def create_hyp_only_corpus(examples: [str], split: str, scispacy_dir: str, lang_model: str = "en_core_sci_sm",
                           combine_ent_tokens: bool = True,
                           prem: str = "no_premise"):
    nlp = spacy.load(lang_model)
    corpus = OrderedDict()

    lm = "sci_lg" if lang_model == "en_core_sci_lg" else "sci_sm"
    file_name = os.path.join(scispacy_dir, 'corpora/corpus_{}_{}_{}_{}.pkl'.format(split, lm, prem,
                                                                          "comb_ents" if combine_ent_tokens else "sep_ents"))

    for i, example in enumerate(examples):
        hypothesis = " ".join(example.split()[2:])
        label = example.split()[1].replace("__label__", "")
        corpus[i] = {'doc': create_doc(hypothesis, nlp, combine_ent_tokens), 'label': label}

    with open(file_name, 'wb') as f:
        pickle.dump(corpus, f)

    return corpus


def get_linked_entities(examples: [str], link_to: str, max_entities_per_mention: int, scispacy_dir: str,
                        lang_model: str = "en_core_sci_sm", ):
    nlp = spacy.load(lang_model)
    linked_ents = OrderedDict()
    linker = EntityLinker(resolve_abbreviations=True, name=link_to, max_entities_per_mention=max_entities_per_mention)
    nlp.add_pipe(linker)

    for i, example in enumerate(examples):
        hypothesis = " ".join(example.split()[2:])
        label = example.split()[1].replace("__label__", "")
        doc = nlp(hypothesis)
        linked_ents[i] = {'doc': None, 'ents': [], 'label': label}
        linked_ents[i]['doc'] = doc

        for ent in doc.ents:
            for umls_ent in ent._.kb_ents:
                linked_ents[i]['ents'].append(linker.kb.cui_to_entity[umls_ent[0]])

    file_name = os.path.join(scispacy_dir, 'linked_ents_{}.pkl'.format(link_to))

    with open(file_name, 'wb') as f:
        pickle.dump(linked_ents, f)

    return linked_ents


def get_ngram_counts_by_label(corpus: OrderedDict, label: str, ngram_max: int = 2, top_k: int = 50):
    docs = [v['doc'].text for _, v in corpus.items() if v['label'] == label]
    cv = CountVectorizer(ngram_range=(1, ngram_max), stop_words='english', max_df=1.0)
    cv_fit = cv.fit_transform(docs)
    counts = pd.DataFrame(cv_fit.toarray(),
                          columns=cv.get_feature_names()).sum(axis=0)
    print(counts.T.sort_values(ascending=False).head(top_k))
    return counts.T.sort_values(ascending=False)


def get_ngram_counts_all(corpus: OrderedDict, ngram_max: int = 2, min_df: int = 2, max_df: float = 1.0):
    all_docs = [v['doc'].text for _, v in corpus.items()]
    cv = CountVectorizer(ngram_range=(1, ngram_max), stop_words=None, min_df=min_df, max_df=max_df)
    cv_fit = cv.fit_transform(all_docs)
    doc_counts = pd.DataFrame(cv_fit.toarray(),
                              columns=cv.get_feature_names()).sum(axis=0)

    return doc_counts


def get_pmi_by_label(corpus: OrderedDict, full_counts: pd.DataFrame, label: str, ngram_max: int = 1,
                     smoothing: int = 100, min_df: int = 2, max_df: float = 1.0):
    label_docs = [v['doc'].text for _, v in corpus.items() if v['label'] == label]
    prob_label = len(label_docs) / len(corpus.keys())

    cv = CountVectorizer(ngram_range=(1, ngram_max), stop_words='english', min_df=min_df, max_df=max_df)
    cv_fit = cv.fit_transform(label_docs)
    doc_counts = pd.DataFrame(cv_fit.toarray(),
                              columns=cv.get_feature_names()).sum(axis=0).reset_index(0)

    full_counts = full_counts.reset_index(0)

    for count_df in [full_counts, doc_counts]:
        count_df.columns = ["token", "count"]
        count_df["count"] += smoothing
        count_df["probs"] = count_df["count"] / count_df["count"].sum()

    full_counts.columns = ["token", "count_full", "p_word"]
    doc_counts.columns = ["token", "count_label", "p_word_label"]
    joint_df = pd.merge(doc_counts, full_counts, on="token", how="left")

    joint_df["pmi"] = [log2(row["p_word_label"] / (row["p_word"] * prob_label)) for _, row in joint_df.iterrows()]

    # print(joint_df.head(n=5))
    joint_df.sort_values("pmi", ascending=False, inplace=True)

    # print(joint_df.head(n=30))
    return joint_df


def get_hyp_len_by_label(corpus: OrderedDict, label: str):
    hyp_lens = [len(v['doc'].text.split()) for _, v in corpus.items() if v['label'] == label]
    return hyp_lens


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Recover scispaCy embeddings for each sentence in the training data')
    parser.add_argument('--config_file', type=str, default='./../../example_cfg.ini', nargs='?')
    parser.add_argument('--splits', type=str, default='train', nargs='?')
    parser.add_argument('--incl_premise', type=str, default="False", nargs='?')
    parser.add_argument('--combine_ents', type=str, default="True", nargs='?')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_file)
    DATA_DIR = config.get("paths", "DATA_DIR")
    SCISPACY_DIR = config.get("paths", "SCISPACY_DIR")
    FT_DIR = config.get("paths", "FT_DIR")

    INCL_PREMISE = eval(args.incl_premise)
    COMBINE_ENTS = eval(args.combine_ents)
    SPLIT = args.splits
    LANG_MODEL = "en_core_sci_lg"  #for this analysis, we use scispacy

    LM_name = "sci_lg" if LANG_MODEL == "en_core_sci_lg" else "sci_sm"
    PREMISE_str = "w_premise" if INCL_PREMISE else "no_premise"
    ENTS_str = "comb_ents" if COMBINE_ENTS else "sep_ents"

    NGRAM_MAX = eval(config.get("lexical", "NGRAM_MAX"))
    MIN_DF = eval(config.get("lexical", "MIN_DF"))
    SMOOTHING = eval(config.get("lexical", "SMOOTHING"))

    print("split: {} | use premise?: {} | combine ents?  {}".format(SPLIT, INCL_PREMISE, COMBINE_ENTS))

    file_name = os.path.join(SCISPACY_DIR, 'corpus_{}_{}_{}_{}.pkl'.format(SPLIT, LM_name, PREMISE_str, ENTS_str))
    lines = [line for line in open(os.path.join(FT_DIR, "mli_{}_{}_v1_sep.txt".format(SPLIT, PREMISE_str))).readlines()]

    try:
        with open(file_name, 'rb') as f:
            scispacy_corpus = pickle.load(f)
            print("corpus loaded")

    except FileNotFoundError:
        scispacy_corpus = create_hyp_only_corpus(lines, SPLIT, SCISPACY_DIR, "en_core_sci_lg", COMBINE_ENTS)

    all_docs_counts = get_ngram_counts_all(scispacy_corpus, ngram_max=1, min_df=1)
    full_df = pd.DataFrame()

    for label in ['entailment', 'neutral', 'contradiction']:
        joint_df = get_pmi_by_label(scispacy_corpus, all_docs_counts, label=label, ngram_max=NGRAM_MAX, min_df=MIN_DF
                                    , smoothing=SMOOTHING)
        # print(joint_df.head())
        full_df[label] = pd.Series([x for x in joint_df.token[:15]])
        full_df["p_t_{}".format(label[0])] = pd.Series(["{:.2%}".format(x) for x in joint_df.p_word_label[:15]])

    print(full_df.head())
    print(full_df.to_latex(index=False))
    #
    # sns.set()
    # for label in ['entailment', 'neutral', 'contradiction']:
    #     print(label)
    #     print(len([v for v in scispacy_corpus.values() if v['label'] == label]))
    #
    #     #get_ngram_counts_by_label(scispacy_corpus, label, ngram_max=1)
    #     #lens = get_hyp_len_by_label(scispacy_corpus, label)
    #     sns.kdeplot(lens, label=label)
    #     #print(np.min(lens), np.mean(lens), np.median(lens), np.max(lens))
    # plt.legend()
    # plt.title("Distribution of Hypothesis Length by Class ({})".format("ents merged" if combine_ents else "ents not merged"))
    # plt.show()
    # plt.savefig("")