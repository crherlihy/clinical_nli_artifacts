import fasttext
import configparser
import os
from sklearn.metrics import f1_score, accuracy_score, classification_report, precision_score, recall_score, jaccard_score
from sklearn.metrics import confusion_matrix
from collections import OrderedDict, Counter
import numpy as np
import pandas as pd
import argparse
import warnings


def train_test_ft_baseline(data_path: str, train_set: str, test_set: str, prem:str, separator:str="ñ", wNgrams:int=2):
    """
    Trains and evaluates a simple fastText classifier on
    :param data_path: Path to the directory containing MedNLI text files
    :param train_set: Name of the MedNLI dataset that should be used for training (e.g., "train")
    :param test_set:  Name of the MedNLI dataset that should be used for testing (e.g., "dev" or "test")
    :param prem: String indicating whether or not the premise is included (e.g., "w_premise"; "no_premise"). For file name
    :param separator:
    :param wNgrams: Max length of word ngram (defaults to 2, which means we consider unigrams and bigrams)
    :return: Micro F1 scores for the fastText classifier and majority class baseline 
    """

    train_file = os.path.join(data_path, "mli_{}_{}_v1_{}.txt".format(train_set, prem, "sep" if separator != "None" else ""))
    test_file = os.path.join(data_path, "mli_{}_{}_v1_{}.txt".format(test_set, prem, "sep" if separator != "None" else ""))
    test_lines = [" ".join(line.split("__label__")[1].split()[1:]).replace(separator, "") if separator is not None
                  else " ".join(line.split("__label__")[1].split()[1:]) for line in open(test_file)]

    model = fasttext.train_supervised(train_file, wordNgrams=wNgrams,thread=1)

    ytrue = [line.split("__label__")[1].split()[0] for line in open(test_file)]
    ypred_ft = [model.predict(line,k=1)[0][0].replace("__label__", "") for line in test_lines]

    n_samples, precision, recall = model.test(test_file)
    print(n_samples, precision,recall)

    majority_class = Counter(ytrue).most_common(1)[0][0]
    ypred_mc = [majority_class for line in test_lines]

    f1_ft = f1_score(y_true=ytrue, y_pred=ypred_ft, average='micro')
    f1_mc = f1_score(y_true=ytrue, y_pred=ypred_mc, average='micro')

    for yp in zip(["majority class", "fastText"],[ypred_mc, ypred_ft]):

        print(classification_report(y_true=ytrue, y_pred=yp[1], digits=5))
        print(pd.DataFrame(confusion_matrix(y_true=ytrue, y_pred=yp[1], labels=['entailment','neutral', 'contradiction'])).to_latex())

    return f1_mc, f1_ft

def train_test_ft_aflite(data_path: str, aflite_dir: str, train_set: str, prem: str, test_set: str, lm: str, ents:str="comb_ents"):

    train_file = os.path.join(data_path, "mli_{}_{}_v1_sep.txt".format(train_set, prem))

    results = OrderedDict()

    for eval_type in ["easy", "difficult"]:

        test_file = os.path.join(aflite_dir, "mli_all_{}_{}_{}_{}.txt".format(eval_type, lm, prem, ents))
        test_lines = [" ".join(line.split("__label__")[1].split()[1:]) for line in open(test_file) if "".join(filter(str.isalpha,line.split()[0])) == test_set]

        model = fasttext.train_supervised(train_file, wordNgrams=2)
        ytrue = [line.split("__label__")[1].split()[0] for line in open(test_file) if "".join(filter(str.isalpha,line.split()[0])) == test_set]
        ypred_ft = [model.predict(line, k=1)[0][0].replace("__label__", "") for line in test_lines]

        n_samples, precision, recall = model.test(test_file)
        print(n_samples, precision, recall)

        majority_class = Counter(ytrue).most_common(1)[0][0]
        ypred_mc = [majority_class for line in test_lines]

        f1_ft = f1_score(y_true=ytrue, y_pred=ypred_ft, average='micro')
        f1_mc = f1_score(y_true=ytrue, y_pred=ypred_mc, average='micro')

        results[eval_type] = {'f1_ft': f1_ft, 'f1_mc': f1_mc}

    return results


if __name__ == "__main__":
    os.getcwd()

    parser = argparse.ArgumentParser(description='Recover scispaCy embeddings for each sentence in the training data')
    parser.add_argument('--config_file', type=str, default='./../../example_cfg.ini', nargs='?')
    parser.add_argument('--embeds_splits', type=str, default='train', nargs='?', )
    #parser.add_argument('--incl_premise', type=str, default="False", nargs='?', const="False")
    parser.add_argument('--combine_ents', type=str, default="True", nargs='?', )
    parser.add_argument('--incl_premise', '--list', nargs='+', default=["w_premise", "no_premise"])
    parser.add_argument('--eval_aflite', type=str, default="False", nargs='?')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_file)
    DATA_DIR = config.get("paths", "DATA_DIR")
    AFLITE_DIR = config.get('paths', "AFLITE_DIR")
    FT_DIR = config.get('paths',  "FT_DIR")
    INCL_PREMISE = args.incl_premise
    COMBINE_ENTS = eval(args.combine_ents)
    EVAL_AFLITE = eval(args.eval_aflite)

    LANG_MODEL = config.get("embeddings", "LANG_MODEL")

    lm_lookup = {"en_core_sci_sm": "sci_sm", "en_core_sci_md": "sci_md", "en_core_sci_lg": "sci_lg", "fasttext_wiki_mimic": "ft_wiki_mimic",
                 "clinical_bert": "clinical_bert"}

    LM_name = lm_lookup[LANG_MODEL]
    ENTS_str = "comb_ents" if COMBINE_ENTS else "sep_ents"
    WORD_NGRAMS = eval(config.get("fasttext_baseline", "WORD_NGRAMS"))

    results = OrderedDict()
    results["dev"] = {"majority class": {"no_premise": {"full": None, "easy": None, "difficult": None}, "w_premise": {"full": None, "easy": None, "difficult": None}},
                      "fastText": {"no_premise": {"full": None, "easy": None, "difficult": None}, "w_premise": {"full": None, "easy": None, "difficult": None}}}
    results["test"]  = {"majority class": {"no_premise": {"full": None, "easy": None, "difficult": None}, "w_premise": {"full": None, "easy": None, "difficult": None}},
                      "fastText": {"no_premise": {"full": None, "easy": None, "difficult": None}, "w_premise": {"full": None, "easy": None, "difficult": None}}}

    for eval_dataset in ["dev", "test"]:
        for PREMISE_str in args.incl_premise:

            print("eval dataset: {} | premise? {}".format(eval_dataset, PREMISE_str))

            f1_mc, f1_ft = train_test_ft_baseline(data_path=FT_DIR, train_set="train", test_set=eval_dataset, prem=PREMISE_str, separator="ñ", wNgrams=WORD_NGRAMS)

            if EVAL_AFLITE:
                eval_res = train_test_ft_aflite(data_path=FT_DIR, aflite_dir=AFLITE_DIR, train_set="train",
                                                prem=PREMISE_str, test_set=eval_dataset,
                                                lm=LM_name, ents=ENTS_str)

                results[eval_dataset]["majority class"][PREMISE_str]["easy"] = eval_res["easy"]["f1_mc"]
                results[eval_dataset]["majority class"][PREMISE_str]["difficult"] = eval_res["difficult"]["f1_mc"]
                results[eval_dataset]["majority class"][PREMISE_str]["full"] = f1_mc
                results[eval_dataset]["fastText"][PREMISE_str]["easy"] = eval_res["easy"]["f1_ft"]
                results[eval_dataset]["fastText"][PREMISE_str]["difficult"] = eval_res["difficult"]["f1_ft"]
                results[eval_dataset]["fastText"][PREMISE_str]["full"] = f1_ft

                table = pd.DataFrame()

                for k, v in results.items():
                    for k_inner, v_inner in v.items():

                        if table.empty:
                            table = pd.DataFrame.from_dict(v_inner, orient='index')
                            table['model'] = k_inner
                            table['eval dataset'] = k

                        else:
                            temp = pd.DataFrame.from_dict(v_inner, orient='index')
                            temp['model'] = k_inner
                            temp['eval dataset'] = k
                            table = table.append(temp)

                table = table[["model", "eval dataset", "full", "easy", "difficult"]]

                table['full'] = table.full.apply(lambda x: round(x, 2))

                table["easy"] = ["{} ({}{})".format(round(row.easy,2), "+" if row.easy - row.full >= 0 else "", round(row.easy - row.full,2)) for _,row in table.iterrows()]
                table["difficult"] = [
                    "{} ({}{})".format(round(row.difficult,2), "+" if row.difficult- row.full >= 0 else "", round(row.difficult - row.full, 2)) for
                    _, row in table.iterrows()]

                print(table.round(decimals=2))
                print(table.round(decimals=2).dropna().sort_values(['model', 'eval dataset']).to_latex(bold_rows=True))