import numpy as np
import os
import configparser
import spacy
import pickle
from collections import OrderedDict
import argparse
from transformers import AutoTokenizer, AutoModel
import torch
import tqdm

def create_doc(statement: str, nlp, combine_ent_tokens: bool = True):
    """
    Helper function to create a scispacy doc and combine the tokens of named entities (if flag set to True)
    :param statement: The string to create a doc for (can be premise and hypothesis, or just hypothesis)
    :param nlp: scispacy nlp object
    :param combine_ent_tokens: Indicates whether to combine the tokens of named entity spans (separating with "_") or not
    :return: the scispacy doc object
    """
    doc = nlp(statement)

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


def create_scispacy_corpus(corpus_file:str, examples: [str], split: str, lang_model: str = "en_core_sci_sm", combine_ent_tokens: bool = True, use_premise: bool = False, separator:str="ñ"):
    """
    Creates a scispacy corpus
    :param corpus_file: File to use when pickling the resulting corpus
    :param examples: Training or evaluation examples
    :param split: split identifier in {train, dev, test, all}
    :param lang_model: Language model
    :param combine_ent_tokens: Indicates whether to combine the tokens of named entity spans (separating with "_") (T) or not (F)
    :param use_premise: Indicates whether the examples contain the premise (T) or not (F)
    :param separator: Character used to separate premise and hypothesis statements. Defaults to "ñ".
    :return: The corpus object
    """
    nlp = spacy.load(lang_model)
    nlp.vocab[separator].is_stop = True
    corpus = OrderedDict()

    for i, example in enumerate(examples):

        label = example.split()[1].replace("__label__", "")
        id, subset = list(filter(str.isdigit, example.split()[0]))[0], "".join([x for x in filter(str.isalpha, example.split()[0])])

        statement = " ".join(example.split()[2:])
        corpus[i] = {'doc': create_doc(statement, nlp, combine_ent_tokens), 'label': label, 'subset': subset, 'subset_id':i}

    with open(corpus_file, 'wb') as f:
        pickle.dump(corpus, f)

    return corpus

def get_ft_wiki_mimic_embeds(mednli_file:str, embeds_file:str, label_file:str, split: str, lang_model: str = "en_core_sci_lg",
                      combine_ent_tokens: bool=True, use_premise: bool = False, separator:str="ñ", ft_dim:int=300):

    lines = [" ".join(line.split()[2:]) for line in open(mednli_file).readlines()]
    labels = ["".join(line.split()[1]).replace("__label__", "") for line in open(mednli_file).readlines()]

    with open('./../mednli/embeddings/ft_wiki_mimic/wiki_en_mimic.fastText.no_clean.300d.pickled', 'rb') as f:
        token_vecs = pickle.load(f)

    if use_premise:
        prems = [np.mean([token_vecs[token] if token in token_vecs.keys() else np.zeros([ft_dim])
                          for token in "".join(line.split(separator)[0]).split() if not token.isspace()],axis=0) for line in lines]

        hyps = [np.mean([token_vecs[token] if token in token_vecs.keys() else np.zeros([ft_dim])
                          for token in "".join(line.split(separator)[1]).split() if not token.isspace()], axis=0) for
                 line in lines]

        #embeds = [np.concatenate((prems[i], hyps[i])) for i in range(0, len(lines))]
        embeds = [np.sum((prems[i], hyps[i]),axis=0) for i in range(0, len(lines))]
        print(len(embeds), embeds[0].shape)

    else:
        embeds = [np.mean([token_vecs[token] if token in token_vecs.keys() else np.zeros([ft_dim]) for token in line.split() if not token.isspace()],axis=0) for line in lines]

    f.close()

    return embeds, labels

def get_clinical_bert_embeds(mednli_file:str, embeds_file:str, label_file:str, split: str, lang_model: str = "en_core_sci_lg",
                      combine_ent_tokens: bool=True, use_premise: bool = False, separator:str="ñ", bert_dim:int=768):

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    lines = [" ".join(line.split()[2:]) for line in open(mednli_file).readlines()]
    labels = ["".join(line.split()[1]).replace("__label__", "") for line in open(mednli_file).readlines()]

    if use_premise:
        input = ["[CLS] {} [SEP] {}".format("".join(line.split(separator)[0]), "".join(line.split(separator)[1])) for line in lines]

    else:
        input = ["[CLS] {} [SEP]".format(line) for line in lines]

    embeds = np.zeros([len(lines), bert_dim])

    for i,example in tqdm.tqdm(enumerate(input)):
        input_sentence = torch.tensor(tokenizer.encode(example, padding=True, max_length=128)).unsqueeze(0)
        out = model(input_sentence)
        embeds[i,:] = torch.mean(out[0],1).squeeze(0).detach().numpy()

    try:
        np.save(embeds_file, embeds)
        np.save(label_file, labels)
    except:
        raise

    return embeds,labels



def get_corpus_embeds(corpus_file:str, mednli_file:str, embeds_file:str, label_file:str, split: str, lang_model: str = "en_core_sci_lg",
                      combine_ent_tokens: bool=True, use_premise: bool = False, separator:str="ñ", **pretrained):

    lines = [line for line in open(mednli_file).readlines()]
    label_ids = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

    # For scispacy LMs, create a corpus to get the built-in  vector rep. of each example's text (avg. over tokens)
    if lang_model in ["en_core_sci_sm", "en_core_sci_md", "en_core_sci_lg"]:
        corpus = create_scispacy_corpus(corpus_file= corpus_file, examples=lines, split=split, lang_model=lang_model,
                                        combine_ent_tokens=combine_ent_tokens, use_premise=use_premise)
        embeds = np.array([v['doc'].vector for v in corpus.values()])
        embeds = embeds.reshape(len(lines), len((embeds[0])))
        labels = np.array([label_ids[v['label']] for v in corpus.values()])

    elif lang_model == "fasttext_wiki_mimic":

        embeds, labels = get_ft_wiki_mimic_embeds(mednli_file=mednli_file, embeds_file=embeds_file, label_file=label_file,
                                          split=split, combine_ent_tokens=combine_ent_tokens, use_premise=use_premise)

    elif lang_model == "clinical_bert":
        embeds, labels = get_clinical_bert_embeds(mednli_file=mednli_file, embeds_file=embeds_file, label_file=label_file,
                                          split=split, combine_ent_tokens=combine_ent_tokens, use_premise=use_premise)


    try:
        np.save(embeds_file, embeds)
        np.save(label_file, labels)
    except:
        raise

    return embeds, labels


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Recover embeddings for each sentence in the training data')
    parser.add_argument('--config_file', type=str, default='./example_cfg.ini', nargs='?')
    parser.add_argument('--embeds_splits', type=str, default='train', nargs='?',)
    parser.add_argument('--incl_premise', type=str, default="False", nargs='?',)
    parser.add_argument('--combine_ents', type=str, default="False", nargs='?')
    parser.add_argument('--separator', type=str, default="ñ", nargs='?')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_file)
    DATA_DIR = config.get("paths", "DATA_DIR")

    INCL_PREMISE = eval(args.incl_premise)
    COMBINE_ENTS = eval(args.combine_ents)
    SPLIT = args.embeds_splits
    LANG_MODEL = config.get("embeddings", "LANG_MODEL")

    lm_lookup = {"en_core_sci_sm": "sci_sm", "en_core_sci_md": "sci_md", "en_core_sci_lg": "sci_lg", "fasttext_wiki_mimic": "ft_wiki_mimic",
                 "clinical_bert": "clinical_bert"}

    LM_name = lm_lookup[LANG_MODEL]
    PREMISE_str = "w_premise" if INCL_PREMISE else "no_premise"
    ENTS_str = "comb_ents" if COMBINE_ENTS else "sep_ents"
    SEP_str = "sep" if args.separator != "None" else ""

    EMBEDS_DIR = os.path.join(config.get('paths', "EMBEDS_DIR"), LM_name)
    print("EMBEDS DIR IS: ", EMBEDS_DIR)

    dirs = [config.get('paths', "EMBEDS_DIR"), EMBEDS_DIR]

    if "sci" in LM_name:
        SCISPACY_CORPORA_DIR = os.path.join(config.get('paths', "SCISPACY_DIR"), "corpora")
        corpus_file = os.path.join(SCISPACY_CORPORA_DIR, "corpus_{}_{}_{}_{}.pkl".format(SPLIT, LM_name, PREMISE_str, ENTS_str))

        for sdir in [config.get('paths', "SCISPACY_DIR"), SCISPACY_CORPORA_DIR]:
            dirs.extend([sdir])

    elif LM_name in ["ft_wiki_mimic", "clinical_bert"]:
        corpus_file = None

    embeds_file = os.path.join(EMBEDS_DIR, "embeds_{}_{}_{}_{}.npy".format(SPLIT, LM_name, PREMISE_str, ENTS_str))
    label_file = os.path.join(EMBEDS_DIR, "labels_{}_{}_{}_{}.npy".format(SPLIT,LM_name, PREMISE_str, ENTS_str))
    mednli_file = os.path.join(DATA_DIR, "fastText", "mli_{}_{}_v1_{}.txt".format(SPLIT, PREMISE_str, SEP_str))

    for d in dirs:
        if not os.path.exists(d):
            os.mkdir(d)

    embeds, labels = get_corpus_embeds(corpus_file=corpus_file, mednli_file=mednli_file, embeds_file=embeds_file,
                                       label_file=label_file, split=SPLIT, lang_model=LANG_MODEL, combine_ent_tokens=COMBINE_ENTS, use_premise=INCL_PREMISE)



