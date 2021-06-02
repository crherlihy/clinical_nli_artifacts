import os
import numpy as np
import pandas as pd
import scispacy
import spacy
from collections import OrderedDict
from scispacy.linking import EntityLinker
import configparser
import argparse
import sys
import requests
import time
from scipy.stats import chisquare

def get_mesh_info_from_unique_id(data_dir:str, split:str, seg:str, save_str:str,chunksize:int=100, ):

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    file_name = os.path.join(data_dir, "linked_ents", "linked_ents_{}_mesh_{}.csv".format(split, seg))

    mesh_df = pd.read_csv(file_name)
    print(len(mesh_df.concept_id.unique()))
    info = OrderedDict()

    while len(info.keys()) < len(mesh_df.concept_id.unique()):

        for chunk_id, chunk in enumerate(chunks(mesh_df.concept_id.unique(), n=chunksize)):
            print(chunk_id)
            for s_id in chunk:
                if s_id not in info:
                    try:
                        response = requests.get('https://id.nlm.nih.gov/mesh/{}.json'.format(s_id))
                        info[s_id] = response.json()
                        response.close()

                    except ConnectionResetError or ConnectionError:
                        time.sleep(20)
                        continue

            if chunk_id % 2 == 0:
                time.sleep(20)


    mesh_df['tree_num'] = mesh_df.apply(lambda row: info[row.concept_id]['treeNumber'],axis=1)
    mesh_df.to_csv(save_str, index=False)

    return mesh_df

def gender_reveal(data_dir:str, split:str, link_to:str ):
    """
    not enough examples to test this
    :param data_dir:
    :param split:
    :param link_to:
    :return:
    """
    pdf = pd.read_csv(os.path.join(data_dir, "linked_ents_{}_{}_premise.csv".format(split, link_to)))
    hdf = pd.read_csv(os.path.join(data_dir, "linked_ents_{}_{}_hypothesis.csv".format(split, link_to)))


    premises = pdf.loc[[(x.split("/")[-1][:3] in ["C12", "C13"]) or
                         (x.split("/")[-1] in ["D27.505.696.875", "M01.975", "C01.635"]) for x in pdf.tree_num], :]

    hypotheses = hdf.loc[[any(word in x.replace("many", "").replace("manic", "").split()  for word in ["male", "female", "man",  "woman", "boy", "girl", "menopause", "menopausal"]) for x
                          in hdf.hypothesis],:]

    out = pd.merge(premises, hypotheses, on=['split', 'example_id', 'label'],  how='outer', indicator=True)
    out[['premise', 'hypothesis', 'split', 'example_id', 'label', 'canonical_name_x', 'tree_num_x', 'tree_num_y', '_merge']].to_csv("../temp.csv")
    return

def hypernym_heuristic(data_dir:str, split:str, link_to:str, heuristics_res_dir:str):

    def identify_hypernyms(prem_nums, hyp_nums):
        for pn in prem_nums:
            res = list(filter(lambda x: pn in x and len(x)>len(pn), hyp_nums))
            if len(res) > 0:
                return True
        return False

    pdf = pd.read_csv(os.path.join(data_dir,"linked_ents_{}_{}_premise.csv".format(split, link_to)))
    hdf = pd.read_csv(os.path.join(data_dir,"linked_ents_{}_{}_hypothesis.csv".format(split, link_to)))

    pdf.tree_num = pdf.tree_num.apply(lambda x: x.replace("http://id.nlm.nih.gov/mesh/", ""))
    hdf.tree_num = hdf.tree_num.apply(lambda x: x.replace("http://id.nlm.nih.gov/mesh/", ""))

    prem_nums = pdf.groupby(['split', 'example_id', 'label'])['tree_num'].unique().reset_index()
    hyp_nums = hdf.groupby(['split', 'example_id', 'label'])['tree_num'].unique().reset_index()

    print(prem_nums.shape)
    print(hyp_nums.shape)
    example_nums = pd.merge(prem_nums, hyp_nums, on=['split', 'example_id', 'label'])
    print(example_nums.shape)
    example_nums.columns = ['split', 'example_id', 'label', 'prem_nums', 'hyp_nums']

    hypernyms = example_nums.loc[example_nums.apply(lambda row: identify_hypernyms(row['prem_nums'], row['hyp_nums']), axis=1), :].reset_index(0)
    print(hypernyms.shape)
    print(hypernyms.groupby(['label']).count())
    hypernyms.to_csv(os.path.join(heuristics_res_dir, "hypernym_heuristic_{}_{}.csv".format(split, link_to)), index=False)

    #print(hypernyms.groupby(['label']).agg('count'))
    return hypernyms

def probable_cause_heuristic(data_dir:str, split:str, link_to:str, heuristics_res_dir:str):

    # "M01.808" = smoking; C25.775 = substance-related disorders; M01.066 = alcoholic
    def identify_causes(hyp_nums:[str], cause_ids:[str] = ["M01.808", "C25.775", "F03", "M01.066","M01.325", "C23.888.144.699"]):
        for hn in hyp_nums:
            res = list(filter(lambda x: hn == x or x in hn, cause_ids))
            if len(res) > 0:
                return True
        return False

    def identify_concepts(prem_nums:[str], concept_ids:[str]= ["C", "D", "E", "F"]):
        for pn in prem_nums:
            res = list(filter(lambda x: pn[0] == x or x in pn, concept_ids))
            if len(res) > 0:
                return True
        return False

    pdf = pd.read_csv(os.path.join(data_dir,"linked_ents_{}_{}_premise.csv".format(split, link_to)))
    hdf = pd.read_csv(os.path.join(data_dir,"linked_ents_{}_{}_hypothesis.csv".format(split, link_to)))

    pdf.tree_num = pdf.tree_num.apply(lambda x: x.replace("http://id.nlm.nih.gov/mesh/", ""))
    hdf.tree_num = hdf.tree_num.apply(lambda x: x.replace("http://id.nlm.nih.gov/mesh/", ""))

    prem_nums = pdf.groupby(['split', 'example_id', 'label'])['tree_num'].unique().reset_index()
    prem_nums = pd.merge(prem_nums,pdf.loc[:, ['split', 'example_id', 'label', 'premise']].drop_duplicates(),
                        how='left', on=['split', 'example_id', 'label'])

    hyp_nums = hdf.groupby(['split', 'example_id', 'label'])['tree_num'].unique().reset_index()
    hyp_nums = pd.merge(hyp_nums, hdf.loc[:,['split', 'example_id', 'label', 'hypothesis']].drop_duplicates(), how='left', on=['split', 'example_id', 'label'])

    example_nums = pd.merge(prem_nums, hyp_nums, on=['split', 'example_id', 'label'])
    example_nums.columns = ['split', 'example_id', 'label', 'prem_nums', 'premise', 'hyp_nums', 'hypothesis']

    causes = example_nums.loc[((example_nums.apply(lambda row: identify_causes(row['hyp_nums']),axis=1)) & (example_nums.apply(lambda row: identify_concepts(row['prem_nums']),axis=1))),:]
    print(causes.groupby(['label']).agg('count'))
    return causes

def negation_heuristic(data_dir:str, split:str, link_to:str, heuristics_res_dir:str):

    def identify_repetition(prem_nums, hyp_nums):
        for pn in prem_nums:
            res = list(filter(lambda x: pn == x and pn != "M01.380.600", np.ravel(hyp_nums)))
            if len(res) > 0:
                return True
        return False

    def identify_negation(hypothesis):
        return any(word in hypothesis for word in ["does not have", "no finding of", "denies"]) or any(word in hypothesis.split() for word in ["no"])

    def identify_healthy(hypothesis):
        return any(word in hypothesis.split() for word in ["normal", "healthy", "discharged"])


    pdf = pd.read_csv(os.path.join(data_dir,"linked_ents_{}_{}_premise.csv".format(split, link_to)))
    hdf = pd.read_csv(os.path.join(data_dir,"linked_ents_{}_{}_hypothesis.csv".format(split, link_to)))

    pdf.tree_num = pdf.tree_num.apply(lambda x: x.replace("http://id.nlm.nih.gov/mesh/", ""))
    hdf.tree_num = hdf.tree_num.apply(lambda x: x.replace("http://id.nlm.nih.gov/mesh/", ""))

    prem_nums = pdf.groupby(['split', 'example_id', 'label'])['tree_num'].unique().reset_index()
    hyp_nums = hdf.groupby(['split', 'example_id', 'label'])['tree_num'].unique().reset_index()
    hyp_nums = pd.merge(hyp_nums, hdf.loc[:,['split', 'example_id', 'label', 'hypothesis']].drop_duplicates(), how='left', on=['split', 'example_id', 'label'])

    example_nums = pd.merge(prem_nums, hyp_nums, on=['split', 'example_id', 'label'])
    example_nums.columns = ['split', 'example_id', 'label', 'prem_nums', 'hyp_nums', 'hypothesis']

    negations = example_nums.loc[example_nums.apply(lambda row: identify_repetition(row['prem_nums'], row['hyp_nums']), axis=1), :].reset_index(0)
    negations = negations.loc[negations.apply(lambda row: identify_negation(row['hypothesis']) or identify_healthy(row['hypothesis']), axis=1),]
    negations.to_csv(os.path.join(heuristics_res_dir, "negation_heuristic_{}_{}.csv".format(split, link_to)), index=False)

    print(negations.groupby(['label']).agg('count'))
    return negations

def chisquare_test(heuristic_df:pd.DataFrame, heuristic_name:str):

    def stars(pval: float):
        if pval < 0.001:
            return '***'
        elif pval < 0.01:
            return '**'
        elif pval < 0.05:
            return '*'
        else:
            return ''

    f_obs = heuristic_df.groupby(['label']).agg('size').reset_index(0)
    f_obs.columns = ['label', 'counts']

    print(f_obs.head())

    # default for f_exp is that the categories are assumed to be equally likely.
    x, pval = chisquare(f_obs=f_obs.counts)
    top_class = f_obs.loc[np.argmax(f_obs.counts), 'label']
    pct = '{:.1%}'.format(np.round(np.max(f_obs.counts)/np.sum(f_obs.counts),3))

    return {'heuristic':heuristic_name, 'test statistic':round(x,2), 'p-value':'{:0.2e}'.format(pval) + stars(pval),
            'most frequent class':"{} ({})".format(top_class, pct)}

def make_chisquare_table(h_dfs:[pd.DataFrame], h_names:[str]):

    cdf = pd.DataFrame(columns = ['heuristic', 'test statistic', 'p-value', 'most frequent class'])

    for (df, name) in zip(h_dfs, h_names):
        res = chisquare_test(df, name)
        cdf = cdf.append(pd.DataFrame.from_dict(res,orient='index').T)

    print(cdf.to_latex(column_format=('cccc'), index=False))

    return cdf

def create_ents_file_all_splits(ents_dir:str, link_to:str, seg: str, splits:[str] =['train', 'dev', 'test']):

    all_df = pd.DataFrame(columns=['concept_id','canonical_name','aliases','types','definition','split','example_id','label',"{}".format(seg.lower()),"tree_num"])

    for s in splits:
        split_df = pd.read_csv(os.path.join(ents_dir, "linked_ents_{}_{}_{}.csv".format(s, link_to, seg)))
        all_df = all_df.append(split_df)

    all_df.to_csv(os.path.join(ents_dir, "linked_ents_all_{}_{}.csv".format(link_to, seg)), index=False)
    return all_df

def get_linked_entities(scispacy_dir:str, examples: [str], link_to: str, max_entities_per_mention:int,
                        lang_model: str = "en_core_sci_sm",split:str='train',segment:str ='hypothesis'):
    nlp = spacy.load(lang_model)
    linked_ents = OrderedDict()
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": link_to})

    all_linked_ents = []
    linker = nlp.get_pipe("scispacy_linker")

    for i, example in enumerate(examples):
        # if segment == "premise":
        #     print(i,example, " ".join(example.split()[2:]))
        #seg = example if segment == "premise" else " ".join(example.split()[2:])
        seg = " ".join(example.split()[2:]) if segment == "premise" else "".join(example.split("ñ")[1].replace("\n", ""))
        label = example.split()[1].replace("__label__", "")
        linked_ents[i] = {'doc': None, 'n_ents': 0, 'label': label}

        try:
            doc = nlp(seg)
            linked_ents[i]['doc'] = doc

            if doc.ents is not None:
                for ent in doc.ents:
                    for linked_ent in ent._.kb_ents:
                        temp = pd.Series(linker.kb.cui_to_entity[linked_ent[0]])
                        temp['split'] = split
                        temp['example_id'] = i
                        temp['label'] = label
                        temp[segment] = seg

                        all_linked_ents.append(temp)
                        linked_ents[i]['n_ents'] +=1

        except:
            print("Unexpected error:", seg, sys.exc_info())
            linked_ents[i] = {'doc': None, 'n_ents': [-np.inf], 'label': label}
            continue

    ents_df = pd.DataFrame(all_linked_ents)
    ents_df.columns = ["concept_id", "canonical_name", "aliases", "types", "definition", "split", "example_id", "label", segment]

    file_name = os.path.join(scispacy_dir, "linked_ents", 'linked_ents_{}_{}_{}.csv'.format(split, link_to, segment))
    ents_df.to_csv(file_name,index=False)

    return linked_ents, ents_df

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Recover scispaCy embeddings for each sentence in the training data')
    parser.add_argument('--config_file', type=str, default='./../example_cfg.ini', nargs='?')
    parser.add_argument('--splits', type=str, default='all', nargs='?')
    parser.add_argument('--segment', type=str, default="hypothesis", nargs='?')
    parser.add_argument('--combine_ents', type=str, default="False", nargs='?')
    parser.add_argument('--link_to', type=str, default="mesh", nargs='?')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_file)
    DATA_DIR = config.get("paths", "DATA_DIR")
    SCISPACY_DIR = config.get("paths", "SCISPACY_DIR")
    FT_DIR = config.get("paths", "FT_DIR")
    HEURISTICS_DIR = config.get('paths', "HEURISTICS_DIR")

    for dir_name in [HEURISTICS_DIR]:
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

    SPLIT = args.splits

    SEGMENT = args.segment
    COMBINE_ENTS = eval(args.combine_ents)
    PREMISE_str = "w_premise" if SEGMENT == "premise" else "no_premise"
    SEP_str = "_sep"
    ENTS_str = "comb_ents" if COMBINE_ENTS else "sep_ents"

    LANG_MODEL = config.get("embeddings", "SEMANTIC_LANG_MODEL")
    LINK_TO = args.link_to
    LM_name = "sci_lg" if LANG_MODEL == "en_core_sci_lg" else "sci_sm"

    NGRAM_MAX = eval(config.get("lexical", "NGRAM_MAX"))
    MIN_DF = eval(config.get("lexical", "MIN_DF"))
    SMOOTHING = eval(config.get("lexical", "SMOOTHING"))

    print("split: {} | segment : {} | combine ents?  {}".format(SPLIT, SEGMENT, COMBINE_ENTS))

    file_name = os.path.join(FT_DIR, "mli_{}_{}_v1{}.txt".format(SPLIT, PREMISE_str, SEP_str))
    #
    # lines = [" ".join(line.split()[0:]).split("ñ")[0][:-1] for line in open(file_name).readlines()] if SEGMENT == "premise" \
    #     else [line for line in open(file_name).readlines()]

    for d in [os.path.join(SCISPACY_DIR, "linked_ents")]:
        if not os.path.isdir(d):
            os.mkdir(d)

    if SPLIT == "all":
        for seg_type in ["premise", "hypothesis"]:

            if not os.path.exists(os.path.join(SCISPACY_DIR, "linked_ents", 'linked_ents_{}_{}_{}.csv'.format(SPLIT, LINK_TO, seg_type))):
                lines = [" ".join(line.split()[0:]).split("ñ")[0][:-1] for line in open(file_name).readlines()] \
                    if seg_type == "premise" else [line for line in open(file_name).readlines()]
                print(seg_type, lines[0:5])
                #lines = lines[0:100]
                _, seg_ents_df = get_linked_entities(scispacy_dir=SCISPACY_DIR, examples=lines, link_to=LINK_TO,
                                                 lang_model=LANG_MODEL,
                                                 max_entities_per_mention=1,
                                                 split=SPLIT,
                                                 segment=seg_type)
            else:
                print('goes here')

                seg_ents_df = pd.read_csv(os.path.join(SCISPACY_DIR, "linked_ents",
                                                   'linked_ents_{}_{}_{}.csv'.format(SPLIT, LINK_TO, seg_type)))

            if 'tree_num' not in seg_ents_df.columns:
                seg_ent_df = get_mesh_info_from_unique_id(data_dir=SCISPACY_DIR, split=SPLIT, seg=seg_type ,
                                             save_str=os.path.join(SCISPACY_DIR, "linked_ents",
                                                                   'linked_ents_{}_{}_{}.csv'.format(SPLIT, LINK_TO,
                                                                                                     seg_type)),chunksize=50)


        ents_df = create_ents_file_all_splits(ents_dir=os.path.join(SCISPACY_DIR, "linked_ents"), link_to=LINK_TO,
                                          seg=seg_type, splits=['all'])

    else:
        if os.path.exists(os.path.join(SCISPACY_DIR, "linked_ents", 'linked_ents_{}_{}_{}.csv'.format(SPLIT, LINK_TO, SEGMENT))):
            ents_df = pd.read_csv(os.path.join(SCISPACY_DIR, "linked_ents", 'linked_ents_{}_{}_{}.csv'.format(SPLIT, LINK_TO, SEGMENT)))

    # elif SPLIT == "all":
    #     print('split all, goes here')
    #     # for the heuristics, we need both of these dataframes to exist, regardless of the current segment arg
    #     for seg_type in ["premise", "hypothesis"]:
    #         if not os.path.exists(os.path.join(SCISPACY_DIR, "linked_ents", 'linked_ents_{}_{}_{}.csv'.format(SPLIT, LINK_TO, seg_type))):
    #
    #             print(os.path.join(SCISPACY_DIR, "linked_ents", 'linked_ents_{}_{}_{}.csv'.format(SPLIT, LINK_TO, seg_type)))
    #             lines =  [" ".join(line.split()[0:]).split("ñ")[0][:-1] for line in open(file_name).readlines()] if seg_type == "premise" else [line for line in open(file_name).readlines()]
    #             print(seg_type, lines[0:5])
    #
    #             #lines = [line for line in open(file_name).readlines()]
    #                 # [" ".join(line.split()[0:]).split("ñ")[0][:-1] for line in
    #                 #      open(file_name).readlines()] if seg_type == "premise" else [line for line in open(file_name).readlines()]
    #             _, seg_ent_df = get_linked_entities(scispacy_dir=SCISPACY_DIR, examples=lines, link_to=LINK_TO,
    #                                              lang_model=LANG_MODEL,
    #                                              max_entities_per_mention=1,
    #                                              split=SPLIT,
    #                                              segment=seg_type)
    #
    #         print(seg_type, seg_ent_df.head())
    #     ents_df = create_ents_file_all_splits(ents_dir=os.path.join(SCISPACY_DIR, "linked_ents"), link_to=LINK_TO,
    #                                           seg=seg_type, splits=['all'])
    #
    #     ents_df = get_mesh_info_from_unique_id(data_dir=SCISPACY_DIR, split=SPLIT, seg=SEGMENT,
    #                                  save_str=os.path.join(SCISPACY_DIR, "linked_ents",
    #                                                        'linked_ents_{}_{}_{}.csv'.format(SPLIT, LINK_TO, SEGMENT)),
    #                                  chunksize=50)
    #
    #     print('after prem and hyp, ents_df', ents_df.head())
        #ents_df = create_ents_file_all_splits(ents_dir= os.path.join(SCISPACY_DIR, "linked_ents"), link_to = LINK_TO, seg=seg_type, splits= ['train', 'dev', 'test'])

        else:
            lines = [" ".join(line.split()[0:]).split("ñ")[0][:-1] for line in
                 open(file_name).readlines()] if SEGMENT == "premise" else [line for line in open(file_name).readlines()]
            _, ents_df = get_linked_entities(scispacy_dir=SCISPACY_DIR, examples=lines, link_to=LINK_TO,
                                         lang_model=LANG_MODEL,
                                         max_entities_per_mention=1,
                                         split=SPLIT,
                                         segment=SEGMENT)

    if 'tree_num' not in ents_df.columns:
        print('goes here tree num')
        ents_df = get_mesh_info_from_unique_id(data_dir=SCISPACY_DIR, split=SPLIT, seg=SEGMENT,
                                     save_str=os.path.join(SCISPACY_DIR, "linked_ents", 'linked_ents_{}_{}_{}.csv'.format(SPLIT, LINK_TO, SEGMENT)),
                                     chunksize=50)

    #print(ents_df.shape)
    #print(ents_df.head())

    #gender_reveal(os.path.join(SCISPACY_DIR, "linked_ents"),  split=SPLIT, link_to=LINK_TO)

    hyp_df = hypernym_heuristic(os.path.join(SCISPACY_DIR, "linked_ents"), split=SPLIT, link_to=LINK_TO, heuristics_res_dir=HEURISTICS_DIR)

    print('hyp df', hyp_df)

    pcdf = probable_cause_heuristic(os.path.join(SCISPACY_DIR, "linked_ents"), split=SPLIT, link_to=LINK_TO, heuristics_res_dir=HEURISTICS_DIR)

    print(pcdf)

    neg_df = negation_heuristic(os.path.join(SCISPACY_DIR, "linked_ents"), split=SPLIT, link_to=LINK_TO,
                        heuristics_res_dir=HEURISTICS_DIR)
    cdf = make_chisquare_table([hyp_df, pcdf], ['hypernym', 'probable cause', 'everything\'s fine'])
    cdf = make_chisquare_table([hyp_df,pcdf, neg_df], ['hypernym', 'probable cause', 'everything\'s fine'])


    # TODO: there  is a bug/dupe, somehow the prem ids and the hyp ids are the same. fix this and the problem should be fixed