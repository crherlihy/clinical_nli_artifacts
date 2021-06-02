import os
import jsonlines
import configparser
import argparse


def create_nli_file(split: str, mli_file: str, nli_file: str, lower: bool, incl_premise: bool, separator:str="ñ"):
    """
    Helper function to create supervised train/dev/test files for the fastText hypothesis-only baseline classifier and premise-hypothesis NLI models.
    :param split: split identifier in {train, dev, test}
    :param mli_file: path to the raw MedNLI file
    :param nli_file: output file path
    :param lower: whether or not to cast the premise/hypothesis to lowercase
    :param incl_premise: whether or not to write the premise (if  false, hypothesis-only)
    :param separator: "None" or str to separate premise from hypothesis (useful for tokenization)
    :return: None
    """

    nli_file = open(nli_file, "w")

    with jsonlines.open(mli_file) as f:
        for i,line in enumerate(f.iter()):

            if incl_premise:
                new_line = '{}{} __label__{}  {} {} {}\n'.format(split, i, line['gold_label'], line['sentence1'], separator, line['sentence2'])\
                    if separator is not None else '__label__{}  {} {}\n'.format(line['gold_label'], line['sentence1'], line['sentence2'])
            else:
                new_line = '{}{} __label__{} {}\n'.format(split, i, line['gold_label'], line['sentence2'])

            nli_file. write(new_line.lower() if lower else new_line)

    nli_file.close()
    return


if __name__ == "__main__":
    # Note: To create fastText-style .txt files, you can also run `sh parse_embeds_aflite.sh`; set flag fastText=true
    parser = argparse.ArgumentParser(description='Create fastText format .txt file from a MedNLI train/dev/test JSON file')
    parser.add_argument('--config_file', type=str, default='./example_cfg.ini', nargs='?')
    parser.add_argument('--split', type=str, default='test', nargs='?')
    parser.add_argument('--incl_premise', type=str, default="False", nargs='?')
    parser.add_argument('--separator', type=str, default="ñ", nargs='?')

    # Parse args and declare global variables
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config_file)
    DATA_DIR = config.get('paths', "DATA_DIR")
    INCL_PREMISE = eval(args.incl_premise)
    SPLIT = args.split
    SEP = args.separator

    # Parse the raw MedNLI JSON (train, dev, test) files into __label__{} hypothesis lines for fastText classifier
    MEDNLI_JSON_FILE = os.path.join(".", DATA_DIR, "1.0.0", "mli_{}_v1.jsonl".format(SPLIT))
    #TXT_FILE = os.path.join(".", DATA_DIR, "fastText", "mli_{}_{}_v1_{}.txt".format(SPLIT, "w_premise" if INCL_PREMISE else "no_premise", "sep" if SEP != "None" else ""))
    TXT_FILE = os.path.join(DATA_DIR, "fastText",
                            "mli_{}_{}_v1_{}.txt".format(SPLIT, "w_premise" if INCL_PREMISE else "no_premise",
                                                         "sep" if SEP != "None" else ""))
    create_nli_file(split=SPLIT, mli_file=MEDNLI_JSON_FILE, nli_file=TXT_FILE, lower=True, incl_premise=INCL_PREMISE, separator="ñ")
